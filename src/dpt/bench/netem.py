"""Validate the transport shim against a real shaped link (`tc netem`).

What this answers
-----------------
Every result in this project rests on ``ShapedTransport`` charging alpha + N*beta
being a faithful stand-in for a network that actually has that latency and
bandwidth. That is an argument, not a measurement, until the two are compared
directly. This does the comparison on Linux, where `tc netem` can shape the
loopback interface.

Why the mapping is fitted, not assumed
--------------------------------------
`tc netem delay D` delays every packet on egress. On loopback both the data
packet and its TCP ACK traverse that qdisc, so the per-message cost could be D
or 2D depending on whether the sender's completion waits on the ACK. Rather than
assume a factor, the script sweeps netem's delay and *fits* the slope of
measured per-step cost against configured delay. A slope near 1 means the shim's
semantics match netem directly; near 2 means the shim's alpha corresponds to
half a netem delay. Either is a usable result -- an unfitted guess is not.

The same is done for bandwidth: netem `rate` is swept and the effective beta
recovered from the slope of per-step cost against message size.

Run inside Linux with NET_ADMIN (see docs/netem-validation.md):

    docker run --rm --cap-add=NET_ADMIN -v "$PWD/results:/work/results" dpt-netem
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..collectives import ALGORITHMS
from ..transport.shaped import ShapedTransport
from .launch import run_ranks

IFACE = "lo"
# Small messages make the per-byte term negligible, isolating latency.
LATENCY_PROBE_BYTES = 4096
# Large messages make the fixed term negligible, isolating bandwidth.
BANDWIDTH_PROBE_BYTES = [1 << 18, 1 << 20, 1 << 22]
# Both cost curves must be linear in the configured delay for the alpha-beta
# abstraction to hold. The slope *ratio* is only a unit conversion.
LINEARITY_THRESHOLD = 0.98


# --------------------------------------------------------------------------
# tc plumbing
# --------------------------------------------------------------------------
def tc_available() -> tuple[bool, str]:
    if platform.system() != "Linux":
        return False, f"tc netem requires Linux, this is {platform.system()}"
    if shutil.which("tc") is None:
        return False, "tc not found (install iproute2)"
    probe = subprocess.run(
        ["tc", "qdisc", "show", "dev", IFACE], capture_output=True, text=True
    )
    if probe.returncode != 0:
        return False, f"tc cannot read qdiscs: {probe.stderr.strip()}"
    return True, "ok"


def tc_clear() -> None:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", IFACE, "root"],
        capture_output=True, text=True,
    )  # no-op if nothing is attached


def tc_apply(delay_us: float | None = None, rate_mbit: float | None = None) -> None:
    """Attach a netem qdisc to loopback. Clears any existing one first."""
    tc_clear()
    if delay_us is None and rate_mbit is None:
        return
    cmd = ["tc", "qdisc", "add", "dev", IFACE, "root", "netem"]
    if delay_us is not None:
        cmd += ["delay", f"{delay_us:g}us"]
    if rate_mbit is not None:
        cmd += ["rate", f"{rate_mbit:g}mbit"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tc failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def tc_describe() -> str:
    out = subprocess.run(
        ["tc", "qdisc", "show", "dev", IFACE], capture_output=True, text=True
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def _probe_job(transport, algo: str, nbytes: int, alpha_ns: float, beta: float, iters: int):
    shaped = ShapedTransport(transport, alpha_ns=alpha_ns, beta_ns_per_byte=beta)
    base = torch.ones(max(1, nbytes // 4), dtype=torch.float32)
    for _ in range(4):
        ALGORITHMS[algo](base.clone(), shaped)
    times = []
    for i in range(iters):
        shaped.set_step(i)
        tensor = base.clone()
        transport.barrier()
        t0 = time.perf_counter_ns()
        ALGORITHMS[algo](tensor, shaped)
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times)


def measure(
    nbytes: int, alpha_ns: float = 0.0, beta: float = 0.0,
    world_size: int = 2, iters: int = 40, trials: int = 3,
) -> float:
    """Median per-step cost of a ring all-reduce, in nanoseconds.

    Ring at p=2 runs exactly 2 steps, so dividing by 2 gives per-step cost
    directly with no model in between.
    """
    per_trial = []
    for _ in range(trials):
        results = run_ranks(
            world_size, _probe_job, "ring", nbytes, alpha_ns, beta, iters,
            timeout=600,
        )
        per_trial.append(max(results) / (2 * (world_size - 1)))
    return float(np.median(per_trial))


def _fit_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y = intercept + slope*x, with R^2."""
    A = np.stack([np.ones(len(xs)), np.array(xs, dtype=float)], axis=1)
    y = np.array(ys, dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(coef[1]), float(coef[0]), (1 - ss_res / ss_tot if ss_tot else float("nan"))


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    baseline_ns: float = 0.0
    shim_slope: float = 0.0
    shim_r2: float = 0.0
    netem_slope: float = 0.0
    netem_r2: float = 0.0
    shim_beta: dict = field(default_factory=dict)
    netem_beta: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)

    @property
    def delay_factor(self) -> float:
        """netem microseconds per shim microsecond of effective latency."""
        return self.netem_slope / self.shim_slope if self.shim_slope else float("nan")


def validate_latency(report: ValidationReport, delays_us: list[float], iters: int) -> None:
    print("\n1. Latency: shim α sweep (no netem)")
    print(f"   {'α (µs)':>9} {'per-step (µs)':>15}")
    xs, ys = [], []
    for d in delays_us:
        tc_clear()
        ns = measure(LATENCY_PROBE_BYTES, alpha_ns=d * 1000, iters=iters)
        xs.append(d)
        ys.append(ns / 1000)
        report.rows.append({"mode": "shim", "delay_us": d, "per_step_us": ns / 1000})
        print(f"   {d:>9.0f} {ns / 1000:>15.1f}")
    report.shim_slope, intercept, report.shim_r2 = _fit_slope(xs, ys)
    print(f"   slope {report.shim_slope:.3f} µs/µs, intercept {intercept:.1f} µs, R²={report.shim_r2:.4f}")

    print("\n2. Latency: tc netem delay sweep (shim disabled)")
    print(f"   {'delay (µs)':>11} {'per-step (µs)':>15}")
    xs, ys = [], []
    for d in delays_us:
        tc_apply(delay_us=d)
        ns = measure(LATENCY_PROBE_BYTES, alpha_ns=0.0, iters=iters)
        xs.append(d)
        ys.append(ns / 1000)
        report.rows.append({"mode": "netem", "delay_us": d, "per_step_us": ns / 1000})
        print(f"   {d:>11.0f} {ns / 1000:>15.1f}")
    tc_clear()
    report.netem_slope, intercept, report.netem_r2 = _fit_slope(xs, ys)
    print(f"   slope {report.netem_slope:.3f} µs/µs, intercept {intercept:.1f} µs, R²={report.netem_r2:.4f}")


def validate_bandwidth(report: ValidationReport, rates_mbit: list[float], iters: int) -> None:
    print("\n3. Bandwidth: effective β from the size sweep")
    print(f"   {'mode':>14} {'nominal β':>11} {'fitted β':>10} {'error':>8}")
    for rate in rates_mbit:
        nominal_beta = 8000.0 / rate  # ns/byte from Mbit/s

        tc_clear()
        xs, ys = [], []
        for n in BANDWIDTH_PROBE_BYTES:
            xs.append(n / 2)  # ring at p=2 moves N/2 per step
            ys.append(measure(n, beta=nominal_beta, iters=iters))
        shim_beta, _, _ = _fit_slope(xs, ys)
        report.shim_beta[rate] = shim_beta

        tc_apply(rate_mbit=rate)
        xs, ys = [], []
        for n in BANDWIDTH_PROBE_BYTES:
            xs.append(n / 2)
            ys.append(measure(n, beta=0.0, iters=iters))
        netem_beta, _, _ = _fit_slope(xs, ys)
        report.netem_beta[rate] = netem_beta
        tc_clear()

        for label, val in (("shim", shim_beta), ("netem", netem_beta)):
            err = abs(val - nominal_beta) / nominal_beta * 100
            print(f"   {label:>14} {nominal_beta:>10.3f}  {val:>9.3f} {err:>7.0f}%")
        report.rows.append(
            {"mode": "bandwidth", "rate_mbit": rate, "nominal_beta": nominal_beta,
             "shim_beta": shim_beta, "netem_beta": netem_beta}
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays-us", type=float, nargs="+", default=[0, 250, 500, 1000, 2000])
    ap.add_argument("--rates-mbit", type=float, nargs="+", default=[1000, 2000])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("results/netem_validation.json"))
    ap.add_argument("--skip-bandwidth", action="store_true")
    args = ap.parse_args()

    ok, reason = tc_available()
    if not ok:
        raise SystemExit(
            f"cannot run: {reason}\n\n"
            "This experiment needs Linux with NET_ADMIN. See docs/netem-validation.md\n"
            "  docker run --rm --cap-add=NET_ADMIN -v \"$PWD/results:/work/results\" dpt-netem"
        )
    if os.geteuid() != 0:
        print("warning: not running as root; tc may refuse to attach qdiscs")

    print("tc netem validation of the transport shim")
    print("=" * 60)
    print(f"interface: {IFACE}   initial qdisc: {tc_describe()}")

    report = ValidationReport()
    try:
        report.baseline_ns = measure(LATENCY_PROBE_BYTES, iters=args.iters)
        print(f"unshaped baseline per-step: {report.baseline_ns / 1000:.1f} µs")
        validate_latency(report, args.delays_us, args.iters)
        if not args.skip_bandwidth:
            validate_bandwidth(report, args.rates_mbit, args.iters)
    finally:
        tc_clear()

    print("\n" + "=" * 60)
    print("VERDICT")
    print("-" * 60)
    print(f"  shim:  {report.shim_slope:.3f} us per us configured  (R^2={report.shim_r2:.4f})")
    print(f"  netem: {report.netem_slope:.3f} us per us configured  (R^2={report.netem_r2:.4f})")
    print(f"  ratio: {report.delay_factor:.2f} netem-us per shim-us")
    print()

    # What validates the shim is that BOTH cost curves are linear in the
    # configured delay. The ratio between their slopes is a unit conversion --
    # it says what physical link a given alpha corresponds to, not whether the
    # abstraction holds. On loopback a packet can cross the egress qdisc on both
    # the send and receive path, so factors of 2 and 4 are both expected; a
    # direct RTT probe on this container measured ~4x. Hard-coding an expected
    # factor would have turned a unit conversion into a spurious failure.
    linear = min(report.shim_r2, report.netem_r2)
    if linear >= LINEARITY_THRESHOLD:
        print(f"  PASS: both costs are linear in configured delay (R^2 >= {LINEARITY_THRESHOLD}).")
        print( "  The alpha-beta abstraction describes this transport, and the shim")
        print( "  is faithful under the unit conversion below.")
        print()
        print(f"    1 us of shim alpha == {report.delay_factor:.2f} us of tc netem delay")
        print()
        print( "  Record this factor in docs/decisions.md and restate the findings'")
        print(f"  alpha values as equivalent netem delays by dividing by {report.delay_factor:.2f}.")
    else:
        worse = "shim" if report.shim_r2 < report.netem_r2 else "netem"
        print(f"  FAIL: the {worse} cost curve is not linear in configured delay")
        print(f"  (R^2={linear:.4f} < {LINEARITY_THRESHOLD}). The alpha-beta model does not")
        print( "  describe this transport, which is itself a reportable finding --")
        print( "  inspect the per-point tables above before trusting absolute alpha.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {
            "baseline_ns": report.baseline_ns,
            "shim_slope": report.shim_slope, "shim_r2": report.shim_r2,
            "netem_slope": report.netem_slope, "netem_r2": report.netem_r2,
            "delay_factor": report.delay_factor,
            "shim_beta": report.shim_beta, "netem_beta": report.netem_beta,
            "rows": report.rows,
        }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
