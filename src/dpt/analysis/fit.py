"""Fitting the software-overhead floor.

The floor is fitted as a per-step cost with a fixed and a per-byte term:

    per_step_time = a_sw + step_bytes * b_sw

which is deliberately the same structure as ``dpt.model.CostModel``, so the
fitted terms slot straight in as ``alpha_sw_ns`` and an addition to beta. The
fit comes from an independent alpha=beta=0 experiment rather than being tuned to
the data it is later tested against.

Why the floor is fitted per world size
--------------------------------------
A single global fit gives R^2 = 0.65 with residuals up to 213%, because both
terms grow with p: a_sw goes 85 -> 140 -> 256 us and effective bandwidth falls
87 -> 53 -> 40 Gbit/s across p = 2, 4, 8.

Decomposing per-rank timings separates the two candidate causes. Taking the max
over ranks rather than the mean costs only 1.02-1.11x, and that gap barely grows
with p -- so this is *not* order statistics. The per-rank *mean* itself grows
1.55-1.99x from p=2 to p=8, which is resource contention between co-located
processes sharing memory bandwidth and cores.

That contention is an artefact of simulating workers on one machine and would
not appear on a real cluster. It is therefore absorbed into a per-p floor rather
than left to contaminate the injected-alpha comparison: the floor is a property
of the machine *at a given co-location level*, measured independently for each.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_rows


def steps(algo: str, p: int) -> int:
    """Number of messages on the critical path."""
    if p < 2:
        return 0
    return {
        "ring": 2 * (p - 1),
        "recursive": 2 * int(math.log2(p)),
        "recursive_fresh": 2 * int(math.log2(p)),
        "ps": p,  # one worker send, then a serial (p-1) broadcast
    }[algo]


def critpath_bytes(algo: str, nbytes: float, p: int) -> float:
    """Total bytes traversing the critical path."""
    if p < 2:
        return 0.0
    if algo == "ring":
        return 2 * (p - 1) * nbytes / p
    if algo in ("recursive", "recursive_fresh"):
        return 2 * nbytes * (1 - 1 / p)
    if algo == "ps":
        return p * nbytes
    raise KeyError(algo)


def step_bytes(algo: str, nbytes: float, p: int) -> float:
    n = steps(algo, p)
    return critpath_bytes(algo, nbytes, p) / n if n else 0.0


@dataclass
class FloorFit:
    world_size: int
    a_sw_ns: float
    b_sw_ns_per_byte: float
    r_squared: float
    n_points: int

    @property
    def effective_gbps(self) -> float:
        return 8.0 / self.b_sw_ns_per_byte if self.b_sw_ns_per_byte > 0 else float("inf")


def load_medians(path: Path) -> dict[tuple[str, int, int], float]:
    """Median critical-path duration per (algo, world_size, nbytes)."""
    groups = defaultdict(list)
    for row in read_rows(path):
        key = (row["algo"], int(row["world_size"]), int(row["nbytes"]))
        groups[key].append(float(row["duration_ns"]))
    return {k: float(np.median(v)) for k, v in groups.items()}


def fit_floor(path: Path) -> dict[int, FloorFit]:
    """Fit (a_sw, b_sw) independently for each world size."""
    medians = load_medians(path)
    by_p: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (algo, p, nbytes), duration in medians.items():
        if p < 2:
            continue
        by_p[p].append((step_bytes(algo, nbytes, p), duration / steps(algo, p)))

    fits = {}
    for p, points in sorted(by_p.items()):
        A = np.array([[1.0, sb] for sb, _ in points])
        y = np.array([t for _, t in points])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        # Negative fixed or per-byte cost is physically meaningless and would
        # signal a misspecified step model rather than a fit.
        coef = np.maximum(coef, 0.0)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        fits[p] = FloorFit(
            world_size=p,
            a_sw_ns=float(coef[0]),
            b_sw_ns_per_byte=float(coef[1]),
            r_squared=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            n_points=len(points),
        )
    return fits


def contention_report(raw_path: Path, nbytes: int = 4096) -> list[dict]:
    """Separate order-statistics cost from resource contention.

    At small message sizes the per-byte term is negligible, so per-step time is
    essentially the fixed cost. Comparing the max over ranks against the mean
    isolates slowest-of-p gating; comparing the mean across p isolates
    contention.
    """
    groups = defaultdict(list)
    for row in read_rows(raw_path):
        if int(row["nbytes"]) != nbytes:
            continue
        key = (row["algo"], int(row["world_size"]), int(row["trial"]), int(row["iter"]))
        groups[key].append(float(row["duration_ns"]))

    agg = defaultdict(lambda: ([], []))
    for (algo, p, _, _), vals in groups.items():
        agg[(algo, p)][0].append(float(np.mean(vals)))
        agg[(algo, p)][1].append(float(np.max(vals)))

    baseline: dict[str, float] = {}
    out = []
    for (algo, p), (means, maxes) in sorted(agg.items()):
        n = steps(algo, p)
        mean_step = float(np.median(means)) / n
        max_step = float(np.median(maxes)) / n
        baseline.setdefault(algo, mean_step)
        out.append(
            {
                "algo": algo,
                "world_size": p,
                "mean_step_ns": mean_step,
                "max_step_ns": max_step,
                "order_stat_factor": max_step / mean_step,
                "contention_factor": mean_step / baseline[algo],
            }
        )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path, default=Path("results/floor_critpath.csv"))
    ap.add_argument("--raw", type=Path, default=Path("results/floor_raw.csv"))
    args = ap.parse_args()

    print("Software overhead floor, fitted per world size (alpha = beta = 0)")
    print("=" * 66)
    print(f"{'p':>3} {'a_sw':>10} {'b_sw':>11} {'effective':>12} {'R^2':>7}")
    print("-" * 66)
    for p, fit in fit_floor(args.floor).items():
        print(
            f"{p:>3} {fit.a_sw_ns / 1000:>8.1f}us {fit.b_sw_ns_per_byte:>9.3f}ns/B "
            f"{fit.effective_gbps:>9.1f}Gb/s {fit.r_squared:>7.4f}"
        )

    if args.raw.exists():
        print("\nDecomposition of the p-dependence (4KB messages)")
        print("=" * 66)
        print(f"{'algo':>10} {'p':>3} {'slowest-of-p':>14} {'contention':>12}")
        print("-" * 66)
        for row in contention_report(args.raw):
            print(
                f"{row['algo']:>10} {row['world_size']:>3} "
                f"{row['order_stat_factor']:>13.2f}x {row['contention_factor']:>11.2f}x"
            )


if __name__ == "__main__":
    main()
