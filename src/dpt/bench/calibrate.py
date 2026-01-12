"""Calibration gate: is injected delay accurate, and up to what worker count?

This is a prerequisite for every other experiment. If the shim cannot deliver a
requested alpha faithfully, the (alpha, beta) axis is not a controlled variable
and the crossover surface means nothing. Run it on any new machine before
trusting results from that machine.

    python -m dpt.bench.calibrate --out results/calibration.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from ..timing import (
    MIN_SLEEP_TARGET_NS,
    SLEEP_FRACTION,
    SLEEP_SAFETY_MARGIN,
    delay_ns,
    sleep_granularity_ns,
    spin_ns,
)

STRATEGIES = ("sleep", "spin", "hybrid", "yield")
HYBRID_SLACK_NS = 300_000


def _sleep_ns(ns: int) -> None:
    time.sleep(ns / 1e9)


def _hybrid_ns(ns: int) -> None:
    deadline = time.perf_counter_ns() + ns
    if ns > HYBRID_SLACK_NS:
        time.sleep((ns - HYBRID_SLACK_NS) / 1e9)
    while time.perf_counter_ns() < deadline:
        pass


_FN = {"sleep": _sleep_ns, "spin": spin_ns, "hybrid": _hybrid_ns,
       "yield": delay_ns}


def _measure(strategy: str, target_ns: int, reps: int) -> list[int]:
    fn = _FN[strategy]
    fn(50_000)  # warm
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn(target_ns)
        out.append(time.perf_counter_ns() - t0)
    return out


def _worker(rank, strategy, target_ns, reps, queue):
    queue.put(_measure(strategy, target_ns, reps))


def _run_concurrent(p: int, strategy: str, target_ns: int, reps: int) -> list[int]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, strategy, target_ns, reps, queue))
        for r in range(p)
    ]
    for proc in procs:
        proc.start()
    samples = [v for _ in range(p) for v in queue.get()]
    for proc in procs:
        proc.join()
    return sorted(samples)


def _summarise(samples: list[int], target_ns: int) -> dict:
    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    median = statistics.median(samples)
    return {
        "median_ns": median,
        "median_err_pct": (median - target_ns) / target_ns * 100,
        "p99_ns": p99,
        "p99_err_pct": (p99 - target_ns) / target_ns * 100,
        "max_ns": samples[-1],
        "n": len(samples),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/calibration.csv"))
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument(
        "--targets-ns",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000],
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The sleep leg of the core-yielding delay is only usable if the platform can
    # deliver a sleep that short. Reported first because it determines whether
    # 'yield' behaves differently from 'spin' at all on this machine, and hence
    # how many cores a given worker count needs.
    granularity = sleep_granularity_ns()
    engages_above = granularity * SLEEP_SAFETY_MARGIN / SLEEP_FRACTION
    print(f"sleep granularity: {granularity / 1000:.1f}us")
    print(f"core-yielding engages for delays above ~{engages_above / 1000:.0f}us")
    if engages_above > 200_000:
        print("  NOTE: above the 200us alpha the experiments use, so 'yield' will")
        print("  pure-spin here and each worker needs a full core. Budget ~3x cores")
        print("  per worker (p=8 -> 24+). A finer-timer machine would need far fewer.")
    else:
        print("  Core-yielding is active at the experiments' 200us alpha: each")
        print("  worker needs ~15% of a core rather than a full one.")
    print()

    rows = []
    print(f"{'p':>3} {'strat':>7} {'target':>10} {'median':>10} {'err%':>7} {'p99 err%':>9}")
    print("-" * 52)
    for target in args.targets_ns:
        reps = 200 if target <= 1_000_000 else 40
        for strategy in STRATEGIES:
            for p in args.workers:
                samples = _run_concurrent(p, strategy, target, reps)
                stats = _summarise(samples, target)
                rows.append(
                    {"workers": p, "strategy": strategy, "target_ns": target, **stats}
                )
                print(
                    f"{p:>3} {strategy:>7} {target/1000:>8.0f}us "
                    f"{stats['median_ns']/1000:>8.1f}us {stats['median_err_pct']:>6.1f}% "
                    f"{stats['p99_err_pct']:>8.1f}%"
                )
        print()

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
