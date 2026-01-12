"""Validate the discrete-event model against measured degradation.

A simulator that has not been checked against reality is an opinion. This
compares simulated p99 degradation to measured p99 degradation at every setting
where both exist, and -- critically -- reports agreement *separately* for
configurations the noise-floor experiment marked usable and those it did not.
Agreement in the unusable region would be meaningless, since the measurement
there is dominated by apparatus contention rather than by the effect.

The verdict this produces is deliberately narrow: it states which (algo, p)
combinations the model may be trusted for, and the rest are reported as
indicative only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_rows

from ..simulate import SimConfig, degradation
from .noise import STABILITY_THRESHOLD, TAIL_THRESHOLD

AGREEMENT_THRESHOLD = 0.25  # within 25% counts as agreement


@dataclass
class Comparison:
    algo: str
    world_size: int
    nbytes: int
    jitter_q: float
    measured_ratio: float
    simulated_ratio: float
    apparatus_usable: bool

    @property
    def relative_error(self) -> float:
        return abs(self.simulated_ratio - self.measured_ratio) / self.measured_ratio

    @property
    def agrees(self) -> bool:
        return self.relative_error <= AGREEMENT_THRESHOLD


def _measured_degradation(path: Path) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_rows(path):
        key = (
            row["algo"], int(row["world_size"]), int(row["nbytes"]),
            float(row["jitter_q"]), float(row["jitter_k"]),
            float(row["alpha_ns"]), float(row["beta_ns_per_byte"]),
        )
        groups[key][int(row["trial"])].append(float(row["duration_ns"]))

    pooled = {k: [v for vs in t.values() for v in vs] for k, t in groups.items()}
    tails = {}
    out = {}
    for key, vals in pooled.items():
        algo, p, n, q, k, a, b = key
        clean_key = (algo, p, n, 0.0, 1.0, a, b)
        if q == 0.0:
            tails[(algo, p, n, a, b)] = float(
                np.percentile(vals, 99) / np.median(vals)
            )
            continue
        if clean_key not in pooled:
            continue
        clean = pooled[clean_key]
        out[key] = {
            "ratio": float(np.percentile(vals, 99) / np.percentile(clean, 99)),
            "clean_tail": None,
        }
    for key in out:
        algo, p, n, q, k, a, b = key
        out[key]["clean_tail"] = tails.get((algo, p, n, a, b))
    return out


def compare(path: Path, iters: int = 200_000) -> list[Comparison]:
    measured = _measured_degradation(path)
    out = []
    for key, info in sorted(measured.items()):
        algo, p, nbytes, q, k, alpha, beta = key
        if algo == "recursive" and (p & (p - 1)):
            continue
        sim = degradation(
            SimConfig(algo, p, nbytes, alpha, beta, q, k), iters=iters
        )
        tail = info["clean_tail"]
        out.append(
            Comparison(
                algo=algo,
                world_size=p,
                nbytes=nbytes,
                jitter_q=q,
                measured_ratio=info["ratio"],
                simulated_ratio=sim["p99_ratio"],
                apparatus_usable=tail is not None and tail <= TAIL_THRESHOLD,
            )
        )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--critpath", type=Path, default=Path("results/h2_jitter_critpath.csv")
    )
    args = ap.parse_args()

    comparisons = compare(args.critpath)

    print("Simulator validation: simulated vs measured p99 degradation")
    print("=" * 84)
    print(
        f"{'algo':>10} {'p':>2} {'N':>8} {'q':>6} {'measured':>9} {'simulated':>10} "
        f"{'error':>8} {'apparatus':>10} {'agrees':>7}"
    )
    print("-" * 84)
    for c in sorted(comparisons, key=lambda x: (x.world_size, x.algo, x.nbytes, x.jitter_q)):
        print(
            f"{c.algo:>10} {c.world_size:>2} {c.nbytes // 1024:>6}KB {c.jitter_q:>6g} "
            f"{c.measured_ratio:>8.2f}x {c.simulated_ratio:>9.2f}x "
            f"{c.relative_error * 100:>7.0f}% "
            f"{'clean' if c.apparatus_usable else 'noisy':>10} "
            f"{'yes' if c.agrees else 'no':>7}"
        )

    clean = [c for c in comparisons if c.apparatus_usable]
    noisy = [c for c in comparisons if not c.apparatus_usable]

    print()
    print("Verdict")
    print("-" * 84)
    for label, group in (("apparatus clean", clean), ("apparatus noisy", noisy)):
        if not group:
            print(f"  {label}: no configurations")
            continue
        agree = sum(c.agrees for c in group)
        errs = [c.relative_error * 100 for c in group]
        print(
            f"  {label}: {agree}/{len(group)} agree within "
            f"{AGREEMENT_THRESHOLD:.0%}, median error {np.median(errs):.0f}%"
        )
    if clean:
        trusted = sorted({(c.algo, c.world_size) for c in clean if c.agrees})
        print()
        print(f"  Model validated for: {trusted if trusted else 'nothing'}")
        print(
            "  Everywhere else the model is indicative only -- either the "
            "measurement is\n  contention-dominated or the model disagrees with it."
        )


if __name__ == "__main__":
    main()
