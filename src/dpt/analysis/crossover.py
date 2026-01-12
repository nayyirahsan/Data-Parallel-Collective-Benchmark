"""Measured vs predicted crossover points -- the H1 test.

Method
------
For a fixed (algo, p, alpha, beta) the predicted cost is linear in message size:

    duration(N) = steps * (alpha + a_sw) + critpath_bytes(N) * (beta + b_sw)

and critpath_bytes is itself linear in N, so each arm is a straight line in N.
Rather than reading a crossover off a plot -- which four discrete message sizes
could not support anyway -- each arm is fitted as ``duration = c0 + c1 * N`` and
the crossover solved analytically as the intersection. Confidence intervals come
from bootstrapping over *trials* (independent process launches), so they carry
launch-to-launch variance rather than only steady-state jitter.

Two predictions are compared against the measurement:

* **textbook** -- the alpha-beta model as published, with no software term.
* **corrected** -- the same model with the per-p floor (a_sw, b_sw) fitted
  independently from the alpha = beta = 0 experiment.

H1 claims the textbook model underestimates the crossover by >=5x.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_rows

from ..model import CostModel, crossover_bytes
from .fit import FloorFit, fit_floor


@dataclass
class LineFit:
    intercept_ns: float
    slope_ns_per_byte: float


@dataclass
class CrossoverResult:
    world_size: int
    alpha_ns: float
    beta_ns_per_byte: float
    algo_a: str
    algo_b: str
    measured_ns: float | None
    ci_low: float | None
    ci_high: float | None
    textbook_ns: float | None
    corrected_ns: float | None
    max_measured_bytes: int = 0

    @property
    def extrapolated(self) -> bool:
        """True when N* falls outside the swept range, so the fit is extrapolating."""
        return (
            self.measured_ns is not None
            and self.max_measured_bytes > 0
            and self.measured_ns > self.max_measured_bytes
        )

    @property
    def textbook_ratio(self) -> float | None:
        if self.measured_ns is None or not self.textbook_ns:
            return None
        if self.textbook_ns < 1024:  # degenerate: model says ring wins from ~0 bytes
            return None
        return self.measured_ns / self.textbook_ns

    @property
    def corrected_ratio(self) -> float | None:
        if self.measured_ns is None or not self.corrected_ns:
            return None
        return self.measured_ns / self.corrected_ns


def load_critpath(path: Path) -> dict[tuple, dict[int, list[float]]]:
    """(algo, p, alpha, beta) -> trial -> per-(N,iter) durations, keyed by N."""
    out: dict[tuple, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in read_rows(path):
        key = (
            row["algo"],
            int(row["world_size"]),
            float(row["alpha_ns"]),
            float(row["beta_ns_per_byte"]),
        )
        out[key][int(row["trial"])][int(row["nbytes"])].append(
            float(row["duration_ns"])
        )
    return out


def _fit_line(by_size: dict[int, float]) -> LineFit | None:
    if len(by_size) < 2:
        return None
    sizes = np.array(sorted(by_size), dtype=float)
    times = np.array([by_size[int(s)] for s in sizes], dtype=float)
    A = np.stack([np.ones_like(sizes), sizes], axis=1)
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    return LineFit(float(coef[0]), float(coef[1]))


def _intersect(a: LineFit, b: LineFit) -> float | None:
    denom = a.slope_ns_per_byte - b.slope_ns_per_byte
    if abs(denom) < 1e-12:
        return None
    n = (b.intercept_ns - a.intercept_ns) / denom
    return n if n > 0 else None


def _trial_medians(trials: dict[int, dict[int, list[float]]]) -> dict[int, dict[int, float]]:
    return {
        t: {n: float(np.median(v)) for n, v in by_n.items()} for t, by_n in trials.items()
    }


def measured_crossover(
    data, algo_a: str, algo_b: str, p: int, alpha: float, beta: float,
    bootstrap: int = 2000, seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Crossover N* with a bootstrap CI over trials."""
    key_a, key_b = (algo_a, p, alpha, beta), (algo_b, p, alpha, beta)
    if key_a not in data or key_b not in data:
        return None, None, None

    med_a, med_b = _trial_medians(data[key_a]), _trial_medians(data[key_b])
    trials = sorted(set(med_a) & set(med_b))
    if len(trials) < 2:
        return None, None, None

    def solve(subset: list[int]) -> float | None:
        pooled_a = {
            n: float(np.median([med_a[t][n] for t in subset if n in med_a[t]]))
            for n in med_a[subset[0]]
        }
        pooled_b = {
            n: float(np.median([med_b[t][n] for t in subset if n in med_b[t]]))
            for n in med_b[subset[0]]
        }
        fa, fb = _fit_line(pooled_a), _fit_line(pooled_b)
        if fa is None or fb is None:
            return None
        return _intersect(fa, fb)

    point = solve(trials)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        subset = list(rng.choice(trials, size=len(trials), replace=True))
        val = solve(subset)
        if val is not None:
            samples.append(val)
    if len(samples) < bootstrap * 0.5:
        return point, None, None
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def analyse(
    critpath: Path, floors: dict[int, FloorFit], algo_a: str = "ring", algo_b: str = "ps"
) -> list[CrossoverResult]:
    data = load_critpath(critpath)
    combos = sorted({(p, a, b) for _, p, a, b in data})
    max_bytes = max(
        n for trials in data.values() for by_n in trials.values() for n in by_n
    )

    results = []
    for p, alpha, beta in combos:
        point, lo, hi = measured_crossover(data, algo_a, algo_b, p, alpha, beta)
        textbook = CostModel(alpha_ns=alpha, beta_ns_per_byte=beta)
        floor = floors.get(p)
        corrected = (
            CostModel(
                alpha_ns=alpha,
                beta_ns_per_byte=beta + floor.b_sw_ns_per_byte,
                alpha_sw_ns=floor.a_sw_ns,
            )
            if floor
            else None
        )
        results.append(
            CrossoverResult(
                world_size=p,
                alpha_ns=alpha,
                beta_ns_per_byte=beta,
                algo_a=algo_a,
                algo_b=algo_b,
                measured_ns=point,
                ci_low=lo,
                ci_high=hi,
                textbook_ns=crossover_bytes(textbook, algo_a, algo_b, p),
                corrected_ns=crossover_bytes(corrected, algo_a, algo_b, p)
                if corrected
                else None,
                max_measured_bytes=max_bytes,
            )
        )
    return results


def _fmt(nbytes: float | None) -> str:
    if nbytes is None:
        return "     --"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):6.2f}MB"
    return f"{nbytes / 1024:6.1f}KB"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--critpath", type=Path, default=Path("results/h1_grid_critpath.csv"))
    ap.add_argument("--floor", type=Path, default=Path("results/floor_critpath.csv"))
    ap.add_argument("--a", default="ring")
    ap.add_argument("--b", default="ps")
    args = ap.parse_args()

    floors = fit_floor(args.floor)
    results = analyse(args.critpath, floors, args.a, args.b)

    print(f"Crossover N* : {args.a} becomes cheaper than {args.b}")
    print("=" * 96)
    print(
        f"{'p':>2} {'alpha':>8} {'a/a_sw':>7} {'beta':>6} {'measured':>9} "
        f"{'95% CI':>21} {'textbook':>9} {'corrected':>9} {'tb err':>7} {'corr err':>8}"
    )
    print("-" * 96)
    for r in results:
        floor = floors.get(r.world_size)
        ratio_sw = r.alpha_ns / floor.a_sw_ns if floor else float("nan")
        ci = (
            f"[{_fmt(r.ci_low).strip()}, {_fmt(r.ci_high).strip()}]"
            if r.ci_low is not None
            else "--"
        )
        tb, corr = r.textbook_ratio, r.corrected_ratio
        flag = " *" if r.extrapolated else "  "
        print(
            f"{r.world_size:>2} {r.alpha_ns / 1000:>6.0f}us {ratio_sw:>7.2f} "
            f"{r.beta_ns_per_byte:>6.2f} {_fmt(r.measured_ns):>9}{flag}{ci:>21} "
            f"{_fmt(r.textbook_ns):>9} {_fmt(r.corrected_ns):>9} "
            f"{(f'{tb:.2f}x' if tb else '--'):>7} {(f'{corr:.2f}x' if corr else '--'):>8}"
        )
    print()
    print("  * crossover lies beyond the largest measured size: extrapolated, treat as a bound")
    print("  '--' textbook: model predicts no crossover in range (it says one arm always wins)")

    usable = [r for r in results if r.textbook_ratio and not r.extrapolated]
    if usable:
        print()
        print("H1 summary over non-extrapolated points with a defined textbook crossover")
        print("-" * 96)
        tb_errs = [r.textbook_ratio for r in usable]
        corr_errs = [abs(r.corrected_ratio - 1) * 100 for r in usable if r.corrected_ratio]
        print(f"  textbook  underestimates by {min(tb_errs):.2f}x - {max(tb_errs):.2f}x "
              f"(n={len(tb_errs)})")
        if corr_errs:
            print(f"  corrected within {min(corr_errs):.0f}% - {max(corr_errs):.0f}% "
                  f"(median {sorted(corr_errs)[len(corr_errs)//2]:.0f}%)")


if __name__ == "__main__":
    main()
