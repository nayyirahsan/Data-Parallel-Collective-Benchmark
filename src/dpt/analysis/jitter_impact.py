"""H2 analysis: how much does injected tail latency cost each algorithm?

The claim is about *degradation*, not absolute speed, so every statistic is a
ratio against the same algorithm's own clean (q=0) run at the same (p, N, alpha,
beta). That controls for the fact that the arms have different baseline costs --
otherwise "ring is slower" would be confounded with "ring degrades more".

The headline statistic is p99, because the mechanism under test is tail
compounding: ring's 2(p-1) serial steps each gated by their slowest participant.
Medians are reported alongside, since a mechanism that moves the tail without
moving the median is evidence *for* compounding rather than a general slowdown.

Bootstrap CIs resample over trials (process launches), not iterations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_rows


@dataclass
class Degradation:
    algo: str
    world_size: int
    nbytes: int
    jitter_q: float
    jitter_k: float
    clean_median_ns: float
    clean_p99_ns: float
    jittered_median_ns: float
    jittered_p99_ns: float
    p99_ratio: float
    p99_ci: tuple[float, float] | None
    median_ratio: float
    steps: int


def _load(path: Path):
    """(algo, p, nbytes, q, k) -> trial -> durations."""
    out = defaultdict(lambda: defaultdict(list))
    for row in read_rows(path):
        key = (
            row["algo"],
            int(row["world_size"]),
            int(row["nbytes"]),
            float(row["jitter_q"]),
            float(row["jitter_k"]),
        )
        out[key][int(row["trial"])].append(float(row["duration_ns"]))
    return out


def _p99(values) -> float:
    return float(np.percentile(values, 99))


def analyse(path: Path, bootstrap: int = 2000, seed: int = 0) -> list[Degradation]:
    from .fit import steps as step_count

    data = _load(path)
    results = []
    rng = np.random.default_rng(seed)

    for key, trials in sorted(data.items()):
        algo, p, nbytes, q, k = key
        if q == 0.0:
            continue
        clean_key = (algo, p, nbytes, 0.0, 1.0)
        if clean_key not in data:
            continue

        clean_trials = data[clean_key]
        clean_all = [v for vs in clean_trials.values() for v in vs]
        jit_all = [v for vs in trials.values() for v in vs]

        clean_p99, jit_p99 = _p99(clean_all), _p99(jit_all)

        shared = sorted(set(clean_trials) & set(trials))
        samples = []
        if len(shared) >= 2:
            for _ in range(bootstrap):
                pick = rng.choice(shared, size=len(shared), replace=True)
                c = [v for t in pick for v in clean_trials[t]]
                j = [v for t in pick for v in trials[t]]
                if c and j and _p99(c) > 0:
                    samples.append(_p99(j) / _p99(c))

        results.append(
            Degradation(
                algo=algo,
                world_size=p,
                nbytes=nbytes,
                jitter_q=q,
                jitter_k=k,
                clean_median_ns=float(np.median(clean_all)),
                clean_p99_ns=clean_p99,
                jittered_median_ns=float(np.median(jit_all)),
                jittered_p99_ns=jit_p99,
                p99_ratio=jit_p99 / clean_p99,
                p99_ci=(
                    (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
                    if len(samples) > bootstrap * 0.5
                    else None
                ),
                median_ratio=float(np.median(jit_all)) / float(np.median(clean_all)),
                steps=step_count(algo, p),
            )
        )
    return results


def winner_flips(path: Path) -> list[dict]:
    """Configurations where jitter changes which algorithm is fastest."""
    data = _load(path)
    by_setting = defaultdict(dict)
    for (algo, p, nbytes, q, k), trials in data.items():
        vals = [v for vs in trials.values() for v in vs]
        by_setting[(p, nbytes, q, k)][algo] = {
            "median": float(np.median(vals)),
            "p99": _p99(vals),
        }

    flips = []
    for (p, nbytes, q, k), arms in sorted(by_setting.items()):
        if q == 0.0 or (p, nbytes, 0.0, 1.0) not in by_setting:
            continue
        clean = by_setting[(p, nbytes, 0.0, 1.0)]
        for stat in ("median", "p99"):
            clean_best = min(clean, key=lambda a: clean[a][stat])
            jit_best = min(arms, key=lambda a: arms[a][stat])
            if clean_best != jit_best:
                flips.append(
                    {
                        "world_size": p,
                        "nbytes": nbytes,
                        "jitter_q": q,
                        "jitter_k": k,
                        "statistic": stat,
                        "clean_winner": clean_best,
                        "jittered_winner": jit_best,
                    }
                )
    return flips


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--critpath", type=Path, default=Path("results/h2_jitter_critpath.csv"))
    args = ap.parse_args()

    results = analyse(args.critpath)

    print("H2: p99 degradation under injected tail jitter")
    print("=" * 94)
    print(
        f"{'algo':>10} {'p':>2} {'steps':>6} {'N':>7} {'jitter':>12} "
        f"{'p99 clean':>10} {'p99 jit':>10} {'p99 ratio':>10} {'95% CI':>15} {'median':>8}"
    )
    print("-" * 94)
    for r in sorted(results, key=lambda x: (x.nbytes, x.jitter_q, x.world_size, x.algo)):
        ci = f"[{r.p99_ci[0]:.2f},{r.p99_ci[1]:.2f}]" if r.p99_ci else "--"
        print(
            f"{r.algo:>10} {r.world_size:>2} {r.steps:>6} "
            f"{r.nbytes // 1024:>6}KB q={r.jitter_q:<4g}k={r.jitter_k:<3g} "
            f"{r.clean_p99_ns / 1e6:>9.2f}ms {r.jittered_p99_ns / 1e6:>9.2f}ms "
            f"{r.p99_ratio:>9.2f}x {ci:>15} {r.median_ratio:>7.2f}x"
        )

    print()
    print("Ring vs PS p99 degradation, paired at identical settings")
    print("-" * 94)
    idx = {(r.algo, r.world_size, r.nbytes, r.jitter_q): r for r in results}
    print(f"{'p':>2} {'N':>8} {'jitter':>12} {'ring':>9} {'ps':>9} {'recursive':>10} {'ring/ps':>9}")
    for (algo, p, nbytes, q), r in sorted(idx.items()):
        if algo != "ring":
            continue
        ps = idx.get(("ps", p, nbytes, q))
        rec = idx.get(("recursive", p, nbytes, q))
        if not ps:
            continue
        print(
            f"{p:>2} {nbytes // 1024:>7}KB q={q:<10g} {r.p99_ratio:>8.2f}x "
            f"{ps.p99_ratio:>8.2f}x {(f'{rec.p99_ratio:.2f}x' if rec else '--'):>10} "
            f"{r.p99_ratio / ps.p99_ratio:>8.2f}x"
        )

    flips = winner_flips(args.critpath)
    print()
    if flips:
        print("Winner flips caused by jitter")
        print("-" * 94)
        for f in flips:
            print(
                f"  p={f['world_size']} N={f['nbytes'] // 1024}KB "
                f"q={f['jitter_q']:g} k={f['jitter_k']:g} [{f['statistic']}]: "
                f"{f['clean_winner']} -> {f['jittered_winner']}"
            )
    else:
        print("Winner flips caused by jitter: none observed")


if __name__ == "__main__":
    main()
