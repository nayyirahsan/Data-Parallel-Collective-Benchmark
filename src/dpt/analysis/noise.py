"""Where can this apparatus support tail statistics?

H2 is a claim about p99. A p99 measurement is only meaningful if the apparatus's
own tail is small relative to the effect being injected. This maps the natural
p99/p50 ratio of *clean* runs across the experiment space and reports the region
where tail claims are admissible.

It also reports between-trial p99 stability: a p99 that swings by 3x across
process launches is not a statistic, whatever its pooled value looks like.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_rows

# A tail claim needs headroom between the apparatus's own noise and the injected
# effect. 1.5x is the working threshold: below it, a 2x injected degradation is
# still several times the natural tail and remains legible.
TAIL_THRESHOLD = 1.5
# p99 must also be stable across process launches to be worth reporting.
STABILITY_THRESHOLD = 1.6


@dataclass
class NoiseProfile:
    algo: str
    world_size: int
    nbytes: int
    beta: float
    p50_ns: float
    p99_ns: float
    tail_ratio: float
    trial_p99_spread: float
    spin_ms_per_collective: float

    @property
    def usable(self) -> bool:
        return (
            self.tail_ratio <= TAIL_THRESHOLD
            and self.trial_p99_spread <= STABILITY_THRESHOLD
        )


def profile(path: Path) -> list[NoiseProfile]:
    from .fit import steps, step_bytes

    groups = defaultdict(lambda: defaultdict(list))
    for row in read_rows(path):
        if float(row["jitter_q"]) != 0.0:
            continue
        key = (
            row["algo"],
            int(row["world_size"]),
            int(row["nbytes"]),
            float(row["beta_ns_per_byte"]),
            float(row["alpha_ns"]),
        )
        groups[key][int(row["trial"])].append(float(row["duration_ns"]))

    out = []
    for (algo, p, nbytes, beta, alpha), trials in sorted(groups.items()):
        pooled = [v for vs in trials.values() for v in vs]
        per_trial_p99 = [float(np.percentile(v, 99)) for v in trials.values() if len(v) > 10]
        n_steps = steps(algo, p)
        spin_ns = n_steps * (alpha + step_bytes(algo, nbytes, p) * beta)
        out.append(
            NoiseProfile(
                algo=algo,
                world_size=p,
                nbytes=nbytes,
                beta=beta,
                p50_ns=float(np.median(pooled)),
                p99_ns=float(np.percentile(pooled, 99)),
                tail_ratio=float(np.percentile(pooled, 99)) / float(np.median(pooled)),
                trial_p99_spread=(max(per_trial_p99) / min(per_trial_p99))
                if len(per_trial_p99) > 1
                else float("nan"),
                spin_ms_per_collective=spin_ns / 1e6,
            )
        )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--critpath", type=Path, nargs="+",
        default=[Path("results/noise_floor_critpath.csv")],
    )
    ap.add_argument(
        "--require-usable-at", type=int, metavar="P",
        help="exit non-zero unless the configurations at world size P that the "
             "tail experiment actually uses are usable. Use this as a gate "
             "before spending time on a p99 experiment the machine cannot "
             "support.",
    )
    ap.add_argument(
        "--gate-beta", type=float, default=None,
        help="restrict the gate to this beta (default: all)",
    )
    ap.add_argument(
        "--gate-min-bytes", type=int, default=0,
        help="restrict the gate to messages at least this large (default: all)",
    )
    args = ap.parse_args()

    profiles = [pr for path in args.critpath for pr in profile(path)]

    print("Apparatus tail-noise floor (clean runs, no injected jitter)")
    print("=" * 88)
    print(
        f"{'algo':>6} {'p':>2} {'N':>8} {'beta':>5} {'spin/coll':>10} "
        f"{'p50':>8} {'p99/p50':>8} {'trial p99 spread':>17} {'usable':>7}"
    )
    print("-" * 88)
    for pr in sorted(profiles, key=lambda x: (x.beta, x.world_size, x.nbytes, x.algo)):
        print(
            f"{pr.algo:>6} {pr.world_size:>2} {pr.nbytes // 1024:>6}KB {pr.beta:>5.2f} "
            f"{pr.spin_ms_per_collective:>8.2f}ms {pr.p50_ns / 1e6:>7.2f}ms "
            f"{pr.tail_ratio:>7.2f}x {pr.trial_p99_spread:>16.2f}x "
            f"{'yes' if pr.usable else 'NO':>7}"
        )

    usable = [p for p in profiles if p.usable]
    print()
    print(
        f"Usable for tail statistics: {len(usable)}/{len(profiles)} configurations "
        f"(p99/p50 <= {TAIL_THRESHOLD}, trial spread <= {STABILITY_THRESHOLD})"
    )
    if usable:
        # Listed as explicit (p, N) pairs. An earlier version printed the
        # marginal sets -- "p in [2,4,8], N in [4KB,64KB,1024KB]" -- which reads
        # as a cross product and is wrong: at p=8, beta=0 only one of six sizes
        # actually passes. That presentation caused a real misreading.
        print("  usable configurations (algo, p, N, beta):")
        for pr in sorted(usable, key=lambda x: (x.beta, x.world_size, x.nbytes)):
            print(
                f"    {pr.algo:>10}  p={pr.world_size}  "
                f"N={pr.nbytes // 1024}KB  beta={pr.beta:g}"
            )

    print()
    print("Correlation between injected spin load and tail noise:")
    spins = np.array([p.spin_ms_per_collective for p in profiles])
    tails = np.array([p.tail_ratio for p in profiles])
    print(f"  Pearson r = {np.corrcoef(spins, tails)[0, 1]:.3f} (n={len(profiles)})")

    if args.require_usable_at is not None:
        target = args.require_usable_at
        at_p = [
            p for p in profiles
            if p.world_size == target
            and p.nbytes >= args.gate_min_bytes
            and (args.gate_beta is None or p.beta == args.gate_beta)
        ]
        print()
        scope = f"p={target}"
        if args.gate_beta is not None:
            scope += f", beta={args.gate_beta:g}"
        if args.gate_min_bytes:
            scope += f", N>={args.gate_min_bytes // 1024}KB"
        if not at_p:
            print(f"GATE FAILED: no configurations measured at {scope}")
            print("  Widen experiments/noise_floor.yaml to cover the operating")
            print("  point the tail experiment will actually use.")
            raise SystemExit(2)
        bad = [p for p in at_p if not p.usable]
        if bad:
            print(f"GATE FAILED: {len(bad)}/{len(at_p)} configurations at {scope} "
                  f"are too noisy for a tail statistic.")
            for pr in sorted(bad, key=lambda p: -p.tail_ratio):
                reasons = []
                if pr.tail_ratio > TAIL_THRESHOLD:
                    reasons.append(
                        f"p99/p50 {pr.tail_ratio:.2f}x > {TAIL_THRESHOLD}"
                    )
                if pr.trial_p99_spread > STABILITY_THRESHOLD:
                    reasons.append(
                        f"p99 varies {pr.trial_p99_spread:.2f}x across launches "
                        f"> {STABILITY_THRESHOLD}"
                    )
                print(
                    f"  {pr.algo} N={pr.nbytes // 1024}KB beta={pr.beta:g}: "
                    + "; ".join(reasons)
                )
            print()
            print("  A p99 experiment here would partly measure the apparatus.")
            print("  Unstable p99 across launches usually means too few cores for")
            print("  p spin-waiting workers; a high p99/p50 at low p can also mean")
            print("  too few iterations to estimate a tail. Add cores, or raise")
            print("  'iters' in the experiment config.")
            raise SystemExit(1)
        print(f"GATE PASSED: all {len(at_p)} configurations at {scope} "
              f"support a tail statistic.")


if __name__ == "__main__":
    main()
