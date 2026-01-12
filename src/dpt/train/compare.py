"""End-to-end comparison of collectives inside a real training loop.

The collective microbenchmarks measure an all-reduce in isolation. This measures
what fraction of an actual training step it accounts for, which is the number
that decides whether any of it matters for the workload.

    python -m dpt.train.compare --workers 4 --steps 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .resnet_cifar import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--alpha-ns", type=float, default=200_000.0)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--norm", default="gn", choices=["bn", "gn"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results/e2e_compare.json"))
    args = ap.parse_args()

    collectives = ["ring", "ps"]
    if args.workers & (args.workers - 1) == 0:
        collectives.append("recursive")

    print(
        f"ResNet-18 data-parallel, p={args.workers}, global batch {args.batch_size}, "
        f"α={args.alpha_ns / 1000:.0f}µs β={args.beta:g}"
        + (" [synthetic data]" if args.synthetic else " [CIFAR-10]")
    )
    print()
    print(
        f"{'collective':>11} {'step':>9} {'compute':>9} {'allreduce':>10} "
        f"{'pack':>8} {'transfer':>9} {'comm share':>11}"
    )
    print("-" * 70)

    summary = {}
    for collective in collectives:
        trials = []
        for trial in range(args.trials):
            results = run(
                workers=args.workers, collective=collective, steps=args.steps,
                batch_size=args.batch_size, lr=args.lr, seed=trial,
                alpha_ns=args.alpha_ns, beta=args.beta,
                synthetic=args.synthetic, norm=args.norm,
            )
            # The collective ends when its slowest rank ends.
            slowest = max(results, key=lambda r: r["breakdown"]["total_ns"])
            trials.append(slowest["breakdown"])

        def avg(field: str) -> float:
            return float(np.mean([b[field] for b in trials])) / args.steps / 1e6

        step = avg("total_ns")
        compute = avg("forward_ns") + avg("backward_ns") + avg("optimizer_ns")
        allreduce = avg("allreduce_ns")
        pack = avg("gather_ns") + avg("scatter_ns")
        transfer = avg("to_host_ns") + avg("to_device_ns")
        comm = avg("comm_ns")
        summary[collective] = {
            "step_ms": step, "compute_ms": compute, "allreduce_ms": allreduce,
            "pack_ms": pack, "transfer_ms": transfer, "comm_share": comm / step,
        }
        print(
            f"{collective:>11} {step:>8.1f}ms {compute:>8.1f}ms {allreduce:>9.1f}ms "
            f"{pack:>7.1f}ms {transfer:>8.1f}ms {comm / step:>10.1%}"
        )

    best = min(summary, key=lambda c: summary[c]["step_ms"])
    worst = max(summary, key=lambda c: summary[c]["step_ms"])
    print()
    print(
        f"fastest: {best} ({summary[best]['step_ms']:.1f}ms/step), "
        f"slowest: {worst} ({summary[worst]['step_ms']:.1f}ms/step) "
        f"-- {summary[worst]['step_ms'] / summary[best]['step_ms']:.2f}x"
    )
    speedup = (summary[worst]["step_ms"] - summary[best]["step_ms"]) / summary[worst]["step_ms"]
    print(f"choosing the right collective cuts step time by {speedup:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": vars(args) | {"out": str(args.out)},
                                    "summary": summary}, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
