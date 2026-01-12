"""ResNet-18 / CIFAR-10 data-parallel training.

The correctness claim this supports: a from-scratch collective must produce the
*same training run* as a single process, not merely the same tensor. Comparing
loss curves across a real optimisation catches ordering and scaling bugs that a
single all-reduce assertion does not.

Determinism matters more than throughput here. Each rank gets a disjoint shard
of a fixed batch sequence, so p ranks at batch B/p see exactly the data one rank
at batch B sees, and the loss curves are directly comparable.

    python -m dpt.train.resnet_cifar --workers 4 --collective ring --steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..bench.launch import run_ranks
from ..transport.base import Transport
from ..transport.shaped import ShapedTransport
from .ddp import GradientSynchroniser, StepBreakdown, train_steps

DATA_ROOT = Path("data")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(seed: int = 0, norm: str = "bn") -> nn.Module:
    """ResNet-18 adapted for 32x32 input.

    The stock stem downsamples 4x before layer1, which throws away most of a
    CIFAR image. Replacing it with a 3x3 stride-1 conv and dropping the maxpool
    is the standard CIFAR adaptation.

    ``norm`` selects the normalisation layer, and the choice matters for
    correctness testing. With BatchNorm, p workers do *not* reproduce a single
    worker's trajectory even with perfect gradient synchronisation: each rank
    computes batch statistics over its own shard (8 samples at p=4, not 32), so
    the forward pass differs before any gradient is exchanged. This is why
    SyncBatchNorm exists, and it is a property of data-parallel training rather
    than a bug.

    GroupNorm is batch-independent, so under it the p-worker and single-worker
    trajectories must agree to floating-point accumulation order -- which makes
    it the right choice for an exact correctness gate.
    """
    from torchvision.models import resnet18

    torch.manual_seed(seed)
    if norm == "gn":
        model = resnet18(
            num_classes=10,
            norm_layer=lambda c: nn.GroupNorm(min(32, c), c),
        )
    elif norm == "bn":
        model = resnet18(num_classes=10)
    else:
        raise ValueError(f"unknown norm {norm!r}, expected 'bn' or 'gn'")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class SyntheticCIFAR(torch.utils.data.Dataset):
    """Deterministic CIFAR-shaped data, for validating the training path offline.

    Labels are a fixed function of the input seed, so the task is learnable and
    the loss curve is meaningful as a *comparison* between 1 and p workers --
    which is what the correctness gate needs. It is not a substitute for real
    CIFAR-10 when reporting accuracy, and runs using it are labelled synthetic.
    """

    def __init__(self, n: int = 4096, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (n,), generator=g)
        # Class-conditional means make the task separable rather than noise.
        centres = torch.randn(10, 3, 32, 32, generator=g)
        self.data = centres[self.labels] + 0.5 * torch.randn(
            n, 3, 32, 32, generator=g
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])


def load_dataset(train: bool = True, synthetic: bool = False):
    if synthetic:
        return SyntheticCIFAR(seed=0 if train else 1)
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    # torchvision's download needs a usable CA bundle; the system Python here
    # has none, so certifi's is wired in explicitly rather than disabling
    # verification.
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return CIFAR10(root=str(DATA_ROOT), train=train, download=True, transform=tf)


def make_batches(dataset, steps: int, batch_size: int, rank: int, world_size: int, seed: int):
    """Deterministic disjoint shards of a fixed global batch sequence."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator)
    per_rank = batch_size // world_size

    batches = []
    for step in range(steps):
        start = step * batch_size
        if start + batch_size > len(order):
            break
        global_idx = order[start : start + batch_size]
        shard = global_idx[rank * per_rank : (rank + 1) * per_rank]
        xs = torch.stack([dataset[int(i)][0] for i in shard])
        ys = torch.tensor([dataset[int(i)][1] for i in shard])
        batches.append((xs, ys))
    return batches


def _worker(
    transport: Transport, collective: str, steps: int, batch_size: int,
    lr: float, seed: int, alpha_ns: float, beta: float, synthetic: bool,
    norm: str,
):
    device = pick_device()
    dataset = load_dataset(train=True, synthetic=synthetic)
    batches = make_batches(
        dataset, steps, batch_size, transport.rank, transport.world_size, seed
    )

    model = build_model(seed, norm).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    sync = None
    if transport.world_size > 1:
        shaped = ShapedTransport(transport, alpha_ns=alpha_ns, beta_ns_per_byte=beta)
        sync = GradientSynchroniser(model, shaped, collective)

    start = time.perf_counter_ns()
    losses, breakdown = train_steps(model, optimizer, batches, sync, device)
    wall_ns = time.perf_counter_ns() - start

    return {
        "rank": transport.rank,
        "losses": losses,
        "breakdown": breakdown.as_dict(),
        "wall_ns": wall_ns,
        "grad_bytes": sync.nbytes if sync else 0,
        "device": str(device),
    }


def run(
    workers: int, collective: str, steps: int, batch_size: int, lr: float,
    seed: int, alpha_ns: float, beta: float, synthetic: bool = False,
    norm: str = "bn",
):
    if workers == 1:
        device = pick_device()
        dataset = load_dataset(train=True, synthetic=synthetic)
        batches = make_batches(dataset, steps, batch_size, 0, 1, seed)
        model = build_model(seed, norm).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        t0 = time.perf_counter_ns()
        losses, breakdown = train_steps(model, optimizer, batches, None, device)
        return [
            {
                "rank": 0, "losses": losses, "breakdown": breakdown.as_dict(),
                "wall_ns": time.perf_counter_ns() - t0, "grad_bytes": 0,
                "device": str(device),
            }
        ]
    return run_ranks(
        workers, _worker, collective, steps, batch_size, lr, seed, alpha_ns, beta,
        synthetic, norm, timeout=3600,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--collective", default="ring", choices=["ring", "ps", "recursive"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128, help="global batch size")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-ns", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--norm", default="bn", choices=["bn", "gn"],
                    help="gn (GroupNorm) makes p-worker training exactly "
                         "reproduce single-worker training")
    ap.add_argument("--synthetic", action="store_true",
                    help="use offline synthetic data instead of CIFAR-10")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = run(
        args.workers, args.collective, args.steps, args.batch_size, args.lr,
        args.seed, args.alpha_ns, args.beta, args.synthetic, args.norm,
    )

    losses = results[0]["losses"]
    breakdown = results[0]["breakdown"]
    wall = max(r["wall_ns"] for r in results)

    print(f"workers={args.workers} collective={args.collective} device={results[0]['device']}")
    if results[0]["grad_bytes"]:
        print(f"gradient buffer: {results[0]['grad_bytes'] / 1024**2:.1f} MiB per all-reduce")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")
    print(f"wall: {wall / 1e6:.0f}ms  ({wall / 1e6 / max(1, len(losses)):.1f}ms/step)")
    print("\nstep breakdown (rank 0, totals):")
    total = breakdown["total_ns"]
    for field in ("forward_ns", "backward_ns", "gather_ns", "allreduce_ns",
                  "scatter_ns", "optimizer_ns"):
        v = breakdown[field]
        print(f"  {field[:-3]:>10}: {v / 1e6:>8.1f}ms  {v / total * 100:>5.1f}%")
    print(f"  {'comm total':>10}: {breakdown['comm_ns'] / 1e6:>8.1f}ms  "
          f"{breakdown['comm_ns'] / total * 100:>5.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"args": vars(args) | {"out": str(args.out)}, "results": results}, indent=2
        ))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
