"""Correctness gate for data-parallel training.

The claim is stronger than "all-reduce returns the right sum": p workers must
produce the *same optimisation trajectory* as one worker over the same global
batch. That catches scaling errors (summing instead of averaging), sharding
errors, and parameter-ordering errors in the flatten/unflatten path, none of
which a single all-reduce assertion detects.

Two things have to be right for the comparison to be meaningful, and both were
wrong in the first version of this file:

1. **Compare the global-batch loss, not rank 0's loss.** Each rank sees its own
   shard, so rank 0's loss cannot match a single worker's by construction. The
   mean across ranks is the global-batch loss.
2. **Use GroupNorm, not BatchNorm.** With BatchNorm the forward pass itself
   differs -- each rank normalises over its shard (8 samples at p=4, not 32) --
   so trajectories diverge before any gradient is exchanged, no matter how
   correct the collective is. That divergence is real data-parallel behaviour
   (it is why SyncBatchNorm exists) and is asserted separately below rather than
   papered over.

The learning rate is kept low deliberately: a diverging run amplifies
floating-point differences chaotically and would make the tolerance meaningless.
"""
from __future__ import annotations

import numpy as np
import pytest

from dpt.train.resnet_cifar import run

STEPS = 5
GLOBAL_BATCH = 32
LR = 0.01
TOL = 1e-3  # far above observed 5e-5 accumulation noise, far below any real bug


def _global_loss(results) -> np.ndarray:
    """Mean across equal-sized rank shards is the global-batch loss."""
    return np.mean([r["losses"] for r in results], axis=0)


def _run(workers: int, collective: str, norm: str = "gn"):
    return run(
        workers=workers, collective=collective, steps=STEPS,
        batch_size=GLOBAL_BATCH, lr=LR, seed=0, alpha_ns=0.0, beta=0.0,
        synthetic=True, norm=norm,
    )


@pytest.fixture(scope="module")
def baseline():
    return _global_loss(_run(1, "ring"))


@pytest.mark.parametrize(
    "workers,collective",
    [(2, "ring"), (4, "ring"), (2, "ps"), (4, "ps"), (2, "recursive"), (4, "recursive")],
)
def test_data_parallel_matches_single_process(baseline, workers, collective):
    losses = _global_loss(_run(workers, collective))
    assert len(losses) == len(baseline)
    assert np.max(np.abs(losses - baseline)) < TOL, (
        f"{collective} p={workers} diverges from single-process training: "
        f"{losses.tolist()} vs {baseline.tolist()}"
    )


def test_collectives_agree_with_each_other():
    """Different algorithms must drive bit-comparable trajectories."""
    ring = _global_loss(_run(4, "ring"))
    ps = _global_loss(_run(4, "ps"))
    recursive = _global_loss(_run(4, "recursive"))
    assert np.max(np.abs(ring - ps)) < TOL
    assert np.max(np.abs(ring - recursive)) < TOL


def test_batchnorm_breaks_equivalence_as_expected():
    """Documents a real property: BatchNorm makes p-worker != 1-worker.

    Asserted rather than described, so that if a future change (SyncBatchNorm,
    say) removes the discrepancy, this test fails and the docs get updated.
    """
    single = _global_loss(_run(1, "ring", norm="bn"))
    four = _global_loss(_run(4, "ring", norm="bn"))
    assert np.max(np.abs(four - single)) > TOL, (
        "BatchNorm trajectories matched unexpectedly -- gradient sync may now "
        "include batch statistics, and docs/findings-e2e.md needs revisiting"
    )
