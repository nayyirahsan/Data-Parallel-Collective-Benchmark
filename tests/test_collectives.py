"""Correctness gate: every from-scratch collective must match a known sum.

The credibility of every timing result in this project rests on the algorithms
being correct, so these run first and cheaply.
"""
from __future__ import annotations

import pytest
import torch

from dpt.bench.launch import run_ranks
from dpt.collectives import ALGORITHMS, recursive_supported
from dpt.transport.shaped import ShapedTransport

SIZES = [1, 7, 1024, 10_007]  # includes sizes indivisible by every world size


def _allreduce_job(transport, algo: str, numel: int, seed: int):
    """Each rank contributes a distinct deterministic vector; sum is checkable."""
    torch.manual_seed(seed)
    contributions = [
        torch.arange(numel, dtype=torch.float64) * (r + 1) for r in range(transport.world_size)
    ]
    expected = torch.stack(contributions).sum(dim=0)

    tensor = contributions[transport.rank].clone()
    shaped = ShapedTransport(transport)  # zero-cost shaping: same code path
    ALGORITHMS[algo](tensor, shaped)
    return torch.max(torch.abs(tensor - expected)).item()


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("algo", ["ring", "ps", "recursive"])
@pytest.mark.parametrize("numel", SIZES)
def test_allreduce_matches_expected_sum(world_size, algo, numel):
    if algo == "recursive" and not recursive_supported(world_size):
        pytest.skip("recursive halving/doubling requires power-of-two world size")
    errors = run_ranks(world_size, _allreduce_job, algo, numel, 0)
    for rank, err in enumerate(errors):
        assert err < 1e-6, f"rank {rank} max abs error {err}"


def _against_reference_job(transport, algo: str, numel: int):
    torch.manual_seed(1234 + transport.rank)
    tensor = torch.randn(numel, dtype=torch.float64)
    baseline = tensor.clone()

    ALGORITHMS[algo](tensor, ShapedTransport(transport))
    ALGORITHMS["reference"](baseline, transport)
    return torch.max(torch.abs(tensor - baseline)).item()


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("algo", ["ring", "ps", "recursive"])
def test_allreduce_matches_gloo_reference(world_size, algo):
    if algo == "recursive" and not recursive_supported(world_size):
        pytest.skip("recursive halving/doubling requires power-of-two world size")
    errors = run_ranks(world_size, _against_reference_job, algo, 4099)
    for rank, err in enumerate(errors):
        assert err < 1e-9, f"rank {rank} deviates from gloo all_reduce by {err}"


def test_recursive_rejects_non_power_of_two():
    from dpt.collectives.recursive import recursive_allreduce

    class _Fake:
        rank, world_size = 0, 6

    with pytest.raises(ValueError, match="power-of-two"):
        recursive_allreduce(torch.zeros(6), _Fake())
