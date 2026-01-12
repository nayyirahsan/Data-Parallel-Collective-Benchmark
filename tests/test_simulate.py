"""Tests for the discrete-event model.

Two findings rest on this model, so it is checked against the independent
analytic cost model in ``dpt.model``. The two were written from the same step
accounting but by different means -- one closed-form, one event-driven -- so
agreement between them is a real consistency check rather than a tautology.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dpt.model import CostModel
from dpt.simulate import SimConfig, degradation, simulate

ALPHA = 200_000.0
BETA = 0.4
NBYTES = 1 << 20


@pytest.mark.parametrize("algo", ["ring", "ps", "recursive"])
@pytest.mark.parametrize("p", [2, 4, 8])
def test_clean_simulation_matches_analytic_model(algo, p):
    """With no jitter every message costs the same, so the event model must
    reproduce the closed-form critical path exactly."""
    cfg = SimConfig(algo, p, NBYTES, ALPHA, BETA)
    simulated = simulate(cfg, iters=64)
    predicted = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=BETA).predict(algo, NBYTES, p)

    assert np.allclose(simulated, simulated[0]), "clean run should be deterministic"
    assert simulated[0] == pytest.approx(predicted, rel=1e-9)


@pytest.mark.parametrize("p", [2, 4, 8, 16])
def test_step_counts(p):
    """Completion time divided by per-message cost recovers the step count."""
    for algo, expected in (
        ("ring", 2 * (p - 1)),
        ("ps", p),
        ("recursive", 2 * int(math.log2(p))),
    ):
        cfg = SimConfig(algo, p, NBYTES, ALPHA, 0.0)
        assert simulate(cfg, iters=4)[0] / ALPHA == pytest.approx(expected, rel=1e-9)


def test_recursive_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        simulate(SimConfig("recursive", 6, NBYTES, ALPHA, BETA), iters=4)


@pytest.mark.parametrize("algo", ["ring", "ps", "recursive"])
def test_jitter_never_speeds_things_up(algo):
    """Injected delay can only add time. A ratio below 1 means a bug."""
    cfg = SimConfig(algo, 8, NBYTES, ALPHA, BETA, jitter_q=0.05, jitter_k=10.0)
    result = degradation(cfg, iters=20_000)
    assert result["p99_ratio"] >= 1.0
    assert result["median_ratio"] >= 1.0


def test_simulation_is_reproducible():
    cfg = SimConfig("ring", 8, NBYTES, ALPHA, BETA, jitter_q=0.01, jitter_k=10.0)
    assert np.array_equal(simulate(cfg, 5_000, seed=3), simulate(cfg, 5_000, seed=3))
    assert not np.array_equal(simulate(cfg, 5_000, seed=3), simulate(cfg, 5_000, seed=4))


def test_more_jitter_degrades_more():
    light = degradation(SimConfig("ring", 8, NBYTES, ALPHA, BETA, 0.01, 10.0), 50_000)
    heavy = degradation(SimConfig("ring", 8, NBYTES, ALPHA, BETA, 0.10, 10.0), 50_000)
    assert heavy["median_ratio"] >= light["median_ratio"]
