"""Tests for the analytic cost model and crossover solver."""
from __future__ import annotations

import math

import pytest

from dpt.model import CostModel, crossover_bytes

ALPHA = 200_000.0
BETA = 0.4
N = 1 << 20


def test_ring_is_bandwidth_optimal_relative_to_ps():
    """Ring moves 2*((p-1)/p)*N bytes; PS's server carries p*N."""
    latency_free = CostModel(alpha_ns=0.0, beta_ns_per_byte=BETA)
    for p in (4, 8, 16):
        assert latency_free.ring(N, p) < latency_free.ps(N, p)


def test_recursive_has_fewest_steps():
    """With no bandwidth term, cost is purely step count."""
    bandwidth_free = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=0.0)
    for p in (8, 16, 32):
        steps = {
            "ring": bandwidth_free.ring(N, p) / ALPHA,
            "ps": bandwidth_free.ps(N, p) / ALPHA,
            "recursive": bandwidth_free.recursive(N, p) / ALPHA,
        }
        assert steps["recursive"] == pytest.approx(2 * math.log2(p))
        assert steps["ring"] == pytest.approx(2 * (p - 1))
        assert steps["ps"] == pytest.approx(p)


def test_ps_broadcast_is_serial_not_parallel():
    """The textbook 2*alpha assumes parallel fan-out; a shared uplink gives p*alpha.

    Getting this wrong manufactures a spurious 2x model error for PS that has
    nothing to do with H1, so it is pinned down here.
    """
    model = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=0.0)
    for p in (2, 4, 8):
        assert model.ps(N, p) == pytest.approx(p * ALPHA)


def test_alpha_sw_shifts_crossover_upward():
    """Adding software overhead should only ever push the crossover later."""
    textbook = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=BETA)
    corrected = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=BETA, alpha_sw_ns=250_000)
    for p in (4, 8):
        a = crossover_bytes(textbook, "ring", "ps", p)
        b = crossover_bytes(corrected, "ring", "ps", p)
        assert a is not None and b is not None
        assert b > a


def test_no_crossover_when_one_arm_always_wins():
    """With beta=0 ring pays more steps for no bandwidth benefit at p>2."""
    model = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=0.0)
    assert crossover_bytes(model, "ring", "ps", 8) is None


def test_degenerate_identical_models_report_no_crossover():
    """At p=2 with beta=0 ring and PS both reduce to exactly 2*alpha.

    Bisecting that returns a meaningless ~0 rather than 'no crossover', which
    previously produced ratios like 281063x in the analysis output.
    """
    model = CostModel(alpha_ns=ALPHA, beta_ns_per_byte=0.0)
    assert model.ring(N, 2) == pytest.approx(model.ps(N, 2))
    assert crossover_bytes(model, "ring", "ps", 2) is None
