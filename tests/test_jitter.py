"""Tests for the jitter model.

The paired-comparison design in H2 depends on jitter being a pure function of
(seed, rank, step): the same realisation has to replay across algorithms that
send different numbers of messages.
"""
from __future__ import annotations

import pytest

from dpt.jitter import JitterModel


def test_disabled_by_default():
    model = JitterModel()
    assert not model.enabled
    assert all(model.scale(r, s) == 1.0 for r in range(8) for s in range(50))


def test_deterministic_across_instances():
    """Two separately constructed models must agree -- this is what lets ring
    and PS runs be compared as paired samples."""
    a = JitterModel(probability=0.1, multiplier=10.0, seed=42)
    b = JitterModel(probability=0.1, multiplier=10.0, seed=42)
    assert [a.scale(r, s) for r in range(8) for s in range(100)] == \
           [b.scale(r, s) for r in range(8) for s in range(100)]


def test_seed_changes_realisation():
    a = JitterModel(probability=0.1, multiplier=10.0, seed=1)
    b = JitterModel(probability=0.1, multiplier=10.0, seed=2)
    assert [a.scale(r, s) for r in range(8) for s in range(200)] != \
           [b.scale(r, s) for r in range(8) for s in range(200)]


@pytest.mark.parametrize("q", [0.01, 0.05, 0.25])
def test_hit_rate_matches_probability(q):
    model = JitterModel(probability=q, multiplier=10.0, seed=7)
    draws = [model.scale(r, s) for r in range(16) for s in range(2000)]
    observed = sum(d > 1.0 for d in draws) / len(draws)
    assert observed == pytest.approx(q, abs=0.015)


def test_ranks_are_independent():
    """A step must not jitter every rank at once -- that would make stragglers
    a global slowdown rather than a per-worker one."""
    model = JitterModel(probability=0.5, multiplier=10.0, seed=3)
    rows = [[model.scale(r, s) for r in range(8)] for s in range(200)]
    assert any(len(set(row)) > 1 for row in rows)
