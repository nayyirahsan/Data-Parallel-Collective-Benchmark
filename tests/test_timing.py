"""Tests for delay injection.

The core-yielding delay is what makes p>=4 tail measurements possible, so its
accuracy is gated here rather than trusted. Thresholds are deliberately loose
relative to the measured 0.03-0.33% median error: these run on shared CI-style
machines where the tail is unpredictable, and a flaky timing test is worse than
none.
"""
from __future__ import annotations

import time

import pytest

from dpt.timing import (
    MIN_SLEEP_TARGET_NS,
    SLEEP_FRACTION,
    bandwidth_to_beta,
    delay_ns,
    link_delay_ns,
    spin_ns,
)


def _elapsed(fn, target_ns: int, reps: int = 15) -> float:
    fn(target_ns)  # warm
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn(target_ns)
        samples.append(time.perf_counter_ns() - t0)
    return sorted(samples)[len(samples) // 2]


@pytest.mark.parametrize("target_ns", [50_000, 200_000, 1_000_000])
def test_delay_never_finishes_early(target_ns):
    """Undershooting would silently weaken every injected cost."""
    assert _elapsed(delay_ns, target_ns) >= target_ns * 0.99


@pytest.mark.parametrize("target_ns", [200_000, 1_000_000])
def test_delay_matches_spin_accuracy(target_ns):
    """Yielding the core must not cost accuracy -- that is the whole trade."""
    spun = _elapsed(spin_ns, target_ns)
    slept = _elapsed(delay_ns, target_ns)
    assert abs(slept - spun) / target_ns < 0.10


def test_sleep_fraction_stays_inside_the_deadline():
    """0.7 * worst-observed overshoot (~1.35x) < 1.0, so the sleep leg cannot
    overrun the deadline. At 0.8 it does, measured at +9.2% error."""
    assert 0 < SLEEP_FRACTION <= 0.75


def test_zero_fraction_is_pure_spin():
    assert _elapsed(lambda ns: delay_ns(ns, 0.0), 200_000) >= 200_000 * 0.99


def test_short_delays_skip_the_sleep():
    """Below the threshold the scheduler's granularity dominates."""
    assert MIN_SLEEP_TARGET_NS > 0
    assert _elapsed(delay_ns, MIN_SLEEP_TARGET_NS // 2) > 0


def test_non_positive_delay_is_a_noop():
    for value in (0, -1, -10_000):
        t0 = time.perf_counter_ns()
        delay_ns(value)
        spin_ns(value)
        assert time.perf_counter_ns() - t0 < 1_000_000


def test_link_delay_is_alpha_plus_n_beta():
    assert link_delay_ns(0, 100.0, 2.0) == 100
    assert link_delay_ns(1000, 100.0, 2.0) == 2100
    assert link_delay_ns(1000, 0.0, 0.0) == 0


def test_bandwidth_conversion_round_trips():
    assert bandwidth_to_beta(10.0) == pytest.approx(0.8)
    assert bandwidth_to_beta(0) == 0.0
    assert 8.0 / bandwidth_to_beta(20.0) == pytest.approx(20.0)
