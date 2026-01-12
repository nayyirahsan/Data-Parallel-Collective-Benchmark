"""High-precision delay injection.

Design note (defensible decision #1)
------------------------------------
We spin-wait rather than sleep. Measured on the target machine (Apple M4 Pro,
12 cores), ``time.sleep`` carries a consistent +20-30% overshoot at *every*
scale from 1us to 10ms -- this is a proportional bias, not a granularity floor,
so it does not wash out at larger delays and cannot be corrected by a fixed
offset. A hybrid (sleep-most, spin-remainder) strategy inherits that bias
whenever the sleep leg is long enough to matter: measured 17.7% error at 5ms.

Pure spin-wait holds median error <=0.4% across 1us-10ms and across 1-8
concurrent processes. See ``docs/calibration.md`` for the full measurement.

The cost is a fully consumed core for the duration of the delay, which bounds
the usable worker count. Empirically p<=8 on this 12-core machine: at p=12 the
spinners starve the scheduler and p99 error blows up to 58-112%.
"""
from __future__ import annotations

import time

NS_PER_S = 1_000_000_000


# Fraction of a delay spent asleep before spinning to the deadline. The
# original design pure-spun, which is accurate but consumes a core for the whole
# delay -- and that is what capped the experiment at p<=8 on a 12-core machine.
#
# Sleeping for a *fixed* slack fails, because the overshoot is proportional
# (~1.25-1.35x) rather than absolute: that is why the fixed-300us hybrid
# measured +17.7% error at 5ms. Sleeping for a *fraction* of the delay keeps the
# overshoot inside the deadline at every scale, because 0.7 * 1.35 < 1.
#
# Measured on the M4 Pro: at 0.7 the median error stays <=0.33% across
# 50us-5ms while CPU occupancy falls from 99% to 11-15%. At 0.8 the sleep starts
# overrunning the deadline (+9.2% at 50us), so 0.7 is the usable maximum.
SLEEP_FRACTION = 0.7

# Below this, a sleep is not worth attempting: the scheduler's own granularity
# dominates and the spin tail would have to absorb most of the delay anyway.
MIN_SLEEP_TARGET_NS = 20_000

# A sleep leg is only safe if the platform can actually deliver a sleep that
# short, and that cannot be assumed. Measured sleep granularity is ~4 us on
# macOS but ~1 ms inside Docker's linuxkit VM, where a 140 us sleep leg (0.7 of
# a 200 us delay) would overrun the deadline five times over and silently
# inflate every injected cost. The same ~1 ms quantum makes tc netem unusable
# below 1 ms there (docs/netem-validation.md), so this is a property of
# virtualised timers rather than of one container -- and any cloud VM, including
# GitHub Codespaces, may have it.
#
# Granularity is therefore measured once, at first use, and the sleep leg is
# skipped whenever the requested sleep is not comfortably above it. This makes
# the shim degrade to pure spinning on coarse-timer platforms rather than
# quietly producing wrong delays.
SLEEP_SAFETY_MARGIN = 2.0
_SLEEP_GRANULARITY_NS: int | None = None


def measure_sleep_granularity_ns(reps: int = 12) -> int:
    """Shortest sleep this platform can actually deliver, in nanoseconds."""
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        time.sleep(1e-6)  # ask for 1us; get whatever the platform floors at
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    return samples[len(samples) // 2]


def sleep_granularity_ns() -> int:
    """Cached sleep granularity for this process."""
    global _SLEEP_GRANULARITY_NS
    if _SLEEP_GRANULARITY_NS is None:
        _SLEEP_GRANULARITY_NS = measure_sleep_granularity_ns()
    return _SLEEP_GRANULARITY_NS


def spin_ns(duration_ns: int) -> None:
    """Busy-wait for ``duration_ns`` nanoseconds. Accurate but burns a core."""
    if duration_ns <= 0:
        return
    deadline = time.perf_counter_ns() + duration_ns
    while time.perf_counter_ns() < deadline:
        pass


def delay_ns(duration_ns: int, sleep_fraction: float = SLEEP_FRACTION) -> None:
    """Wait ``duration_ns``, yielding the core for most of it.

    Sleeps for ``sleep_fraction`` of the target, then spins to the deadline.
    Falls back to pure spinning for very short delays, when ``sleep_fraction``
    is zero, and -- critically -- when the platform's measured sleep granularity
    is too coarse to deliver the sleep leg without overrunning the deadline.
    """
    if duration_ns <= 0:
        return
    deadline = time.perf_counter_ns() + duration_ns
    if sleep_fraction > 0 and duration_ns >= MIN_SLEEP_TARGET_NS:
        sleep_target = duration_ns * sleep_fraction
        if sleep_target >= sleep_granularity_ns() * SLEEP_SAFETY_MARGIN:
            time.sleep(sleep_target / NS_PER_S)
    while time.perf_counter_ns() < deadline:
        pass


def link_delay_ns(nbytes: int, alpha_ns: float, beta_ns_per_byte: float) -> int:
    """alpha-beta link cost for a message of ``nbytes``.

    ``alpha_ns`` is per-message latency; ``beta_ns_per_byte`` is inverse
    bandwidth. Returns total nanoseconds to charge for the transfer.
    """
    return int(alpha_ns + nbytes * beta_ns_per_byte)


def bandwidth_to_beta(gbps: float) -> float:
    """Convert a link bandwidth in Gbit/s to beta in ns/byte."""
    if gbps <= 0:
        return 0.0
    return 8.0 / gbps  # bits/byte / (bits/ns)
