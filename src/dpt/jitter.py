"""Deterministic, replayable jitter for paired algorithm comparison.

Design note (defensible decision #3): jitter attaches to (worker, step), not to
message index
--------------------------------------------------------------------------
H2 compares algorithms under identical tail-latency conditions, which requires
replaying the *same* jitter realization against each. Keying jitter on message
index cannot do that: ring all-reduce sends 2(p-1) messages per collective and
a parameter server sends 2, so message k in one is not message k in the other.

Instead jitter is a property of a *worker during a step* -- "rank r is
transiently slow at step s" -- which is both physically meaningful (OS noise,
thermal throttling, a noisy neighbour) and algorithm-independent. Every message
rank r sends during step s is scaled by the same draw. Both algorithms then see
an identical slowness pattern across workers and time, making the comparison
properly paired rather than two independent samples.

The draw is a pure function of (seed, rank, step), so any run is reproducible
from its config and no jitter trace needs to be stored.
"""
from __future__ import annotations

from dataclasses import dataclass

_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """Deterministic, well-distributed 64-bit mix. Avoids Python's salted hash()."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _uniform(seed: int, rank: int, step: int) -> float:
    """Reproducible uniform in [0, 1) from the (seed, rank, step) key."""
    key = _splitmix64(seed * 0x1000193 ^ _splitmix64(rank * 0x01000193 ^ step))
    return (key >> 11) / float(1 << 53)


@dataclass(frozen=True)
class JitterModel:
    """Two-point heavy-tail model: with probability ``q``, delay scales by ``k``.

    A mixture rather than a Gaussian because the phenomenon H2 is about is tail
    latency, not variance. Straggler behaviour in real clusters is rare-and-
    large, and a symmetric distribution would average out across ring steps
    instead of compounding -- which would test the wrong thing.

    ``q=0`` disables jitter, making the clean benchmark a special case of the
    same code path rather than a separate one.
    """

    probability: float = 0.0
    multiplier: float = 1.0
    seed: int = 0

    def scale(self, rank: int, step: int) -> float:
        """Delay multiplier for messages sent by ``rank`` during ``step``."""
        if self.probability <= 0.0:
            return 1.0
        if _uniform(self.seed, rank, step) < self.probability:
            return self.multiplier
        return 1.0

    @property
    def enabled(self) -> bool:
        return self.probability > 0.0

    def describe(self) -> str:
        if not self.enabled:
            return "none"
        return f"q={self.probability:g},k={self.multiplier:g}"
