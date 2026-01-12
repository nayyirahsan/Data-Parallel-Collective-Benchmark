"""Analytic alpha-beta cost models and crossover solving.

These are the predictions H1 tests. Two corrections to the textbook forms are
required for the models to describe *this* testbed, and both are consequences of
the send-side charging decision (see docs/decisions.md #4):

1. The parameter server's broadcast is serial, not parallel. Textbook PS cost is
   quoted as ``2*alpha`` on the assumption that the server fans out
   concurrently. Under a shared uplink -- which is the whole reason the PS
   bandwidth bottleneck exists -- the server pays alpha per outbound copy, so
   the critical path is ``alpha + (p-1)*alpha = p*alpha``. Using the textbook
   form here would have manufactured a spurious 2x model error for PS that had
   nothing to do with H1.

2. Cost is measured along the *critical path*, not summed over all ranks. The
   collective ends when its slowest participant ends.

``alpha_sw`` is the per-step software overhead term H1 claims the standard model
omits. It defaults to zero, which recovers the textbook prediction; fitting it
from the alpha=beta=0 floor run gives the corrected model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Predicted critical-path time in nanoseconds for an all-reduce of N bytes."""

    alpha_ns: float
    beta_ns_per_byte: float
    alpha_sw_ns: float = 0.0

    def _step(self, nbytes: float) -> float:
        return self.alpha_ns + self.alpha_sw_ns + nbytes * self.beta_ns_per_byte

    def ring(self, nbytes: float, p: int) -> float:
        """2(p-1) steps, each carrying N/p bytes. Bandwidth-optimal."""
        if p < 2:
            return 0.0
        return 2 * (p - 1) * self._step(nbytes / p)

    def recursive(self, nbytes: float, p: int) -> float:
        """2*log2(p) steps; step k of reduce-scatter carries N/2^(k+1) bytes.

        Total bytes moved equals ring's 2*((p-1)/p)*N, but spread over
        logarithmically many steps -- so it pays the per-step cost far less often.
        """
        if p < 2:
            return 0.0
        rounds = int(math.log2(p))
        total = 0.0
        for k in range(rounds):
            total += 2 * self._step(nbytes / (2 ** (k + 1)))
        return total

    def ps(self, nbytes: float, p: int) -> float:
        """One worker send, then a serial p-1 broadcast from the server."""
        if p < 2:
            return 0.0
        return self._step(nbytes) + (p - 1) * self._step(nbytes)

    def predict(self, algo: str, nbytes: float, p: int) -> float:
        return {"ring": self.ring, "recursive": self.recursive, "ps": self.ps}[algo](
            nbytes, p
        )


def crossover_bytes(
    model: CostModel, algo_a: str, algo_b: str, p: int,
    lo: float = 1.0, hi: float = 1 << 32, tol: float = 1.0,
) -> float | None:
    """Smallest N where ``algo_a`` becomes at least as cheap as ``algo_b``.

    Bisection rather than solving in closed form: the recursive model's summed
    geometric terms make the closed form awkward, and bisection extends
    unchanged to the empirically fitted models used in the analysis, so
    prediction and measurement are solved by identical machinery.
    """
    def diff(n: float) -> float:
        return model.predict(algo_a, n, p) - model.predict(algo_b, n, p)

    d_lo, d_hi = diff(lo), diff(hi)
    scale = max(abs(model.predict(algo_a, hi, p)), 1.0)
    if abs(d_lo) < 1e-9 * scale and abs(d_hi) < 1e-9 * scale:
        # The two models are identical across the range -- e.g. ring and PS at
        # p=2 with beta=0, where both reduce to 2*alpha. Bisecting this returns
        # a meaningless ~0 rather than "no crossover", so it is caught here.
        return None
    if d_lo * d_hi > 0:
        return None  # no sign change: one arm dominates across the range
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if diff(lo) * diff(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
