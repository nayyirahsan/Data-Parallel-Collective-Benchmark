"""Network shaping: injects a controlled alpha-beta link cost.

Design note (defensible decision #4): the delay is charged at send-initiate
--------------------------------------------------------------------------
We model a store-and-forward link: the sender pays serialisation plus
propagation (``alpha + N*beta``) before the bytes are handed to the substrate.
The alternative -- delaying the receiver's completion -- models a cut-through
link instead, and would let a sender race ahead issuing messages it has not yet
paid for, decoupling the injected cost from the algorithm's dependency
structure. Since H2 is precisely a claim about dependency structure (ring's
2(p-1) serial steps), the delay has to sit on the critical path where the
algorithm actually serialises. Charging at send-initiate does that.

The consequence to state plainly: this emulates a link whose cost is borne by
the sender, so a broadcast to p receivers costs p times, as it would on a real
shared uplink. That is the behaviour the parameter-server bottleneck depends
on, and it is validated against `tc netem` in the Colab notebook.
"""
from __future__ import annotations

from typing import Any

import torch

from ..jitter import JitterModel
from ..timing import SLEEP_FRACTION, delay_ns, link_delay_ns
from .base import Transport


class ShapedTransport(Transport):
    """Wraps a transport, charging a synthetic link cost on every send.

    ``sleep_fraction`` controls how much of each injected delay is spent asleep
    rather than spinning (see ``dpt.timing``). It defaults to the calibrated
    value; pass 0 to force pure spinning, which is more accurate in the tail at
    very small delays but consumes a core per worker.

    ``alpha_ns`` is per-message latency, ``beta_ns_per_byte`` inverse bandwidth.
    Both default to zero, which makes the unshaped case -- used to measure the
    pure software overhead floor that H1 turns on -- the same code path with the
    same overheads, not a separate one.
    """

    def __init__(
        self,
        inner: Transport,
        alpha_ns: float = 0.0,
        beta_ns_per_byte: float = 0.0,
        jitter: JitterModel | None = None,
        sleep_fraction: float = SLEEP_FRACTION,
    ):
        self._inner = inner
        self.sleep_fraction = sleep_fraction
        self.alpha_ns = alpha_ns
        self.beta_ns_per_byte = beta_ns_per_byte
        self.jitter = jitter or JitterModel()
        self._step = 0
        self.injected_ns = 0
        self.messages_sent = 0

    # -- step bookkeeping -------------------------------------------------
    def set_step(self, step: int) -> None:
        """Advance the logical step used to key jitter draws."""
        self._step = step

    def reset_counters(self) -> None:
        self.injected_ns = 0
        self.messages_sent = 0

    # -- Transport --------------------------------------------------------
    @property
    def rank(self) -> int:
        return self._inner.rank

    @property
    def world_size(self) -> int:
        return self._inner.world_size

    def _charge(self, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        delay = link_delay_ns(nbytes, self.alpha_ns, self.beta_ns_per_byte)
        if self.jitter.enabled:
            delay = int(delay * self.jitter.scale(self.rank, self._step))
        self.injected_ns += delay
        self.messages_sent += 1
        delay_ns(delay, self.sleep_fraction)

    def isend(self, tensor: torch.Tensor, dst: int) -> Any:
        self._charge(tensor)
        return self._inner.isend(tensor, dst)

    def irecv(self, tensor: torch.Tensor, src: int) -> Any:
        return self._inner.irecv(tensor, src)

    def barrier(self) -> None:
        self._inner.barrier()
