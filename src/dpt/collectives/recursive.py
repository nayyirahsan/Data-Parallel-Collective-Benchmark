"""Recursive halving reduce-scatter + recursive doubling all-gather.

The theoretically strongest comparator: bandwidth-optimal like ring (each rank
moves 2*((p-1)/p)*N bytes) but with only 2*log2(p) steps instead of 2(p-1).
H1 predicts this should win the small-message corner of the surface, where the
per-step cost -- network alpha plus software overhead -- dominates.

Requires p to be a power of two. Non-power-of-two support needs a separate
pre/post reduction phase that would change the step count and muddy the very
comparison being made, so it is refused rather than approximated.
"""
from __future__ import annotations

import torch

from ..transport.base import Transport
from .util import PaddedFlat


def is_supported(world_size: int) -> bool:
    return world_size > 0 and (world_size & (world_size - 1)) == 0


def recursive_allreduce(
    tensor: torch.Tensor, transport: Transport, reuse_scratch: bool = True
) -> torch.Tensor:
    """In-place sum-all-reduce by recursive halving/doubling.

    ``reuse_scratch`` selects between one preallocated receive buffer and a
    fresh allocation per step. It exists so the two can be A/B tested inside
    a single interleaved sweep rather than across separate runs, where
    machine drift is larger than the effect being measured.
    """
    p = transport.world_size
    if p == 1:
        return tensor
    if not is_supported(p):
        raise ValueError(
            f"recursive halving/doubling requires a power-of-two world size, got {p}"
        )

    rank = transport.rank

    with PaddedFlat(tensor, p) as pf:
        buf = pf.buffer
        lo, hi = 0, pf.padded_size

        # One scratch buffer, reused by slicing, rather than a fresh allocation
        # per step. The first halving step needs N/2 elements and every later
        # step needs less, so this is the high-water mark. Allocating per step
        # is not merely wasteful: at ResNet-18 scale it means a 22 MB
        # allocation inside the timed region, which made recursive look slower
        # than ring for reasons that had nothing to do with the algorithm.
        scratch = (
            torch.empty(pf.padded_size // 2, dtype=buf.dtype)
            if reuse_scratch
            else None
        )

        # Reduce-scatter by recursive halving. Each round exchanges the half of
        # the active range the partner will own and accumulates the half we keep.
        d = p // 2
        while d >= 1:
            partner = rank ^ d
            mid = (lo + hi) // 2
            if rank & d == 0:
                keep_lo, keep_hi, send_lo, send_hi = lo, mid, mid, hi
            else:
                keep_lo, keep_hi, send_lo, send_hi = mid, hi, lo, mid
            size = keep_hi - keep_lo
            recv = scratch[:size] if scratch is not None else torch.empty(
                size, dtype=buf.dtype
            )
            transport.sendrecv(buf[send_lo:send_hi], partner, recv, partner)
            buf[keep_lo:keep_hi] += recv
            lo, hi = keep_lo, keep_hi
            d //= 2

        # All-gather by recursive doubling, unwinding the same partner sequence.
        d = 1
        while d < p:
            partner = rank ^ d
            span = hi - lo
            if rank & d == 0:
                recv_lo, recv_hi = hi, hi + span
            else:
                recv_lo, recv_hi = lo - span, lo
            recv = scratch[:span] if scratch is not None else torch.empty(
                span, dtype=buf.dtype
            )
            transport.sendrecv(buf[lo:hi], partner, recv, partner)
            buf[recv_lo:recv_hi] = recv
            lo, hi = min(lo, recv_lo), max(hi, recv_hi)
            d *= 2

    return tensor


def recursive_allreduce_fresh(tensor: torch.Tensor, transport: Transport) -> torch.Tensor:
    """Variant allocating a receive buffer per step. Benchmark comparator only."""
    return recursive_allreduce(tensor, transport, reuse_scratch=False)
