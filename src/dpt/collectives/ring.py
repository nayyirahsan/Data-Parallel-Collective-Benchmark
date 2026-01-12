"""Ring all-reduce: reduce-scatter followed by all-gather.

2(p-1) serially dependent steps, each moving N/p bytes. Bandwidth-optimal --
each rank sends exactly 2*((p-1)/p)*N bytes, the theoretical minimum -- at the
cost of a step count linear in p. That linear step count is exactly what H2
predicts makes it fragile to tail latency: every step is gated by its slowest
participant, so per-step tail noise compounds 2(p-1) times.
"""
from __future__ import annotations

import torch

from ..transport.base import Transport
from .util import PaddedFlat


def ring_allreduce(tensor: torch.Tensor, transport: Transport) -> torch.Tensor:
    """In-place sum-all-reduce of ``tensor`` across all ranks."""
    p = transport.world_size
    if p == 1:
        return tensor

    rank = transport.rank
    nxt, prv = (rank + 1) % p, (rank - 1) % p

    with PaddedFlat(tensor, p) as pf:
        chunks = pf.buffer.view(p, pf.chunk_size)
        recv = torch.empty(pf.chunk_size, dtype=pf.buffer.dtype)

        # Reduce-scatter: after p-1 steps rank r holds the fully reduced
        # chunk (r+1) % p.
        for step in range(p - 1):
            send_idx = (rank - step) % p
            recv_idx = (rank - step - 1) % p
            transport.sendrecv(chunks[send_idx], nxt, recv, prv)
            chunks[recv_idx] += recv

        # All-gather: circulate the reduced chunks back around the ring.
        for step in range(p - 1):
            send_idx = (rank - step + 1) % p
            recv_idx = (rank - step) % p
            transport.sendrecv(chunks[send_idx], nxt, recv, prv)
            chunks[recv_idx].copy_(recv)

    return tensor
