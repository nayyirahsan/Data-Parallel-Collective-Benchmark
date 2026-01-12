"""Synchronous parameter server: rank 0 reduces and broadcasts.

Two steps regardless of p -- latency-minimal -- but the server's link carries
p*N bytes in each direction, so it bottlenecks on bandwidth as p grows. Under
the send-side charging model in ``ShapedTransport`` the server pays beta for
each of its p-1 outbound copies, which is what makes that bottleneck appear.
"""
from __future__ import annotations

import torch

from ..transport.base import Transport

SERVER_RANK = 0


def ps_allreduce(tensor: torch.Tensor, transport: Transport) -> torch.Tensor:
    """In-place sum-all-reduce via a central server at rank 0."""
    p = transport.world_size
    if p == 1:
        return tensor

    rank = transport.rank
    flat = tensor.view(-1)

    if rank == SERVER_RANK:
        scratch = torch.empty_like(flat)
        # Gather and accumulate. Receives are posted one at a time: gloo
        # matches on (src, tag), so ordering across distinct sources is not
        # guaranteed and a shared scratch buffer must not be reused
        # concurrently.
        for src in range(1, p):
            transport.irecv(scratch, src).wait()
            flat += scratch
        handles = [transport.isend(flat, dst) for dst in range(1, p)]
        for h in handles:
            h.wait()
    else:
        transport.isend(flat, SERVER_RANK).wait()
        transport.irecv(flat, SERVER_RANK).wait()

    return tensor
