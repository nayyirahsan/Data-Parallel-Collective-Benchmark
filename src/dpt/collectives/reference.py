"""gloo's own all-reduce -- correctness ground truth, not a benchmark arm.

This bypasses ``ShapedTransport`` entirely and therefore sees no injected link
cost, so its timings are not comparable to the other arms. It exists so every
from-scratch collective can be checked against a known-good implementation.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from ..transport.base import Transport


def reference_allreduce(tensor: torch.Tensor, transport: Transport) -> torch.Tensor:
    if transport.world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
