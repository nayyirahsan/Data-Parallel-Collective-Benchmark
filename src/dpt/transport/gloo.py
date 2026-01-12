"""torch.distributed gloo point-to-point transport."""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from .base import Transport


class GlooTransport(Transport):
    """Thin wrapper over gloo p2p primitives.

    gloo is CPU-only, so callers holding accelerator tensors must stage through
    host memory. That transfer is a real cost of this design and is measured
    explicitly in the end-to-end training benchmark rather than hidden.
    """

    def __init__(self, rank: int, world_size: int, init_file: str | None = None):
        if not dist.is_initialized():
            if init_file is not None:
                init_method = f"file://{os.path.abspath(init_file)}"
            else:
                init_method = "env://"
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def isend(self, tensor: torch.Tensor, dst: int) -> Any:
        return dist.isend(tensor, dst)

    def irecv(self, tensor: torch.Tensor, src: int) -> Any:
        return dist.irecv(tensor, src)

    def barrier(self) -> None:
        dist.barrier()

    @staticmethod
    def shutdown() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
