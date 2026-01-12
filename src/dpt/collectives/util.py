"""Shared buffer handling for collectives."""
from __future__ import annotations

import torch


class PaddedFlat:
    """Context manager exposing ``tensor`` as a flat buffer padded to a multiple of ``p``.

    Ring and recursive algorithms both need the vector to divide evenly across
    ranks. Padding with zeros is safe for sum-reduction and keeps the algorithms
    free of ragged-chunk special cases. On exit the (possibly padded) buffer is
    copied back into the caller's tensor.
    """

    def __init__(self, tensor: torch.Tensor, p: int):
        self._tensor = tensor
        self._flat = tensor.view(-1)
        self.numel = self._flat.numel()
        self.chunk_size = (self.numel + p - 1) // p
        self.padded_size = self.chunk_size * p
        self._needs_pad = self.padded_size != self.numel

    def __enter__(self) -> "PaddedFlat":
        if self._needs_pad:
            self.buffer = torch.zeros(self.padded_size, dtype=self._flat.dtype)
            self.buffer[: self.numel] = self._flat
        else:
            self.buffer = self._flat.contiguous()
        return self

    def __exit__(self, *exc) -> None:
        if self._needs_pad:
            self._flat.copy_(self.buffer[: self.numel])
        elif self.buffer.data_ptr() != self._flat.data_ptr():
            self._flat.copy_(self.buffer)
