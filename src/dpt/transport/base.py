"""Transport abstraction.

Scope boundary (defensible decision #2)
---------------------------------------
This project implements collective *algorithms*, not a wire protocol. The
point-to-point substrate (``isend``/``irecv``/``barrier``) is delegated to
``torch.distributed``'s gloo backend; every collective built on top of it in
``dpt.collectives`` is written from scratch. Reimplementing sockets would have
consumed weeks and measured nothing the hypotheses care about.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Transport(ABC):
    """Point-to-point message passing over ``world_size`` ranks."""

    @property
    @abstractmethod
    def rank(self) -> int: ...

    @property
    @abstractmethod
    def world_size(self) -> int: ...

    @abstractmethod
    def isend(self, tensor: torch.Tensor, dst: int) -> Any:
        """Begin sending ``tensor`` to ``dst``. Returns a waitable handle."""

    @abstractmethod
    def irecv(self, tensor: torch.Tensor, src: int) -> Any:
        """Begin receiving into ``tensor`` from ``src``. Returns a handle."""

    @abstractmethod
    def barrier(self) -> None: ...

    def sendrecv(
        self,
        send_tensor: torch.Tensor,
        dst: int,
        recv_tensor: torch.Tensor,
        src: int,
    ) -> None:
        """Simultaneous send and receive.

        Posting the receive first avoids the deadlock that paired blocking
        sends would produce in a ring.
        """
        recv_handle = self.irecv(recv_tensor, src)
        send_handle = self.isend(send_tensor, dst)
        recv_handle.wait()
        send_handle.wait()
