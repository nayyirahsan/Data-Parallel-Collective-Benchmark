"""Data-parallel training with a pluggable all-reduce.

Gradients are flattened into one contiguous buffer before synchronisation. That
is both the realistic thing to do -- per-tensor all-reduce on 62 ResNet-18
parameter tensors would pay the per-step cost 62 times -- and what makes the
collective benchmark relevant: the flattened buffer is a single 42.7 MiB message,
sitting well above every crossover measured in the H1 sweep.

The accelerator/host boundary is measured, not hidden. gloo is CPU-only, so on
MPS the gradient buffer must cross to host memory and back on every step. That
transfer is a real cost of this design and is reported as its own line in the
step breakdown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..collectives import ALGORITHMS
from ..transport.base import Transport


@dataclass
class StepBreakdown:
    """Where a training step's wall time went."""

    forward_ns: int = 0
    backward_ns: int = 0
    gather_ns: int = 0        # flatten grads into the contiguous buffer
    to_host_ns: int = 0       # accelerator -> host staging for gloo
    allreduce_ns: int = 0
    to_device_ns: int = 0     # host -> accelerator
    scatter_ns: int = 0       # write reduced grads back into .grad
    optimizer_ns: int = 0

    @property
    def total_ns(self) -> int:
        return (
            self.forward_ns + self.backward_ns + self.gather_ns + self.to_host_ns
            + self.allreduce_ns + self.to_device_ns + self.scatter_ns
            + self.optimizer_ns
        )

    @property
    def comm_ns(self) -> int:
        """Everything attributable to synchronisation, staging included."""
        return self.gather_ns + self.to_host_ns + self.allreduce_ns + \
            self.to_device_ns + self.scatter_ns

    def as_dict(self) -> dict:
        return {
            "forward_ns": self.forward_ns,
            "backward_ns": self.backward_ns,
            "gather_ns": self.gather_ns,
            "to_host_ns": self.to_host_ns,
            "allreduce_ns": self.allreduce_ns,
            "to_device_ns": self.to_device_ns,
            "scatter_ns": self.scatter_ns,
            "optimizer_ns": self.optimizer_ns,
            "total_ns": self.total_ns,
            "comm_ns": self.comm_ns,
        }


class GradientSynchroniser:
    """Flattens gradients, all-reduces them, and writes the mean back."""

    def __init__(self, model: nn.Module, transport: Transport, algo: str):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.transport = transport
        self.collective = ALGORITHMS[algo]
        self.world_size = transport.world_size
        numel = sum(p.numel() for p in self.params)
        self.nbytes = numel * 4

        # Two buffers, one per memory space. Gradients are packed into the
        # device-side buffer with on-device copies, then moved to the host in a
        # *single* transfer.
        #
        # The obvious implementation -- copying each parameter's gradient
        # straight into a host buffer -- issues one device-to-host transfer per
        # parameter tensor. Measured on ResNet-18 (62 tensors, 42.6 MiB) that
        # costs 23.4ms on MPS against 7.8ms for one fused transfer, a 3x
        # difference, because each small transfer pays its own launch and
        # synchronisation overhead. On CPU both are ~1ms, so the cost is
        # entirely the crossing, not the packing.
        device = self.params[0].device
        self.flat = torch.zeros(numel, dtype=torch.float32)
        self.device_flat = (
            self.flat if device.type == "cpu"
            else torch.zeros(numel, dtype=torch.float32, device=device)
        )
        self._staged = self.device_flat is not self.flat

    def _sync_device(self, device: torch.device) -> None:
        """Accelerator work is async; timings are meaningless without this."""
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def step(self, breakdown: StepBreakdown, device: torch.device) -> None:
        if self.world_size == 1:
            return

        t0 = time.perf_counter_ns()
        offset = 0
        for p in self.params:
            n = p.numel()
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            self.device_flat[offset : offset + n].copy_(grad.detach().reshape(-1))
            offset += n
        self._sync_device(device)
        t1 = time.perf_counter_ns()
        breakdown.gather_ns += t1 - t0

        if self._staged:
            self.flat.copy_(self.device_flat)
            self._sync_device(device)
        t2 = time.perf_counter_ns()
        breakdown.to_host_ns += t2 - t1
        self.collective(self.flat, self.transport)
        t3 = time.perf_counter_ns()
        breakdown.allreduce_ns += t3 - t2

        self.flat /= self.world_size
        if self._staged:
            self.device_flat.copy_(self.flat)
            self._sync_device(device)
        t4 = time.perf_counter_ns()
        breakdown.to_device_ns += t4 - t3

        offset = 0
        for p in self.params:
            n = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(self.device_flat[offset : offset + n].view_as(p.grad))
            offset += n
        self._sync_device(device)
        breakdown.scatter_ns += time.perf_counter_ns() - t4


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches,
    synchroniser: GradientSynchroniser | None,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> tuple[list[float], StepBreakdown]:
    """Run one pass over ``batches``. Returns per-step losses and a breakdown."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    breakdown = StepBreakdown()
    losses = []

    model.train()
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)

        t0 = time.perf_counter_ns()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter_ns()
        breakdown.forward_ns += t1 - t0

        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter_ns()
        breakdown.backward_ns += t2 - t1

        if synchroniser is not None:
            synchroniser.step(breakdown, device)

        t3 = time.perf_counter_ns()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        breakdown.optimizer_ns += time.perf_counter_ns() - t3

        losses.append(float(loss.detach().cpu()))

    return losses, breakdown
