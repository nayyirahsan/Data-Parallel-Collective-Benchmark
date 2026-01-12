"""Measurement harness.

Design note: a trial is a process launch, not an iteration
---------------------------------------------------------
Reporting the spread across iterations *within* one launch is the classic error
in this kind of benchmark: it measures only steady-state jitter and hides
launch-to-launch variance (rendezvous cost, page-cache state, core assignment),
which is often the larger term. Here a trial means a genuinely cold spawn of p
processes, and confidence intervals are taken across trials.

Spawning is expensive (~2-3s), so each trial runs the *whole* config list inside
one launch rather than one config per launch. To stop that reintroducing an
ordering artefact -- thermal drift or cache warming favouring whichever config
ran first -- the config order is shuffled per trial with a seeded permutation.

Output is long format, one row per (config, trial, iteration, rank). Aggregation
to a critical path (max over ranks) and to summary statistics happens in
analysis, so the raw data stays interrogable and no decision is baked in here.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator

import torch

from ..collectives import ALGORITHMS, recursive_supported
from ..jitter import JitterModel
from ..transport.base import Transport
from ..transport.shaped import ShapedTransport
from .launch import run_ranks

DTYPE = torch.float32
ELEM_SIZE = 4


@dataclass(frozen=True)
class Config:
    """One measurable point in the experiment space."""

    algo: str
    world_size: int
    nbytes: int
    alpha_ns: float
    beta_ns_per_byte: float
    jitter_q: float = 0.0
    jitter_k: float = 1.0
    jitter_seed: int = 0

    @property
    def numel(self) -> int:
        return max(1, self.nbytes // ELEM_SIZE)

    @property
    def supported(self) -> bool:
        if self.algo.startswith("recursive"):
            return recursive_supported(self.world_size)
        return True

    def jitter_model(self) -> JitterModel:
        return JitterModel(
            probability=self.jitter_q, multiplier=self.jitter_k, seed=self.jitter_seed
        )

    def key(self) -> str:
        return (
            f"{self.algo}|p{self.world_size}|n{self.nbytes}|a{self.alpha_ns:g}"
            f"|b{self.beta_ns_per_byte:g}|q{self.jitter_q:g}|k{self.jitter_k:g}"
        )


@dataclass
class RunSpec:
    """A full experiment: the config list plus repetition policy."""

    name: str
    configs: list[Config]
    trials: int = 5
    warmup: int = 5
    iters: int = 30
    seed: int = 0
    notes: str = ""


def _rank_job(transport: Transport, configs, warmup, iters, trial, seed):
    """Run every config on this rank and return long-format timing rows."""
    order = list(range(len(configs)))
    random.Random(seed * 10_000 + trial).shuffle(order)

    rows = []
    for slot, idx in enumerate(order):
        cfg: Config = configs[idx]
        collective = ALGORITHMS[cfg.algo]
        shaped = ShapedTransport(
            transport,
            alpha_ns=cfg.alpha_ns,
            beta_ns_per_byte=cfg.beta_ns_per_byte,
            jitter=cfg.jitter_model(),
        )
        base = torch.ones(cfg.numel, dtype=DTYPE)

        # Warmup is charged the same injected cost as the timed region, so the
        # transport, allocator and gloo connections reach the same state they
        # will be in when measurement starts.
        for w in range(warmup):
            shaped.set_step(-(w + 1))  # negative steps: distinct jitter draws
            collective(base.clone(), shaped)

        transport.barrier()
        for it in range(iters):
            shaped.set_step(it)
            shaped.reset_counters()
            tensor = base.clone()
            transport.barrier()
            start = time.perf_counter_ns()
            collective(tensor, shaped)
            elapsed = time.perf_counter_ns() - start
            rows.append(
                {
                    **asdict(cfg),
                    "trial": trial,
                    "iter": it,
                    "slot": slot,
                    "rank": transport.rank,
                    "duration_ns": elapsed,
                    "injected_ns": shaped.injected_ns,
                    "messages": shaped.messages_sent,
                }
            )
    return rows


def run(spec: RunSpec, progress: bool = True) -> list[dict]:
    """Execute ``spec`` and return long-format rows.

    Configs are grouped by world size because a process launch is tied to one.
    """
    usable = [c for c in spec.configs if c.supported]
    skipped = len(spec.configs) - len(usable)
    if skipped and progress:
        print(f"  ({skipped} configs skipped: unsupported world size)")

    by_world: dict[int, list[Config]] = {}
    for cfg in usable:
        by_world.setdefault(cfg.world_size, []).append(cfg)

    rows: list[dict] = []
    for world_size in sorted(by_world):
        configs = by_world[world_size]
        for trial in range(spec.trials):
            t0 = time.perf_counter()
            per_rank = run_ranks(
                world_size,
                _rank_job,
                configs,
                spec.warmup,
                spec.iters,
                trial,
                spec.seed,
                timeout=1800,
            )
            for chunk in per_rank:
                rows.extend(chunk)
            if progress:
                print(
                    f"  p={world_size} trial {trial + 1}/{spec.trials}: "
                    f"{len(configs)} configs in {time.perf_counter() - t0:.1f}s"
                )
    return rows


def critical_path(rows: list[dict]) -> list[dict]:
    """Collapse per-rank rows to one row per (config, trial, iter).

    A collective completes when its slowest participant completes, so the
    critical path is the max over ranks -- not the mean, which would flatter
    exactly the algorithms H2 predicts are fragile.
    """
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["algo"], row["world_size"], row["nbytes"], row["alpha_ns"],
               row["beta_ns_per_byte"], row["jitter_q"], row["jitter_k"],
               row["trial"], row["iter"])
        buckets.setdefault(key, []).append(row)

    out = []
    for key, group in buckets.items():
        slowest = max(group, key=lambda r: r["duration_ns"])
        out.append(
            {
                **{k: v for k, v in slowest.items() if k not in ("rank", "slot")},
                "ranks_observed": len(group),
                "spread_ns": slowest["duration_ns"] - min(r["duration_ns"] for r in group),
            }
        )
    return out
