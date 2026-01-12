"""Discrete-event model of collective dependency structure.

Why this exists
---------------
H2 is a claim about tail latency at p >= 4, where ring's 2(p-1) serial steps
exceed PS's p. The noise-floor experiment (docs/findings-h2.md) showed the
single-machine apparatus cannot support a p99 measurement there: its own clean
p99/p50 is 2-10x at p >= 4, driven by contention between co-located processes,
which swamps any injected effect. Only p=2 is clean -- and at p=2 ring and PS
have *identical* step counts, so the mechanism under test does not even exist
there.

Rather than report a p99 that the apparatus cannot support, the dependency
structure is modelled directly. Each algorithm is a DAG of message events; each
message costs alpha + bytes*beta scaled by a jitter draw; the collective
completes when the last rank's chain completes. This isolates exactly the
structural question H2 asks -- does a longer chain of slowest-of-p steps
compound tail latency? -- with no contention, and cheaply enough to run the
100k+ iterations a stable p99 needs.

The model is an abstraction and is only trusted because it is validated against
measured degradation in the regime where the apparatus is clean
(``validate.py``). Simulated numbers are always labelled as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis.fit import step_bytes


@dataclass(frozen=True)
class SimConfig:
    algo: str
    world_size: int
    nbytes: int
    alpha_ns: float
    beta_ns_per_byte: float
    jitter_q: float = 0.0
    jitter_k: float = 1.0


def _costs(cfg: SimConfig, iters: int, rng: np.random.Generator) -> np.ndarray:
    """Per-(iteration, rank) message cost, with jitter applied.

    Jitter keys on (rank, iteration) exactly as ``dpt.jitter.JitterModel`` does,
    so a slow rank is slow for every message it sends in that iteration -- the
    same semantics the real runs use.
    """
    base = cfg.alpha_ns + step_bytes(cfg.algo, cfg.nbytes, cfg.world_size) * cfg.beta_ns_per_byte
    cost = np.full((iters, cfg.world_size), base, dtype=float)
    if cfg.jitter_q > 0:
        hit = rng.random((iters, cfg.world_size)) < cfg.jitter_q
        cost[hit] *= cfg.jitter_k
    return cost


def _simulate_ring(cost: np.ndarray, p: int) -> np.ndarray:
    """2(p-1) steps; rank r receives from r-1 each step."""
    iters = cost.shape[0]
    clock = np.zeros((iters, p))
    for _ in range(2 * (p - 1)):
        prev = np.roll(np.arange(p), 1)
        clock = np.maximum(clock, clock[:, prev] + cost[:, prev])
    return clock.max(axis=1)


def _simulate_recursive(cost: np.ndarray, p: int) -> np.ndarray:
    """2*log2(p) steps; rank r exchanges with r XOR d."""
    iters = cost.shape[0]
    clock = np.zeros((iters, p))
    ranks = np.arange(p)
    distances = [1 << k for k in range(int(math.log2(p)))]
    for d in distances + distances[::-1]:
        partner = ranks ^ d
        clock = np.maximum(clock, clock[:, partner] + cost[:, partner])
    return clock.max(axis=1)


def _simulate_ps(cost: np.ndarray, p: int) -> np.ndarray:
    """Serial ingest at the server, then a serial (p-1) broadcast."""
    iters = cost.shape[0]
    clock = np.zeros((iters, p))
    server = np.zeros(iters)
    for w in range(1, p):
        server = np.maximum(server, clock[:, w] + cost[:, w])
    for w in range(1, p):
        server = server + cost[:, 0]
        clock[:, w] = np.maximum(clock[:, w], server)
    clock[:, 0] = server
    return clock.max(axis=1)


_SIMULATORS = {
    "ring": _simulate_ring,
    "recursive": _simulate_recursive,
    "ps": _simulate_ps,
}


def simulate(cfg: SimConfig, iters: int = 100_000, seed: int = 0) -> np.ndarray:
    """Completion time per iteration, in nanoseconds."""
    if cfg.algo == "recursive" and (cfg.world_size & (cfg.world_size - 1)):
        raise ValueError("recursive requires a power-of-two world size")
    rng = np.random.default_rng(seed)
    cost = _costs(cfg, iters, rng)
    return _SIMULATORS[cfg.algo](cost, cfg.world_size)


def degradation(cfg: SimConfig, iters: int = 100_000, seed: int = 0) -> dict:
    """p99 and median degradation of ``cfg`` against its own clean baseline."""
    clean_cfg = SimConfig(
        algo=cfg.algo, world_size=cfg.world_size, nbytes=cfg.nbytes,
        alpha_ns=cfg.alpha_ns, beta_ns_per_byte=cfg.beta_ns_per_byte,
    )
    clean = simulate(clean_cfg, iters, seed)
    jittered = simulate(cfg, iters, seed)
    return {
        "algo": cfg.algo,
        "world_size": cfg.world_size,
        "jitter_q": cfg.jitter_q,
        "jitter_k": cfg.jitter_k,
        "clean_p99_ns": float(np.percentile(clean, 99)),
        "jittered_p99_ns": float(np.percentile(jittered, 99)),
        "p99_ratio": float(np.percentile(jittered, 99) / np.percentile(clean, 99)),
        "median_ratio": float(np.median(jittered) / np.median(clean)),
        "steps": {"ring": 2 * (cfg.world_size - 1), "ps": cfg.world_size,
                  "recursive": 2 * int(math.log2(cfg.world_size))}[cfg.algo],
    }
