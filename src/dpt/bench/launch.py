"""Multi-process launcher for rank-parallel work.

Uses spawn rather than fork: fork after torch has initialised is unsafe on
macOS, and spawn also gives each trial a genuinely cold process, which is what
"trial" has to mean for the variance estimates to be honest.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import traceback
from typing import Any, Callable

from ..transport.gloo import GlooTransport


def _entry(rank, world_size, init_file, fn, args, kwargs, queue):
    try:
        transport = GlooTransport(rank, world_size, init_file=init_file)
        result = fn(transport, *args, **kwargs)
        queue.put(("ok", rank, result))
    except Exception:
        queue.put(("err", rank, traceback.format_exc()))
    finally:
        try:
            GlooTransport.shutdown()
        except Exception:
            pass


def run_ranks(
    world_size: int,
    fn: Callable[..., Any],
    *args: Any,
    timeout: float = 300.0,
    **kwargs: Any,
) -> list[Any]:
    """Run ``fn(transport, *args, **kwargs)`` on ``world_size`` processes.

    Returns results ordered by rank. Raises RuntimeError with the remote
    traceback if any rank fails.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "rendezvous")
        procs = [
            ctx.Process(
                target=_entry,
                args=(r, world_size, init_file, fn, args, kwargs, queue),
            )
            for r in range(world_size)
        ]
        for p in procs:
            p.start()

        collected = []
        try:
            for _ in range(world_size):
                collected.append(queue.get(timeout=timeout))
        finally:
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()

    failures = [(r, tb) for status, r, tb in collected if status == "err"]
    if failures:
        rank, tb = failures[0]
        raise RuntimeError(f"rank {rank} failed:\n{tb}")

    results = {r: val for _, r, val in collected}
    return [results[r] for r in range(world_size)]


# Caller requirement: because the spawn start method re-imports the caller's
# __main__ module in every child, any script invoking run_ranks must guard its
# entry point with ``if __name__ == "__main__":``. Without it the child re-runs
# the script top level and the launch recurses until it hangs.
