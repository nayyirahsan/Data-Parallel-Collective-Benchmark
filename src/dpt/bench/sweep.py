"""Declarative sweep driver.

    python -m dpt.bench.sweep --config experiments/floor.yaml

Experiments live in YAML so the grid is a committed artefact rather than an
argv incantation that has to be remembered to reproduce a figure.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from ..io import write_rows
from .runner import Config, RunSpec, critical_path, run

# The calibration gate (docs/calibration.md) found the delay shim's p99 error
# reaches +33% at p=8, alpha=50us, and blows up entirely at p=12. Tail-statistic
# experiments in that regime would confound injected jitter with scheduler
# noise, so they are refused rather than silently reported.
JITTER_MIN_ALPHA_NS = 200_000
MAX_WORLD_SIZE = 8


def _as_list(value):
    return value if isinstance(value, list) else [value]


def load_spec(path: Path) -> RunSpec:
    raw = yaml.safe_load(path.read_text())

    jitters = _as_list(raw.get("jitter", [{"probability": 0.0, "multiplier": 1.0}]))
    configs = []
    for algo, world_size, nbytes, alpha, beta, jit in itertools.product(
        _as_list(raw["algorithms"]),
        _as_list(raw["world_sizes"]),
        _as_list(raw["bytes"]),
        _as_list(raw["alpha_ns"]),
        _as_list(raw.get("beta_ns_per_byte", 0.0)),
        jitters,
    ):
        q = float(jit.get("probability", 0.0))
        if world_size > MAX_WORLD_SIZE:
            raise ValueError(
                f"world_size {world_size} exceeds the calibrated limit of "
                f"{MAX_WORLD_SIZE}; see docs/calibration.md"
            )
        if q > 0 and alpha < JITTER_MIN_ALPHA_NS:
            raise ValueError(
                f"jitter config at alpha={alpha}ns is below the "
                f"{JITTER_MIN_ALPHA_NS}ns floor where the shim's own p99 error "
                f"({'33%' } at p=8) would confound the measurement; "
                "see docs/calibration.md"
            )
        configs.append(
            Config(
                algo=algo,
                world_size=int(world_size),
                nbytes=int(nbytes),
                alpha_ns=float(alpha),
                beta_ns_per_byte=float(beta),
                jitter_q=q,
                jitter_k=float(jit.get("multiplier", 1.0)),
                jitter_seed=int(jit.get("seed", raw.get("seed", 0))),
            )
        )

    return RunSpec(
        name=raw.get("name", path.stem),
        configs=configs,
        trials=int(raw.get("trials", 5)),
        warmup=int(raw.get("warmup", 5)),
        iters=int(raw.get("iters", 30)),
        seed=int(raw.get("seed", 0)),
        notes=raw.get("notes", ""),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    spec = load_spec(args.config)
    print(f"experiment: {spec.name}")
    if spec.notes:
        print(f"  {spec.notes}")
    print(
        f"  {len(spec.configs)} configs x {spec.trials} trials x {spec.iters} iters"
    )

    t0 = time.perf_counter()
    rows = run(spec)
    elapsed = time.perf_counter() - t0

    # Raw per-rank rows are the bulk of the data and compress ~10x, so they
    # are stored gzipped; readers handle either form.
    raw_path = write_rows(rows, args.out_dir / f"{spec.name}_raw.csv", compress=True)
    cp_path = write_rows(critical_path(rows), args.out_dir / f"{spec.name}_critpath.csv")
    print(f"\n{len(rows)} rows in {elapsed:.1f}s")
    print(f"  raw           -> {raw_path}")
    print(f"  critical path -> {cp_path}")


if __name__ == "__main__":
    main()
