"""Guardrails in the sweep driver.

The calibration and noise-floor experiments established regimes where the
apparatus cannot support a measurement. Those limits are enforced in code rather
than left in prose, so an invalid experiment fails loudly instead of producing
a plausible-looking number.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dpt.bench.sweep import load_spec


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_rejects_jitter_below_calibrated_alpha_floor(tmp_path):
    cfg = _write(tmp_path, """
        name: bad
        algorithms: [ring]
        world_sizes: [8]
        bytes: [4096]
        alpha_ns: [50000]
        jitter:
          - {probability: 0.01, multiplier: 10.0}
    """)
    with pytest.raises(ValueError, match="confound"):
        load_spec(cfg)


def test_allows_jitter_above_alpha_floor(tmp_path):
    cfg = _write(tmp_path, """
        name: fine
        algorithms: [ring]
        world_sizes: [8]
        bytes: [4096]
        alpha_ns: [200000]
        jitter:
          - {probability: 0.01, multiplier: 10.0}
    """)
    assert len(load_spec(cfg).configs) == 1


def test_rejects_world_size_beyond_calibrated_limit(tmp_path):
    cfg = _write(tmp_path, """
        name: toobig
        algorithms: [ring]
        world_sizes: [12]
        bytes: [4096]
        alpha_ns: [0]
    """)
    with pytest.raises(ValueError, match="calibrated limit"):
        load_spec(cfg)


def test_clean_runs_allowed_at_any_alpha(tmp_path):
    """The floor experiment needs alpha=0, which only jitter runs are barred from."""
    cfg = _write(tmp_path, """
        name: floor
        algorithms: [ring, ps]
        world_sizes: [2, 4, 8]
        bytes: [4096, 65536]
        alpha_ns: [0]
    """)
    assert len(load_spec(cfg).configs) == 12


def test_grid_is_full_product(tmp_path):
    cfg = _write(tmp_path, """
        name: grid
        algorithms: [ring, ps]
        world_sizes: [2, 4]
        bytes: [4096, 65536]
        alpha_ns: [0, 200000]
        beta_ns_per_byte: [0.0, 0.4]
    """)
    assert len(load_spec(cfg).configs) == 2 * 2 * 2 * 2 * 2
