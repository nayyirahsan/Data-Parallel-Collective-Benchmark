"""CSV helpers with transparent gzip support.

Raw per-rank output is the bulk of this repo's data (~37 MB uncompressed) and
compresses about 10x, so it is stored gzipped. Readers accept either form, which
keeps the raw rows committed and re-analysable rather than forcing a choice
between repo size and reproducibility.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Iterator


def open_text(path: Path, mode: str = "rt"):
    """Open ``path``, decompressing if it is gzipped."""
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def resolve(path: Path) -> Path:
    """Return ``path``, or its .gz sibling if only that exists."""
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    return path


def read_rows(path: Path) -> Iterator[dict]:
    with open_text(resolve(path)) as fh:
        yield from csv.DictReader(fh)


def write_rows(rows: list[dict], path: Path, compress: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress and path.suffix != ".gz":
        path = path.with_suffix(path.suffix + ".gz")
    with open_text(path, "wt") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
