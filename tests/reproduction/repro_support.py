"""Helpers shared by the reproduction tests: the manuscript cache and one-line comparisons."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def cache_root() -> Path:
    return Path(os.environ["INTERPRETFM_PAPER_CACHE"])


def cached_path(*parts: str) -> Optional[Path]:
    """A file of the manuscript cache, or ``None`` when it is not part of the configured cache."""
    p = cache_root().joinpath(*parts)
    return p if p.exists() else None



def report(label: str, observed, manuscript=None, cached=None, tol=None) -> str:
    """One line per compared number (shown with ``pytest -rA`` / ``-s``)."""
    parts = [f"{label}: observed {observed}"]
    if manuscript is not None:
        parts.append(f"manuscript {manuscript}")
    if cached is not None:
        parts.append(f"cached source {cached}")
    if tol is not None:
        parts.append(f"tolerance {tol}")
    line = " | ".join(parts)
    print(line)
    return line


def check(label: str, observed: float, manuscript=None, cached=None, tol: float = 0.005, tol_cached: Optional[float] = None):
    """Assert closeness to the manuscript value and (when available) to the cached source value."""
    line = report(label, observed, manuscript, cached, tol)
    if manuscript is not None:
        assert abs(float(observed) - float(manuscript)) <= tol, line
    if cached is not None:
        assert abs(float(observed) - float(cached)) <= (tol if tol_cached is None else tol_cached), line
