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



def dev_table(path: Path, tau: float):
    """A cached held-out F1 table restricted to threshold ``tau`` (the manuscript's column is ``threshold_pct``)."""
    import numpy as np
    import pandas as pd

    cols = ["concept", "feature", "threshold_pct", "f1", "tp"]
    parts = [c[np.isclose(c.threshold_pct, tau)] for c in pd.read_csv(path, usecols=cols, chunksize=2_000_000)]
    return pd.concat(parts, ignore_index=True).rename(columns={"threshold_pct": "threshold"})


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


def known_difference(label: str) -> Optional[dict]:
    """The entry of ``expected.KNOWN_DIFFERENCES`` for this comparison (printed), or ``None``."""
    from expected import KNOWN_DIFFERENCES

    known = KNOWN_DIFFERENCES.get(label)
    if known is not None:
        print(f"{label}: listed in KNOWN_DIFFERENCES, not asserted against the printed value ({known['note']})")
    return known


def check(label: str, observed: float, manuscript=None, cached=None, tol: float = 0.005, tol_cached: Optional[float] = None):
    """Assert closeness to the manuscript value and (when available) to the cached source value; a comparison
    listed in ``expected.KNOWN_DIFFERENCES`` is reported (and checked against the pipeline's own value when the
    entry gives one)."""
    line = report(label, observed, manuscript, cached, tol)
    known = known_difference(label)
    if known is not None:
        if known.get("observed") is not None:
            assert abs(float(observed) - float(known["observed"])) <= known.get("tol", tol), f"{line} | pipeline value {known['observed']}"
        return
    bad = []
    if manuscript is not None and abs(float(observed) - float(manuscript)) > tol:
        bad.append("manuscript")
    if cached is not None and abs(float(observed) - float(cached)) > (tol if tol_cached is None else tol_cached):
        bad.append("cached source")
    if bad and os.environ.get("INTERPRETFM_REPRO_REPORT_ALL"):
        # survey mode: list every comparison outside its tolerance instead of stopping at the first
        print(f"OUT OF TOLERANCE ({', '.join(bad)}): {line}")
        return
    assert not bad, line
