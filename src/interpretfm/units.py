"""Spatial units: the one place where coordinate units are converted.

A dataset declares ``dataset.spatial.um_per_unit`` (micrometres per stored coordinate unit).
Every stage works in micrometres: coordinates are converted once at ingest with
:func:`coords_to_um`, and parameters given in µm (radii, block sizes) are converted back to the
native unit only where an underlying 0.9 function needs native units, with
:func:`um_to_native`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from interpretfm.config import Spatial


def coords_to_um(coords: np.ndarray, um_per_unit: float) -> np.ndarray:
    """Convert native coordinates (``n_cells × 2`` or more) to micrometres."""
    if um_per_unit <= 0:
        raise ValueError(f"um_per_unit must be > 0, got {um_per_unit}")
    return np.asarray(coords, dtype=np.float64) * float(um_per_unit)


def um_to_native(value_um: float, um_per_unit: float) -> float:
    """Convert a length in micrometres (for example a radius) to native coordinate units."""
    if um_per_unit <= 0:
        raise ValueError(f"um_per_unit must be > 0, got {um_per_unit}")
    return float(value_um) / float(um_per_unit)


def native_to_um(value_native: float, um_per_unit: float) -> float:
    """Convert a length in native coordinate units to micrometres."""
    return float(value_native) * float(um_per_unit)


def adata_coords_um(adata, spatial: "Spatial") -> np.ndarray:
    """Coordinates of an AnnData object in micrometres, from the ``obsm`` key declared in the config."""
    if spatial.obsm not in adata.obsm:
        raise KeyError(f"dataset.spatial.obsm: '{spatial.obsm}' not found in adata.obsm (keys: {list(adata.obsm)})")
    coords = np.asarray(adata.obsm[spatial.obsm])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial.obsm}'] must be n_cells × 2 (or more), got shape {coords.shape}")
    return coords_to_um(coords[:, :2], spatial.um_per_unit)
