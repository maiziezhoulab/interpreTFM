"""Neighbourhood size as a function of the radius, and the radius that gives a target neighbourhood size.

The mean number of *other* cells within a radius ``r`` of a cell (``2 · |pairs within r| / n``,
the mean degree of the neighbour graph) is what makes neighbourhoods on different platforms
comparable: the same physical radius holds very different numbers of cells when cell densities
or segmentation differ, and the same radius in native units differs in µm (CosMx pixels versus
Xenium µm). :func:`radius_curve` gives the curve on a grid of radii; :func:`match_radius`
inverts it: the radius at which a section reaches a target mean count, by linear interpolation
on a grid of radii in steps of ``step`` (native units), rounded to ``round_decimals`` decimals, with the
exact count at the rounded radius recomputed. The manuscript's rule (CosMx grid 0.5 px, Xenium
0.1 µm, radius rounded to 0.1 unit) is this function with those steps.

Cross-section matching is a recipe, not a stage: compute the mean count of the reference section
at its radius with :func:`mean_other_cells`, then ``match_radius(coords_of_the_other_section,
that_count)``; or set ``stages.lr_screen.match_neighbors_to`` to the count and let the stage do
the matching for its own section (see ``docs/pipeline/stage_lr_screen.md``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def _tree(coords: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(coords, dtype=np.float64)[:, :2])


def mean_other_cells(coords: np.ndarray, radius: float) -> float:
    """Mean number of other cells within ``radius`` (same units as ``coords``): ``2 · |pairs| / n``
    (counted with ``count_neighbors``, so no pair list is materialised)."""
    tree = _tree(coords)
    n = tree.n
    return (float(tree.count_neighbors(tree, float(radius))) - n) / n


def radius_curve(coords: np.ndarray, radii: Sequence[float]) -> np.ndarray:
    """Mean number of other cells within each radius of ``radii`` (sorted ascending), same units as ``coords``."""
    tree = _tree(coords)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim != 1 or len(r) == 0 or (np.diff(r) < 0).any():
        raise ValueError("radii must be a non-empty ascending sequence")
    cnt = tree.count_neighbors(tree, r)  # ordered pairs within r, self included
    return (np.asarray(cnt, dtype=np.float64) - tree.n) / tree.n


def interpolate_radius(radii: Sequence[float], curve: Sequence[float], target: float) -> float:
    """Linear interpolation of the radius at which the monotone ``curve`` reaches ``target``
    (between the last grid point below the target and the first at or above it)."""
    radii, curve = np.asarray(radii, dtype=np.float64), np.asarray(curve, dtype=np.float64)
    if not (curve[0] < target <= curve[-1]):
        raise ValueError(f"target {target} is outside the curve range [{curve[0]}, {curve[-1]}]")
    j = int(np.searchsorted(curve, target))  # first index with curve >= target
    r0, r1, y0, y1 = radii[j - 1], radii[j], curve[j - 1], curve[j]
    return float(r0 + (target - y0) * (r1 - r0) / (y1 - y0)) if y1 > y0 else float(r1)


def match_radius(coords: np.ndarray, target: float, step: float = 0.1, round_decimals: int = 1, r_max: Optional[float] = None,
                 max_points: int = 64) -> Dict[str, Any]:
    """The radius (native units of ``coords``) whose mean other-cell count equals ``target``.

    The curve is evaluated on multiples of ``step``: a geometric bracket first, then the bracket is
    refined until at most ``max_points`` grid steps remain, so the two grid points used by the
    interpolation are the same as on the full ``step`` grid. The result is rounded to
    ``round_decimals`` decimals and the exact count at the rounded radius is returned.
    ``r_max`` bounds the search (default: the diagonal of the coordinate extent).
    """
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    if target <= 0:
        raise ValueError(f"match_neighbors_to: the target mean count must be > 0, got {target}")
    if step <= 0:
        raise ValueError(f"the grid step must be > 0, got {step}")
    if r_max is None:
        ext = coords.max(0) - coords.min(0)
        r_max = float(np.hypot(ext[0], ext[1]))
    n_steps_max = int(np.ceil(r_max / step))
    if n_steps_max < 1:
        raise ValueError("r_max is below one grid step")
    # geometric bracket on multiples of step: k = 1, 2, 4, ... (in grid steps)
    ks = [1]
    while ks[-1] < n_steps_max:
        ks.append(min(ks[-1] * 2, n_steps_max))
    k_grid = np.asarray(sorted(set(ks)), dtype=np.int64)
    curve = radius_curve(coords, k_grid * step)
    if target > curve[-1]:
        raise ValueError(f"match_neighbors_to: target {target} is above the mean count {curve[-1]:.3g} at the largest radius {k_grid[-1] * step:g}; "
                         "raise r_max or lower the target")
    if target <= curve[0]:
        lo_k, hi_k = 0, int(k_grid[0])
        lo_y = 0.0
    else:
        j = int(np.searchsorted(curve, target))
        lo_k, hi_k, lo_y = int(k_grid[j - 1]), int(k_grid[j]), float(curve[j - 1])
    hi_y = float(curve[int(np.searchsorted(k_grid, hi_k))])
    n_eval = len(k_grid)
    while hi_k - lo_k > max_points:
        m = int(np.ceil((hi_k - lo_k) / max_points))
        inner = np.arange(lo_k + m, hi_k, m, dtype=np.int64)
        if len(inner) == 0:
            break
        yv = radius_curve(coords, inner * step)
        n_eval += len(inner)
        pts_k = np.concatenate([[lo_k], inner, [hi_k]])
        pts_y = np.concatenate([[lo_y], yv, [hi_y]])
        j = int(np.searchsorted(pts_y, target))
        lo_k, hi_k, lo_y, hi_y = int(pts_k[j - 1]), int(pts_k[j]), float(pts_y[j - 1]), float(pts_y[j])
    fine_k = np.arange(lo_k, hi_k + 1, dtype=np.int64)
    fine_r = fine_k * step
    if lo_k == 0:
        fine_y = np.concatenate([[0.0], radius_curve(coords, fine_r[1:])]) if len(fine_r) > 1 else np.asarray([0.0])
        n_eval += len(fine_r) - 1
    else:
        fine_y = radius_curve(coords, fine_r)
        n_eval += len(fine_r)
    r = interpolate_radius(fine_r, fine_y, target)
    r_round = round(r, int(round_decimals))
    return {"radius": float(r_round), "radius_unrounded": float(r), "target": float(target), "mean_other_cells": mean_other_cells(coords, r_round),
            "step": float(step), "round_decimals": int(round_decimals), "n_grid_evaluations": int(n_eval), "n_cells": int(coords.shape[0]),
            "rule": "mean other cells within r = 2 * |pairs within r| / n; r by linear interpolation on the step grid, rounded"}


__all__ = ["interpolate_radius", "match_radius", "mean_other_cells", "radius_curve"]
