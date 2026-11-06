"""Spatial niches from cell-level features, with the expression-only and label-composition
baselines and the seed-stability summary.

The niche pipeline is the manuscript's (:mod:`interp_pipeline.downstream.context`): for every
cell the mean feature vector of its neighbours within a radius (including itself; ``m``),
optionally concatenated with the cell's own features (``xm``), standardized, reduced to
``n_pcs`` principal components and clustered with k-means or a full-covariance Gaussian
mixture. :func:`niche_sweep` runs every combination of radius, space, method, number of
clusters and seed of a :class:`NicheGrid`, drops configurations whose smallest cluster is
below ``min_cluster`` cells and scores the rest with

    S = w_ari · ARI_seed + w_purity_z · clip(z, ±purity_z_clip) + w_sep · separability,

where ARI_seed is the mean adjusted Rand index between the first seed and the others, z the
neighbourhood purity against a tile-shuffle null (labels permuted within tiles of
``tile_mult`` × radius) and separability the mean cosine distance between cluster centroids.
:func:`select_config` picks the highest score; :func:`cluster_niches` reproduces the labels of
one configuration and :func:`niche_validation` scores them (purity, null, z, ARI/NMI against
reference labels).

Radii are given in micrometres and converted to the coordinate unit of the dataset with
:func:`interpretfm.units.um_to_native` (the converted value is rounded to 1e-6 units to absorb
floating-point error of the conversion); neighbourhoods and tiles are computed in the native
unit, which is equivalent to working in µm.

Baselines: :func:`expression_svd` (50 truncated-SVD components of log-normalized expression,
run through the same sweep and score, and at the model's selected configuration),
:func:`label_composition` (one-hot label fractions over the radius neighbours, PCA to 15
components, k-means with 20 initialisations; the composition excludes the cell itself) and
the optional :func:`expression_variant` preprocessing variants. :func:`seed_stability` is the
median seed-to-seed ARI over the finer configurations (k in {4, 5}) of a sweep table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from interp_pipeline.downstream.context.cluster import ClusterConfig, build_cluster_matrix, cluster_labels, pca_embed
from interp_pipeline.downstream.context.metrics import (
    mean_neighbor_purity,
    min_cluster_size,
    separability_cosine,
    tile_shuffle_null_purity,
)
from interp_pipeline.downstream.context.microenv import microenv_embedding
from interp_pipeline.downstream.context.neighbors import NeighborConfig
from interp_pipeline.downstream.context.sweep import SweepConfig, run_niche_sweep
from interpretfm.units import native_to_um, um_to_native

METHODS = ("kmeans", "gmm")
SPACES = ("m", "xm")
EXPRESSION_VARIANTS = ("cp10k_scaled", "cp10k_unscaled", "cp10k_svd", "median_scaled")


def radius_to_native(radius_um: float, um_per_unit: float) -> float:
    """A radius in µm in the dataset's coordinate unit, rounded to 1e-6 units."""
    return round(um_to_native(radius_um, um_per_unit), 6)


# --- grid and sweep -----------------------------------------------------------------------------


@dataclass
class NicheGrid:
    """The sweep grid and the score weights (defaults: the manuscript's)."""

    radii_um: List[float]
    k: List[int] = field(default_factory=lambda: [3, 4, 5])
    methods: List[str] = field(default_factory=lambda: list(METHODS))
    spaces: List[str] = field(default_factory=lambda: list(SPACES))
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    min_cluster: int = 150
    n_pcs: int = 50
    kernel: str = "uniform"
    sigma_frac: float = 0.25
    tile_mult: float = 4.0
    n_null: int = 10
    null_seed: int = 0
    w_ari: float = 1.5
    w_purity_z: float = 0.1
    w_sep: float = 1.0
    purity_z_clip: float = 10.0

    def __post_init__(self):
        if not self.radii_um or any(float(r) <= 0 for r in self.radii_um):
            raise ValueError("niches: radii_um must be a non-empty list of positive lengths in µm")
        if not self.k or any(int(k) < 2 for k in self.k):
            raise ValueError("niches: k must be a non-empty list of integers >= 2")
        bad = [m for m in self.methods if m not in METHODS]
        if bad or not self.methods:
            raise ValueError(f"niches: methods must be a non-empty subset of {list(METHODS)}, got {self.methods}")
        bad = [s for s in self.spaces if s not in SPACES]
        if bad or not self.spaces:
            raise ValueError(f"niches: spaces must be a non-empty subset of {list(SPACES)}, got {self.spaces}")
        if not self.seeds:
            raise ValueError("niches: seeds must be a non-empty list")
        if self.kernel not in ("uniform", "gaussian"):
            raise ValueError(f"niches: kernel must be 'uniform' or 'gaussian', got {self.kernel!r}")

    def radii_native(self, um_per_unit: float) -> List[float]:
        return [radius_to_native(float(r), um_per_unit) for r in self.radii_um]

    def pca_components(self, n_features: int) -> int:
        """PCA dimension clamped to the feature count (the manuscript's ``max(2, min(50, n_features))``)."""
        return int(max(2, min(int(self.n_pcs), int(n_features))))

    def sweep_config(self, um_per_unit: float, n_features: int) -> SweepConfig:
        return SweepConfig(radii=self.radii_native(um_per_unit), n_clusters_list=[int(k) for k in self.k],
                           method_list=list(self.methods), space_list=list(self.spaces), kernel=self.kernel,
                           sigma_frac=float(self.sigma_frac), pca_components=self.pca_components(n_features),
                           seeds=[int(s) for s in self.seeds], min_cluster_size=int(self.min_cluster),
                           tile_mult=float(self.tile_mult), n_null=int(self.n_null), null_seed=int(self.null_seed),
                           w_stability=float(self.w_ari), w_purityz=float(self.w_purity_z), w_sep=float(self.w_sep),
                           purityz_clip=float(self.purity_z_clip))

    def to_dict(self, um_per_unit: Optional[float] = None) -> Dict[str, Any]:
        d = {"radii_um": [float(r) for r in self.radii_um], "k": [int(k) for k in self.k], "methods": list(self.methods),
             "spaces": list(self.spaces), "seeds": [int(s) for s in self.seeds], "min_cluster": int(self.min_cluster),
             "n_pcs": int(self.n_pcs), "kernel": self.kernel, "sigma_frac": float(self.sigma_frac),
             "tile_mult": float(self.tile_mult), "n_null": int(self.n_null), "null_seed": int(self.null_seed),
             "score": {"ari": float(self.w_ari), "purity_z": float(self.w_purity_z), "sep": float(self.w_sep),
                       "purity_z_clip": float(self.purity_z_clip)}}
        if um_per_unit is not None:
            d["radii_native"] = self.radii_native(um_per_unit)
            d["um_per_unit"] = float(um_per_unit)
        return d


def config_label(radius_um: float, space: str, method: str, k: int, radius_native: Optional[float] = None) -> str:
    """``r18um/xm/kmeans/k3`` (with the native radius appended as ``r100`` when given)."""
    s = f"r{float(radius_um):g}um/{space}/{method}/k{int(k)}"
    return s if radius_native is None else f"{s} (r{float(radius_native):g} native)"


def _dense(X) -> np.ndarray:
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.ascontiguousarray(X, dtype=np.float32)


def niche_sweep(X, coords: np.ndarray, grid: NicheGrid, um_per_unit: float) -> pd.DataFrame:
    """The sweep of :mod:`interp_pipeline.downstream.context.sweep` over ``grid``, one row per
    configuration that passes ``min_cluster`` (sorted by score), with the radius in both units."""
    X = _dense(X)
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    table = run_niche_sweep(X, coords, grid.sweep_config(um_per_unit, X.shape[1]))
    if not len(table):
        return pd.DataFrame(columns=["radius_um", "radius", "space", "method", "n_clusters", "score", "config"])
    table = table.rename(columns={"radius": "radius_native"})
    table.insert(0, "radius", table["radius_native"].astype(float))
    table.insert(0, "radius_um", [native_to_um(r, um_per_unit) for r in table["radius"]])
    um_of = {rn: float(ru) for ru, rn in zip(grid.radii_um, grid.radii_native(um_per_unit))}
    table["radius_um"] = [um_of.get(float(r), native_to_um(r, um_per_unit)) for r in table["radius"]]
    table = table.drop(columns=["radius_native"])
    table["config"] = [config_label(r.radius_um, r.space, r.method, r.n_clusters) for r in table.itertuples()]
    table["pca_components"] = grid.pca_components(X.shape[1])
    return table.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def select_config(table: pd.DataFrame, k: Optional[int] = None) -> Dict[str, Any]:
    """The highest-scoring row of a sweep table (restricted to ``k`` clusters when given)."""
    t = table if k is None else table[table["n_clusters"].astype(int) == int(k)]
    if not len(t):
        raise ValueError("niche sweep: no configuration passed the minimum cluster size"
                         + (f" with k = {k}" if k is not None else "") + "; lower min_cluster or widen the grid")
    row = t.sort_values("score", ascending=False, kind="stable").iloc[0]
    out = {c: (v.item() if hasattr(v, "item") else v) for c, v in row.items()}
    out["rank"] = int(np.flatnonzero((table.index == row.name))[0]) + 1
    out["n_configs"] = int(len(table))
    return out


# --- one configuration ----------------------------------------------------------------------------


def cluster_niches(X, coords: np.ndarray, radius_native: float, space: str, method: str, k: int, seed: int = 0,
                   n_pcs: int = 50, kernel: str = "uniform", sigma_frac: float = 0.25,
                   ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Niche labels of one configuration (the sweep's first-seed labels when ``seed`` is that seed).

    Returns ``(labels, embedding, neighbours)``: integer labels, the PCA embedding the
    clustering ran in and the radius-neighbour lists (including the cell itself).
    """
    X = _dense(X)
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    m, neigh = microenv_embedding(X, coords, NeighborConfig(radius=float(radius_native), kernel=kernel, sigma_frac=sigma_frac))
    Z = build_cluster_matrix(X, m, space=space)
    n_comp = int(max(1, min(int(n_pcs), Z.shape[0], Z.shape[1])))
    E = pca_embed(Z, n_components=n_comp, seed=0)  # the sweep's fixed PCA seed
    labels = cluster_labels(E, ClusterConfig(space=space, pca_components=n_comp, method=method, n_clusters=int(k),
                                             seed=int(seed))).astype(int)
    return labels, E, neigh


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0, 1, ... by decreasing size (ties: lower original label first)."""
    labels = np.asarray(labels).astype(int)
    vals, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: (-int(counts[i]), int(vals[i])))
    mapping = {int(vals[i]): new for new, i in enumerate(order)}
    return np.array([mapping[int(v)] for v in labels], dtype=int)


def label_agreement(a: Sequence, b: Sequence) -> Dict[str, float]:
    """Adjusted Rand index and normalized mutual information between two labelings."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a = np.asarray(a).astype(str)
    b = np.asarray(b).astype(str)
    return {"ari": float(adjusted_rand_score(a, b)), "nmi": float(normalized_mutual_info_score(a, b))}


def niche_validation(labels: np.ndarray, neigh: List[np.ndarray], coords: np.ndarray, embedding: Optional[np.ndarray],
                     tile_size: float, n_null: int = 10, null_seed: int = 0, ref_labels: Optional[Sequence] = None,
                     purity_z_clip: float = 10.0) -> Dict[str, Any]:
    """Purity, tile-shuffle null and z of one labeling, its separability, sizes and (optionally)
    the agreement with reference labels."""
    labels = np.asarray(labels).astype(int)
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    purity = float(mean_neighbor_purity(labels, neigh))
    null_mu, null_sd = tile_shuffle_null_purity(labels, coords, neigh, tile_size=float(tile_size), n_null=int(n_null),
                                                seed=int(null_seed))
    z = (purity - null_mu) / null_sd if null_sd > 0 else 0.0
    sizes = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    out = {"n_cells": int(len(labels)), "n_niches": int(len(sizes)), "sizes": sizes, "min_cluster_size": int(min_cluster_size(labels)),
           "purity": purity, "purity_null_mean": float(null_mu), "purity_null_std": float(null_sd), "purity_z": float(z),
           "purity_z_clipped": float(np.clip(z, -purity_z_clip, purity_z_clip)), "tile_size": float(tile_size), "n_null": int(n_null)}
    if embedding is not None:
        out["separability"] = float(separability_cosine(np.asarray(embedding, dtype=np.float32), labels))
    if ref_labels is not None:
        out.update(label_agreement(ref_labels, labels))
    return out


# --- baselines ------------------------------------------------------------------------------------


def log_normalize(counts, target_sum: float = 1e4):
    """``log1p`` of counts scaled to ``target_sum`` per cell (CSR, float32). ``target_sum=None``
    scales to the median library size of the cells with counts."""
    from scipy import sparse

    X = sparse.csr_matrix(counts).astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    target = float(np.median(lib[lib > 0])) if target_sum is None else float(target_sum)
    Xn = sparse.diags(target / np.maximum(lib, 1.0)) @ X
    Xn = sparse.csr_matrix(Xn)
    Xn.data = np.log1p(Xn.data)
    return Xn


def expression_svd(counts, n_components: int = 50, seed: int = 0) -> np.ndarray:
    """The expression-only features: truncated SVD of log-normalized counts (uncentred)."""
    from sklearn.decomposition import TruncatedSVD

    Xn = log_normalize(counts)
    n_comp = int(min(n_components, Xn.shape[1] - 1, Xn.shape[0] - 1))
    if n_comp < 2:
        raise ValueError("expression baseline: fewer than 3 genes or cells")
    return TruncatedSVD(n_comp, random_state=int(seed)).fit_transform(Xn).astype(np.float32)


def expression_variant(counts, variant: str, n_components: int = 50, seed: int = 0) -> np.ndarray:
    """One of the four preprocessing variants of the expression features (``cp10k_scaled``:
    CP10k, log1p, genes standardized and clipped at ±10, PCA; ``cp10k_unscaled``: centred PCA
    without scaling; ``cp10k_svd``: uncentred truncated SVD, = :func:`expression_svd`;
    ``median_scaled``: library size scaled to the median, log1p, scaled and clipped, PCA)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if variant not in EXPRESSION_VARIANTS:
        raise ValueError(f"expression variant must be one of {list(EXPRESSION_VARIANTS)}, got {variant!r}")
    if variant == "cp10k_svd":
        return expression_svd(counts, n_components, seed)
    D = log_normalize(counts, None if variant == "median_scaled" else 1e4).toarray().astype(np.float32)
    n_comp = int(min(n_components, D.shape[1], D.shape[0]))
    if variant != "cp10k_unscaled":
        D = np.clip(StandardScaler().fit_transform(D), -10, 10).astype(np.float32)
    return PCA(n_comp, random_state=int(seed)).fit_transform(D).astype(np.float32)


def label_composition(labels: Sequence, coords: np.ndarray, radius_native: float, k: int, n_pcs: int = 15,
                      n_init: int = 20, seed: int = 0) -> Tuple[np.ndarray, pd.DataFrame]:
    """The label-composition baseline: the fraction of each label among a cell's neighbours
    within the radius (the cell itself excluded; cells without neighbours get a zero row), PCA
    to ``n_pcs`` components, k-means with ``n_init`` initialisations. Returns the labels and
    the composition table."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    cats = pd.Categorical(np.asarray(labels).astype(str))
    n = len(cats)
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    onehot = sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), cats.codes)), shape=(n, len(cats.categories)))
    pairs = cKDTree(coords).query_pairs(float(radius_native), output_type="ndarray")
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    W = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    comp = (W @ onehot).toarray().astype(np.float32)
    s = comp.sum(1)
    s[s == 0] = 1.0
    comp = comp / s[:, None]
    n_comp = int(max(1, min(int(n_pcs), comp.shape[1], n - 1)))
    E = PCA(n_components=n_comp, random_state=int(seed)).fit_transform(comp)
    lab = KMeans(n_clusters=int(k), random_state=int(seed), n_init=int(n_init)).fit_predict(E).astype(int)
    return lab, pd.DataFrame(comp, columns=[str(c) for c in cats.categories])


def seed_stability(table: pd.DataFrame, ks: Sequence[int] = (4, 5)) -> Dict[str, Any]:
    """Median seed-to-seed ARI (``ari_mean``) over the configurations with ``k`` in ``ks`` of a
    sweep table, with the other k values as a control."""
    t = table[table["n_clusters"].astype(int).isin([int(k) for k in ks])]
    rest = table[~table["n_clusters"].astype(int).isin([int(k) for k in ks])]
    return {"k": [int(k) for k in ks], "n_configs": int(len(t)),
            "median_seed_ari": float(t["ari_mean"].median()) if len(t) else None,
            "n_configs_other_k": int(len(rest)), "median_seed_ari_other_k": float(rest["ari_mean"].median()) if len(rest) else None}


__all__ = ["EXPRESSION_VARIANTS", "METHODS", "SPACES", "NicheGrid", "cluster_niches", "config_label", "expression_svd",
           "expression_variant", "label_agreement", "label_composition", "log_normalize", "niche_sweep", "niche_validation",
           "radius_to_native", "relabel_by_size", "seed_stability", "select_config"]
