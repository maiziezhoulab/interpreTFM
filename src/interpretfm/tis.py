"""Transcriptomic interpretability score (TIS) of one layer, with its controls.

TIS asks, per activation dimension, whether a cell's measured expression profile predicts
whether the cell sits above the median of that dimension. For each of ``trials`` trials a
query cell is drawn, ``k_ref`` reference cells are drawn from the upper- and lower-quantile
pools of the dimension (the query excluded), and the query is assigned to the pool whose
cells are more similar in expression (cosine similarity of library-size normalized,
log-transformed, L2-normalized profiles). TIS is the accuracy over trials.

Controls: each dimension permuted independently across cells (``shuffle``; the layer's
*TIS gain* is the mean TIS minus the mean shuffled TIS) and a dimension-matched PCA
(``pca``): the activation matrix rotated to its principal components (``pca_of:
activations``, the default and what the manuscript reports) or a PCA of the expression
matrix with as many components (``pca_of: expression``). The computations are those of
:mod:`interp_pipeline.tis.core`; this module adds the cell-level matrix from the store
(:func:`interpretfm.stores.cell_matrix`) and the per-layer result record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from interp_pipeline.tis.core import (
    TISConfig,
    compute_tis_mis_quantile_streamed,
    pca_activations,
    shuffle_activations,
)
from interp_pipeline.tis.io import build_judge_matrix

CONTROLS = ("shuffle", "pca")


@dataclass
class TISParams:
    """Parameters of the TIS stage (``stages.tis``)."""

    trials: int = 800
    k_ref: int = 32
    quantiles: Sequence[float] = (0.25, 0.75)
    judge: str = "log1p_cp10k"
    controls: Sequence[str] = ("shuffle", "pca")
    pca_of: str = "activations"
    seed: int = 42
    subsample_eval: Optional[int] = None
    max_shards: Optional[int] = None
    keep_cell_matrix: bool = True

    def tis_config(self) -> TISConfig:
        return TISConfig(K=int(self.k_ref), n_trials=int(self.trials), seed=int(self.seed),
                         q_low=float(self.quantiles[0]), q_high=float(self.quantiles[1]),
                         subsample_eval=self.subsample_eval)


@dataclass
class TISResult:
    """Per-layer TIS: per-dimension scores of the model and its controls, and their summaries."""

    layer: str
    tis: np.ndarray
    shuffled: Optional[np.ndarray] = None
    pca: Optional[np.ndarray] = None
    n_cells: int = 0
    hidden_dim: int = 0
    extraction: Dict[str, Any] = field(default_factory=dict)

    @property
    def gain(self) -> Optional[float]:
        if self.shuffled is None:
            return None
        return float(np.nanmean(self.tis) - np.nanmean(self.shuffled))

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"layer": self.layer, "n_cells": int(self.n_cells), "hidden_dim": int(self.hidden_dim),
                               "model": summarize(self.tis)}
        if self.shuffled is not None:
            out["shuffled"] = summarize(self.shuffled)
            out["gain"] = self.gain
        if self.pca is not None:
            out["pca"] = summarize(self.pca)
        out["extraction"] = dict(self.extraction)
        return out

    def per_dim_frame(self):
        import pandas as pd

        d = {"dim": np.arange(len(self.tis)), "tis": self.tis}
        if self.shuffled is not None:
            d["tis_shuffled"] = self.shuffled
        if self.pca is not None:
            d["tis_pca"] = self.pca
        return pd.DataFrame(d)


def summarize(arr: np.ndarray) -> Dict[str, Any]:
    """Mean, median, 90th/95th percentile and the number of scored dimensions."""
    arr = np.asarray(arr, dtype=np.float32)
    good = np.isfinite(arr)
    if not good.any():
        return {"mean": None, "median": None, "p90": None, "p95": None, "n_valid": 0, "n_dims": int(arr.size)}
    return {"mean": float(np.nanmean(arr)), "median": float(np.nanmedian(arr)),
            "p90": float(np.percentile(arr[good], 90)), "p95": float(np.percentile(arr[good], 95)),
            "n_valid": int(good.sum()), "n_dims": int(arr.size)}


def params_from_mapping(params: Dict[str, Any], default_seed: int = 42) -> TISParams:
    """Validate ``stages.tis`` and return :class:`TISParams`."""
    from interpretfm.params import get_list, get_param

    q = get_list(params, "quantiles", (0.25, 0.75), "tis")
    if len(q) != 2 or not (0 < float(q[0]) < float(q[1]) < 1):
        raise ValueError(f"stages.tis.quantiles: expected [q_low, q_high] with 0 < q_low < q_high < 1, got {q}")
    p = TISParams(
        trials=get_param(params, "trials", 800, "tis", int),
        k_ref=get_param(params, "k_ref", 32, "tis", int),
        quantiles=(float(q[0]), float(q[1])),
        judge=get_param(params, "judge", "log1p_cp10k", "tis", str, choices=("log1p_cp10k", "raw")),
        controls=get_list(params, "controls", list(CONTROLS), "tis", choices=CONTROLS),
        pca_of=get_param(params, "pca_of", "activations", "tis", str, choices=("expression", "activations")),
        seed=get_param(params, "seed", default_seed, "tis", int),
        subsample_eval=get_param(params, "subsample_eval", None, "tis", int),
        max_shards=get_param(params, "max_shards", None, "tis", int),
        keep_cell_matrix=get_param(params, "keep_cell_matrix", True, "tis", bool),
    )
    if p.trials <= 0 or p.k_ref <= 0:
        raise ValueError("stages.tis.trials and stages.tis.k_ref must be > 0")
    return p


def tis_for_matrix(A: np.ndarray, adata, params: TISParams, layer: str = "",
                   extraction: Optional[Dict[str, Any]] = None) -> TISResult:
    """TIS of an aligned cell x dimension matrix ``A`` against the expression in ``adata``
    (same cells, same order), with the requested controls."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    cfg = params.tis_config()
    J = build_judge_matrix(adata, mode=params.judge)
    res = TISResult(layer=layer, tis=compute_tis_mis_quantile_streamed(A, J, cfg.q_low, cfg.q_high, cfg),
                    n_cells=int(A.shape[0]), hidden_dim=int(A.shape[1]), extraction=dict(extraction or {}))
    if "shuffle" in params.controls:
        res.shuffled = compute_tis_mis_quantile_streamed(shuffle_activations(A, seed=cfg.seed), J, cfg.q_low, cfg.q_high, cfg)
    if "pca" in params.controls:
        if params.pca_of == "activations":
            P = pca_activations(A, seed=cfg.seed)
        else:
            Jd = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
            n_comp = int(min(A.shape[1], Jd.shape[1], Jd.shape[0]))
            P = _pca_first(np.asarray(Jd, dtype=np.float32), n_comp, cfg.seed)
        res.pca = compute_tis_mis_quantile_streamed(P, J, cfg.q_low, cfg.q_high, cfg)
    return res


def _pca_first(X: np.ndarray, n_comp: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=int(n_comp), random_state=int(seed)).fit_transform(X).astype(np.float32)


def tis_for_layer(store_root, layer: str, adata, params: TISParams, pooling: str, token: Optional[str],
                  special_tokens: Sequence[str] = ()) -> TISResult:
    """TIS of one layer read from the activation store: the cell-level matrix is the
    classification-token row (``pooling='token'``) or the mean over gene tokens
    (``pooling='mean'``), aligned to ``adata.obs_names``; cells without a row are dropped."""
    from interpretfm.stores import cell_matrix

    A, has_row, extraction = cell_matrix(store_root, layer, [str(o) for o in adata.obs_names], pooling,
                                         token=token, special_tokens=special_tokens, max_shards=params.max_shards)
    sub = adata if has_row.all() else adata[has_row]
    res = tis_for_matrix(A[has_row], sub, params, layer=layer, extraction=extraction)
    res.extraction["has_row"] = has_row
    return res


def layer_table(results: Sequence[TISResult]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        s = r.summary()
        row = {"layer": r.layer, "n_cells": r.n_cells, "hidden_dim": r.hidden_dim, "tis_mean": s["model"]["mean"],
               "tis_median": s["model"]["median"], "tis_n_valid": s["model"]["n_valid"]}
        if "shuffled" in s:
            row.update({"shuffled_mean": s["shuffled"]["mean"], "tis_gain": s["gain"]})
        if "pca" in s:
            row["pca_mean"] = s["pca"]["mean"]
        rows.append(row)
    return rows


def params_dict(p: TISParams) -> Dict[str, Any]:
    d = asdict(p)
    d["quantiles"] = list(p.quantiles)
    d["controls"] = list(p.controls)
    return d
