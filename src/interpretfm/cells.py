"""Cell-level features: retention of concept-labelled SAE features, projection of gene-token
features to cells, and population specificity (GSI excess, MHR@k) against raw dimensions and
single genes.

**Retention** (:func:`select_features`): for each concept with ``size_range`` panel genes, the
unit with the highest validation F1 at threshold ``tau`` in the row split (ties: lowest
index) is kept when that F1 is at least ``min_valid_f1``; features are then de-duplicated,
each labelled by its best concept (highest validation F1, then fewer panel genes, then
concept id). Selection uses validation rows only; the test F1 of the kept pairs is reported.

**Projection** (:func:`accumulate_cell_profiles`): a cell's profile is the mean activation
over its gene-token rows (special tokens excluded); cells recorded more than once in a store
are reduced to their first copy.

**Population specificity** (:func:`gsi_excess`, :func:`mhr_split_half`, :func:`specificity`),
following scPISCES: the Gini specificity index of a feature is the Gini coefficient of its
mean activation across populations with at least ``min_pop`` cells; GSI excess subtracts the
feature's mean GSI over label shuffles and is averaged over active features. MHR@k: cells are
split in half within each population; in half 1 the markers of a population are the ``k``
genes with the largest mean expression difference against all other cells and the
``n_leading`` features most correlated with population membership are selected; in half 2,
MHR is the share of the ``n_assoc`` genes most correlated with each selected feature that are
markers, averaged over populations. SAE features, raw dimensions (split into their positive and
negative parts) and single genes (log-normalized expression) are scored on the same cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from interp_pipeline.get_annotation.f1_alignment import first_copy_row_mask
from interpretfm.sae import Encoder
from interpretfm.stores import list_shards, load_shard, resolve_example_rows, select_shards

# --- retention ----------------------------------------------------------------------------------


@dataclass
class RetentionRule:
    tau: float = 0.15
    min_valid_f1: float = 0.40
    size_range: Tuple[int, int] = (5, 100)
    min_features: int = 10
    unit: str = "row"

    def to_dict(self) -> Dict[str, Any]:
        return {"tau": self.tau, "min_valid_f1": self.min_valid_f1, "concept_size": list(self.size_range),
                "min_features": self.min_features, "unit": self.unit}


def _at_threshold(table, tau: float):
    t = table[np.isclose(table["threshold"].astype(float), tau)][["concept", "feature", "f1"]].copy()
    t["concept"] = t["concept"].astype(str)
    t["feature"] = t["feature"].astype(int)
    return t


def select_features(valid, test, concept_sizes, rule: RetentionRule, term_names: Optional[Dict[str, str]] = None):
    """Apply the retention rule to a layer's validation and test tables.

    ``valid``/``test``: the ``concept_eval`` tables (concept, feature, threshold, f1, ...);
    ``concept_sizes``: ``Series`` concept id -> number of panel genes. Returns
    ``(best, kept, features, info)``: the best unit per eligible concept, the concepts passing
    ``min_valid_f1``, the de-duplicated feature table (feature, concept, valid_f1, test_f1,
    n_panel_genes, n_concepts_sharing_feature, var_name[, term_name]) and a summary.
    """
    import pandas as pd

    sizes = pd.Series(concept_sizes).astype(int)
    sizes.index = sizes.index.astype(str)
    eligible = set(sizes.index[(sizes >= rule.size_range[0]) & (sizes <= rule.size_range[1])])
    v = _at_threshold(valid, rule.tau)
    n_concepts_valid = int(v["concept"].nunique())
    v = v[v["concept"].isin(eligible)]
    best = (v.sort_values(["concept", "f1", "feature"], ascending=[True, False, True])
             .drop_duplicates("concept").rename(columns={"f1": "valid_f1"}).reset_index(drop=True))
    best["n_panel_genes"] = sizes.reindex(best["concept"]).to_numpy()
    t = _at_threshold(test, rule.tau).rename(columns={"f1": "test_f1"})
    best = best.merge(t, on=["concept", "feature"], how="left")
    kept = best[best["valid_f1"] >= rule.min_valid_f1].copy()
    feats = (kept.sort_values(["feature", "valid_f1", "n_panel_genes", "concept"], ascending=[True, False, True, True])
                 .drop_duplicates("feature").copy())
    feats["n_concepts_sharing_feature"] = feats["feature"].map(kept.groupby("feature").size()).astype(int)
    feats = feats.sort_values(["valid_f1", "feature"], ascending=[False, True]).reset_index(drop=True)
    names = {str(k): str(v).strip() for k, v in (term_names or {}).items() if str(v).strip()}
    seen: set = set()
    var_names = []
    for r in feats.itertuples(index=False):
        pretty = names.get(r.concept, "")
        pretty = "" if pretty == r.concept else pretty
        base = (f"{pretty}|{r.concept}|feat{r.feature}" if pretty else f"{r.concept}|feat{r.feature}").replace("/", "_")
        nm, j = base, 2
        while nm in seen:
            nm, j = f"{base}__{j}", j + 1
        seen.add(nm)
        var_names.append(nm)
    feats["var_name"] = var_names
    if names:
        feats["term_name"] = feats["concept"].map(names)
    info = {"n_concepts_in_valid_table": n_concepts_valid, "n_eligible_concepts": int(len(eligible)),
            "n_eligible_with_valid_rows": int(len(best)), "n_concepts_kept": int(len(kept)),
            "n_features": int(len(feats)), "enough_features": bool(len(feats) >= rule.min_features),
            "valid_f1_mean": float(feats["valid_f1"].mean()) if len(feats) else None,
            "test_f1_mean": float(feats["test_f1"].mean()) if len(feats) and feats["test_f1"].notna().any() else None,
            "test_f1_median": float(feats["test_f1"].median()) if len(feats) and feats["test_f1"].notna().any() else None,
            "test_f1_all_kept_concepts_mean": float(kept["test_f1"].mean()) if len(kept) and kept["test_f1"].notna().any() else None,
            "n_kept_pairs_missing_test_row": int(kept["test_f1"].isna().sum()) if len(kept) else 0}
    return best, kept, feats, info


# --- cell profiles ------------------------------------------------------------------------------


@dataclass
class CellProfiles:
    """Mean activation per cell over its gene-token rows."""

    cell_index: np.ndarray  #: row index into the dataset's observations
    sae: Optional[np.ndarray]  #: cells x features (the selected ones, or all)
    raw: Optional[np.ndarray]  #: cells x 2·hidden: relu(x) then relu(-x)
    counts: np.ndarray  #: gene-token rows per cell
    stats: Dict[str, Any] = field(default_factory=dict)


def pick_shards_for_cells(shards: Sequence, target: Optional[int]) -> List:
    """Evenly spaced shards holding about ``target`` cells (all shards when ``target`` is ``None``
    or large). The cells per shard are read from the first shard's index."""
    shards = list(shards)
    if target is None or not shards:
        return shards
    _X, idx = load_shard(shards[0])
    per_shard = len(set(str(e) for e in idx.get("example_ids", [])))
    n_pick = min(len(shards), int(math.ceil(target / max(per_shard - 2, 1))))
    return select_shards(shards, n_pick, evenly_spaced=True)


def accumulate_cell_profiles(store_root, layer: str, encoder: Optional[Encoder], obs_names: Sequence[str],
                             special_tokens: Sequence[str] = (), shards: Optional[Sequence] = None,
                             feature_idx: Optional[Sequence[int]] = None, raw_halves: bool = False,
                             dedup_cells: bool = True, drop_edge_cells: bool = False, batch_size: int = 8192,
                             device: Union[str, torch.device] = "cpu") -> CellProfiles:
    """One pass over ``shards`` (default: all shards of the layer) accumulating per-cell means.

    ``encoder`` gives the SAE latents (``None``: no SAE profile); ``feature_idx`` restricts the
    latents kept; ``raw_halves`` also accumulates ``relu(x)`` and ``relu(-x)`` of the hidden
    state. ``drop_edge_cells`` excludes the first and last cell of every shard (cells that may
    be split across shards when only a subset of shards is read). Rows of special tokens are
    excluded; with ``dedup_cells`` only the first copy of a cell counts.
    """
    special = {str(t) for t in special_tokens}
    device = torch.device(device)
    n_obs = len(obs_names)
    shards = list(shards) if shards is not None else list_shards(store_root, layer)
    fidx = None if feature_idx is None else torch.as_tensor(np.asarray(feature_idx, dtype=np.int64), device=device)
    n_feat = 0 if encoder is None else (encoder.n_latents if fidx is None else int(fidx.numel()))
    sum_z = torch.zeros((n_obs, n_feat), dtype=torch.float64) if encoder is not None else None
    sum_r: Optional[torch.Tensor] = None
    counts = np.zeros(n_obs, dtype=np.int64)
    seen: set = set()
    stats = {"n_shards": len(shards), "n_rows_seen": 0, "n_rows_used": 0, "n_rows_special": 0, "n_rows_unmatched": 0,
             "n_rows_dropped_later_copy": 0, "n_rows_dropped_edge": 0, "dedup_cells": dedup_cells,
             "drop_edge_cells": drop_edge_cells}
    for shard in shards:
        X, idx = load_shard(shard)
        ex = idx.get("example_ids")
        tok = idx.get("token_ids")
        if ex is None or len(ex) != X.shape[0]:
            continue
        stats["n_rows_seen"] += int(X.shape[0])
        keep = np.ones(X.shape[0], dtype=bool)
        if dedup_cells:
            keep = first_copy_row_mask(ex, seen)
            stats["n_rows_dropped_later_copy"] += int((~keep).sum())
        if drop_edge_cells and len(ex):
            edge = {str(ex[0]), str(ex[-1])}
            is_edge = np.array([str(e) in edge for e in ex], dtype=bool)
            stats["n_rows_dropped_edge"] += int((keep & is_edge).sum())
            keep &= ~is_edge
        if tok is not None and len(tok) == X.shape[0]:
            is_special = np.array([str(t) in special for t in tok], dtype=bool)
            stats["n_rows_special"] += int((keep & is_special).sum())
            keep &= ~is_special
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            continue
        cells = resolve_example_rows([ex[r] for r in rows], obs_names)
        ok = cells >= 0
        stats["n_rows_unmatched"] += int((~ok).sum())
        rows, cells = rows[ok], cells[ok]
        if rows.size == 0:
            continue
        if sum_r is None and raw_halves:
            sum_r = torch.zeros((n_obs, 2 * int(X.shape[1])), dtype=torch.float64)
        for s in range(0, rows.size, batch_size):
            br = torch.as_tensor(rows[s:s + batch_size])
            bc = torch.as_tensor(cells[s:s + batch_size])
            x = X[br].to(device)
            if sum_z is not None:
                z = encoder.encode(x)
                if fidx is not None:
                    z = z[:, fidx]
                sum_z.index_add_(0, bc, z.cpu().double())
            if sum_r is not None:
                sum_r.index_add_(0, bc, torch.cat([torch.relu(x), torch.relu(-x)], 1).cpu().double())
        np.add.at(counts, cells, 1)
        stats["n_rows_used"] += int(rows.size)
    has = counts > 0
    cell_index = np.flatnonzero(has)
    denom = torch.as_tensor(counts[has], dtype=torch.float64)[:, None]
    sae = None if sum_z is None else (sum_z[torch.as_tensor(cell_index)] / denom).float().numpy()
    raw = None if sum_r is None else (sum_r[torch.as_tensor(cell_index)] / denom).float().numpy()
    stats.update({"n_obs": int(n_obs), "n_has_row": int(has.sum()), "n_cells_seen": len(seen),
                  "rows_per_cell_median": float(np.median(counts[has])) if has.any() else None})
    return CellProfiles(cell_index=cell_index, sae=sae, raw=raw, counts=counts[has], stats=stats)


def log_normalized_expression(X, target_sum: float = 1e4) -> np.ndarray:
    """``log1p`` of counts scaled to ``target_sum`` per cell (dense float32)."""
    from scipy import sparse

    M = X.toarray() if sparse.issparse(X) else np.asarray(X)
    M = np.asarray(M, dtype=np.float32)
    tot = M.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(M / tot * float(target_sum)).astype(np.float32)


# --- GSI ----------------------------------------------------------------------------------------


def gini_columns(m: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of every column (populations x features); NaN where a column sums to 0."""
    s, _ = torch.sort(m, dim=0)
    p = s.shape[0]
    tot = s.sum(0)
    i = torch.arange(1, p + 1, device=m.device, dtype=m.dtype)[:, None]
    g = 2 * (i * s).sum(0) / (p * tot) - (p + 1) / p
    g[tot <= 0] = float("nan")
    return g


def population_means(feat: torch.Tensor, lab: torch.Tensor, n_pop: int) -> torch.Tensor:
    m = torch.zeros(n_pop, feat.shape[1], device=feat.device, dtype=feat.dtype)
    m.index_add_(0, lab, feat)
    return m / torch.bincount(lab, minlength=n_pop)[:, None].to(feat.dtype)


def label_shuffles(n_cells: int, n_shuffles: int, rng: np.random.Generator) -> List[np.ndarray]:
    """``n_shuffles`` permutations of the cell order (shared by every representation)."""
    return [rng.permutation(n_cells) for _ in range(n_shuffles)]


def gsi_excess(feat: torch.Tensor, lab: torch.Tensor, n_pop: int, shuffles: Sequence[np.ndarray]) -> Dict[str, Any]:
    """GSI of every feature, its mean over the label ``shuffles`` (permutations of the cells), and the excess.

    ``feat``: cells x features (non-negative); ``lab``: population code per cell. Returns the
    per-feature arrays (``gini``, ``shuffled``, ``excess``) and the summary over features
    with a defined Gini (``excess_mean``, ``excess_median``, ``excess_p90``, ``gini_mean``,
    ``shuffled_mean``, ``n_features``).
    """
    n_shuffles = len(shuffles)
    g = gini_columns(population_means(feat, lab, n_pop))
    shuffled = [lab[torch.as_tensor(np.asarray(s), device=lab.device)] for s in shuffles]
    sh = torch.stack([gini_columns(population_means(feat, s, n_pop)) for s in shuffled]).nanmean(0) if shuffled else torch.zeros_like(g)
    excess = g - sh
    ok = ~torch.isnan(excess)
    e = excess[ok]
    summary = {"n_features": int(ok.sum()), "n_shuffles": int(n_shuffles),
               "excess_mean": float(e.mean()) if ok.any() else None,
               "excess_median": float(torch.quantile(e, 0.5)) if ok.any() else None,
               "excess_p90": float(torch.quantile(e, 0.9)) if ok.any() else None,
               "gini_mean": float(g[ok].mean()) if ok.any() else None,
               "shuffled_mean": float(sh[ok].mean()) if ok.any() else None}
    return {"gini": g.cpu().numpy(), "shuffled": sh.cpu().numpy(), "excess": excess.cpu().numpy(), "summary": summary}


# --- MHR ----------------------------------------------------------------------------------------


def zscore_columns(m: torch.Tensor) -> torch.Tensor:
    sd = m.std(0, keepdim=True)
    return (m - m.mean(0, keepdim=True)) / torch.where(sd > 0, sd, torch.full_like(sd, float("nan")))


def split_half(lab: np.ndarray, n_pop: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified 50/50 split: ``True`` marks half 1 (half the cells of every population)."""
    half = np.zeros(len(lab), dtype=bool)
    for p in range(n_pop):
        w = np.flatnonzero(lab == p)
        half[rng.choice(w, size=len(w) // 2, replace=False)] = True
    return half


def population_markers(expr_a: torch.Tensor, lab_a: torch.Tensor, n_pop: int, k: int) -> List[torch.Tensor]:
    """Per population, the ``k`` genes with the largest mean-expression difference against all other cells (half 1)."""
    if k > expr_a.shape[1]:
        raise ValueError(f"mhr_k={k} exceeds the number of genes ({expr_a.shape[1]})")
    out = []
    for p in range(n_pop):
        inp = lab_a == p
        out.append(torch.topk(expr_a[inp].mean(0) - expr_a[~inp].mean(0), k).indices)
    return out


def leading_features(feat_a: torch.Tensor, lab_a: torch.Tensor, n_pop: int, n_leading: int) -> torch.Tensor:
    """Per population, the ``n_leading`` features with the highest correlation with membership (half 1): n_leading x P."""
    onehot = torch.nn.functional.one_hot(lab_a, n_pop).to(feat_a.dtype)
    r = zscore_columns(feat_a).T @ zscore_columns(onehot) / (feat_a.shape[0] - 1)
    return torch.topk(torch.nan_to_num(r, nan=-2.0), min(n_leading, feat_a.shape[1]), dim=0).indices


def mhr(feat_b: torch.Tensor, expr_b_z: torch.Tensor, lead: torch.Tensor, markers: Sequence[torch.Tensor],
        n_assoc: int) -> np.ndarray:
    """MHR per population on half 2: the share of each leading feature's ``n_assoc`` most correlated genes
    that are markers, averaged over the population's leading features."""
    n_pop = lead.shape[1]
    out = np.full(n_pop, np.nan)
    n_assoc = min(n_assoc, expr_b_z.shape[1])
    for p in range(n_pop):
        f = feat_b[:, lead[:, p]]
        r = zscore_columns(f).T @ expr_b_z / (feat_b.shape[0] - 1)
        r = r[~torch.isnan(r).all(1)]  # a feature constant on half 2 has no gene ranking
        if r.shape[0] == 0:
            continue
        top = torch.topk(torch.nan_to_num(r, nan=-2.0), n_assoc, dim=1).indices
        out[p] = float(torch.isin(top, markers[p]).float().mean())
    return out


@dataclass
class SpecificityParams:
    min_pop: int = 30
    n_shuffles: int = 5
    mhr_k: int = 100
    n_leading: int = 3
    n_assoc: int = 50
    n_null: int = 50
    seed: int = 0


def mhr_split_half(feat: torch.Tensor, expr: torch.Tensor, lab: torch.Tensor, n_pop: int, half: np.ndarray,
                   params: SpecificityParams, rng: np.random.Generator) -> Dict[str, Any]:
    """MHR@k of one representation with a given split (half 1 = ``half``), plus the random-feature null."""
    A = torch.as_tensor(half)
    B = torch.as_tensor(~half)
    markers = population_markers(expr[A], lab[A], n_pop, params.mhr_k)
    expr_b_z = torch.nan_to_num(zscore_columns(expr[B]), nan=0.0)
    lead = leading_features(feat[A], lab[A], n_pop, params.n_leading)
    per_pop = mhr(feat[B], expr_b_z, lead, markers, params.n_assoc)
    null = []
    for _ in range(params.n_null):
        ridx = torch.as_tensor(rng.integers(0, feat.shape[1], size=(lead.shape[0], n_pop)))
        null.append(float(np.nanmean(mhr(feat[B], expr_b_z, ridx, markers, params.n_assoc))))
    return {"mhr": float(np.nanmean(per_pop)), "per_population": per_pop, "leading": lead.cpu().numpy(),
            "null_mean": float(np.mean(null)) if null else None, "null_sd": float(np.std(null)) if len(null) > 1 else None,
            "chance": params.mhr_k / int(expr.shape[1])}


def specificity(reps: Dict[str, np.ndarray], expr: np.ndarray, labels: Sequence[str], params: SpecificityParams,
                ) -> Dict[str, Any]:
    """GSI excess and MHR@k of every representation on the same cells.

    ``reps``: name -> cells x features (non-negative; inactive features are dropped);
    ``expr``: cells x genes log-normalized expression; ``labels``: one label per cell.
    Populations with fewer than ``min_pop`` cells are excluded. The same label shuffles and the
    same half split (seeded) serve every representation. Returns a summary per representation,
    the per-feature GSI arrays and the per-population MHR table.
    """
    import pandas as pd

    labels = np.asarray([str(x) for x in labels], dtype=object)
    vc = pd.Series(labels).value_counts()
    pops = sorted(vc[vc >= params.min_pop].index)
    keep = np.isin(labels, pops)
    n_pop = len(pops)
    result: Dict[str, Any] = {"n_cells_in": int(len(labels)), "n_cells": int(keep.sum()), "n_populations": n_pop,
                              "populations": {p: int(vc[p]) for p in pops}, "n_genes": int(expr.shape[1]),
                              "params": params.__dict__.copy(), "representations": {}, "per_feature": {}, "per_population": None}
    if n_pop < 2:
        result["skipped"] = f"fewer than two populations with >= {params.min_pop} cells"
        return result
    lab_np = pd.Categorical(labels[keep], categories=pops).codes.astype(np.int64)
    lab = torch.as_tensor(lab_np)
    E = torch.as_tensor(np.asarray(expr, dtype=np.float32)[keep])
    rng = np.random.default_rng(params.seed)
    shuffles = label_shuffles(len(lab_np), params.n_shuffles, rng)  # the same shuffles and split for every representation
    half = split_half(lab_np, n_pop, rng)
    per_pop_rows = []
    for name, M in reps.items():
        F = torch.as_tensor(np.asarray(M, dtype=np.float32)[keep])
        alive = F.max(0).values > 0
        F = F[:, alive]
        entry: Dict[str, Any] = {"n_features": int(M.shape[1]), "n_active": int(alive.sum())}
        if F.shape[1] == 0:
            entry["skipped"] = "no active feature"
            result["representations"][name] = entry
            continue
        g = gsi_excess(F, lab, n_pop, shuffles)
        entry["gsi"] = g["summary"]
        full = {k: np.full(M.shape[1], np.nan, dtype=np.float32) for k in ("gini", "shuffled", "excess")}
        idx = np.flatnonzero(alive.numpy())
        for k in full:
            full[k][idx] = g[k]
        result["per_feature"][name] = full
        m = mhr_split_half(F, E, lab, n_pop, half, params, np.random.default_rng(params.seed + 1))
        entry["mhr"] = {k: m[k] for k in ("mhr", "null_mean", "null_sd", "chance")}
        entry["mhr"]["k"] = params.mhr_k
        lead_full = idx[m["leading"]]
        for p in range(n_pop):
            per_pop_rows.append({"representation": name, "population": pops[p], "n_cells": int((lab_np == p).sum()),
                                 "mhr": m["per_population"][p], "leading_features": ";".join(str(int(f)) for f in lead_full[:, p])})
        result["representations"][name] = entry
    result["per_population"] = pd.DataFrame(per_pop_rows)
    return result


def specificity_params(params: Dict[str, Any], default_seed: int = 0, stage: str = "cells") -> SpecificityParams:
    from interpretfm.params import get_param

    p = SpecificityParams(
        min_pop=get_param(params, "min_pop", 30, stage + ".specificity", int),
        n_shuffles=get_param(params, "n_shuffles", 5, stage + ".specificity", int),
        mhr_k=get_param(params, "mhr_k", 100, stage + ".specificity", int),
        n_leading=get_param(params, "n_leading", 3, stage + ".specificity", int),
        n_assoc=get_param(params, "n_assoc", 50, stage + ".specificity", int),
        n_null=get_param(params, "n_null", 50, stage + ".specificity", int),
        seed=get_param(params, "seed", default_seed, stage + ".specificity", int),
    )
    if p.min_pop < 2 or p.mhr_k <= 0 or p.n_leading <= 0 or p.n_assoc <= 0 or p.n_shuffles < 0 or p.n_null < 0:
        raise ValueError(f"stages.{stage}.specificity: min_pop >= 2, mhr_k/n_leading/n_assoc > 0, n_shuffles/n_null >= 0")
    return p


__all__ = ["CellProfiles", "RetentionRule", "SpecificityParams", "accumulate_cell_profiles", "gini_columns", "gsi_excess",
           "label_shuffles", "leading_features", "log_normalized_expression", "mhr", "mhr_split_half", "pick_shards_for_cells",
           "population_markers", "population_means", "select_features", "specificity", "specificity_params",
           "split_half", "zscore_columns"]
