"""Concept labels of niches: which concept-labelled features distinguish each niche, and how
far two models' niches share them.

:func:`niche_feature_tests` compares every feature inside a niche with the cells outside it
(Wilcoxon rank-sum test, Benjamini–Hochberg correction over the features of one niche),
with the effect = difference of mean activation, the active-fraction difference (fraction of
cells with activation above ``active_threshold``) and ``score`` = effect × max(active-fraction
difference, 0). :func:`top_concepts` keeps the ``k`` positive-effect features per niche ranked
by effect (then active-fraction difference, then score), which is the manuscript's ranking;
``rank_by="score"`` ranks by the product instead.

Cross-model agreement (:func:`cross_model_jaccard`) needs two runs: the top-``k`` concept id
sets of matched niches are compared by Jaccard index. Niches are matched by
:func:`match_niches` (Hungarian assignment on the maximum absolute difference of their label
composition, :func:`niche_composition`) or by an explicit mapping. Recipe for two finished
run directories ``a`` and ``b`` (same dataset, same layer naming) of a spatial dataset with
labels::

    import pandas as pd
    from interpretfm.niche_concepts import cross_model_jaccard, match_niches, niche_composition
    ta = pd.read_csv("<run a>/10_niches/<layer>/top_concepts.csv")
    tb = pd.read_csv("<run b>/10_niches/<layer>/top_concepts.csv")
    la = pd.read_csv("<run a>/10_niches/<layer>/labels.csv", index_col="cell")["niche"]
    lb = pd.read_csv("<run b>/10_niches/<layer>/labels.csv", index_col="cell")["niche"].reindex(la.index)
    ref = pd.read_csv("<run a>/01_ingest/cells.csv", index_col="cell")["label"].reindex(la.index)
    ok = la.notna() & lb.notna()
    mapping, cost = match_niches(niche_composition(la[ok], ref[ok]), niche_composition(lb[ok], ref[ok]))
    pairs = cross_model_jaccard(ta, tb, mapping)        # one row per niche pair; matched pairs flagged
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RANK_KEYS = {"effect": ["effect", "delta_frac_active", "score"], "score": ["score", "effect", "delta_frac_active"]}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (``nan`` stays ``nan``)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    order = np.argsort(v)
    n = len(v)
    ranked = v[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q


def _feature_table(n: int, features: Optional[pd.DataFrame]) -> pd.DataFrame:
    if features is None:
        return pd.DataFrame({"feature": np.arange(n), "concept": [f"feature_{i}" for i in range(n)]})
    f = features.reset_index() if "feature" not in features.columns else features.copy()
    if "feature" not in f.columns:
        f = f.rename(columns={f.columns[0]: "feature"})
    if len(f) != n:
        raise ValueError(f"niche concepts: {len(f)} feature rows for {n} feature columns")
    if "concept" not in f.columns:
        f["concept"] = f["feature"].astype(str)
    return f.reset_index(drop=True)


def niche_feature_tests(X, niches: Sequence, features: Optional[pd.DataFrame] = None, active_threshold: float = 0.0,
                        min_cells: int = 20) -> pd.DataFrame:
    """Inside-versus-outside tests of every feature for every niche.

    ``X``: cells × features (the projected features); ``niches``: one label per cell;
    ``features``: optional table with one row per column of ``X`` (``feature``, ``concept``
    and any other columns, kept in the output). Niches with fewer than ``min_cells`` cells
    inside or outside are skipped. Returns one row per niche × feature with ``n_in``,
    ``n_out``, ``mean_in``, ``mean_out``, ``effect``, ``frac_active_in``, ``frac_active_out``,
    ``delta_frac_active``, ``score``, ``pval``, ``qval`` and ``rank`` (by effect).
    """
    from scipy.stats import ranksums

    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(niches)
    feats = _feature_table(X.shape[1], features)
    out = []
    for niche in sorted(set(y.tolist()), key=lambda v: (str(type(v)), v)):
        inside = y == niche
        n_in, n_out = int(inside.sum()), int((~inside).sum())
        if n_in < min_cells or n_out < min_cells:
            continue
        Xin, Xout = X[inside], X[~inside]
        mean_in, mean_out = Xin.mean(0).astype(float), Xout.mean(0).astype(float)
        frac_in = (Xin > active_threshold).mean(0).astype(float)
        frac_out = (Xout > active_threshold).mean(0).astype(float)
        pvals = np.array([float(ranksums(Xin[:, j], Xout[:, j])[1]) for j in range(X.shape[1])])
        t = feats.copy()
        t.insert(0, "niche", niche)
        t["n_in"], t["n_out"] = n_in, n_out
        t["mean_in"], t["mean_out"] = mean_in, mean_out
        t["effect"] = mean_in - mean_out
        t["frac_active_in"], t["frac_active_out"] = frac_in, frac_out
        t["delta_frac_active"] = frac_in - frac_out
        t["score"] = t["effect"] * np.maximum(t["delta_frac_active"], 0.0)
        t["pval"] = pvals
        t["qval"] = bh_fdr(pvals)
        t = t.sort_values(RANK_KEYS["effect"], ascending=False, kind="stable").reset_index(drop=True)
        t["rank"] = np.arange(1, len(t) + 1)
        out.append(t)
    if not out:
        return pd.DataFrame(columns=["niche", "feature", "concept", "effect", "pval", "qval", "rank"])
    return pd.concat(out, ignore_index=True)


def top_concepts(tests: pd.DataFrame, k: int = 25, rank_by: str = "effect", require_positive_effect: bool = True,
                 max_q: Optional[float] = None) -> pd.DataFrame:
    """The top ``k`` features per niche (``rank_by`` ``effect`` or ``score``; positive effects
    only by default; ``max_q`` keeps features with an adjusted p-value at most that)."""
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {list(RANK_KEYS)}, got {rank_by!r}")
    t = tests.copy()
    if require_positive_effect:
        t = t[t["effect"] > 0]
    if max_q is not None:
        t = t[t["qval"] <= float(max_q)]
    parts = []
    for niche, g in t.groupby("niche", sort=True):
        g = g.sort_values(RANK_KEYS[rank_by], ascending=False, kind="stable").head(int(k)).copy()
        g["rank"] = np.arange(1, len(g) + 1)
        g["top_k"] = int(k)
        g["rank_by"] = rank_by
        parts.append(g)
    if not parts:
        return t.iloc[0:0].assign(top_k=pd.Series(dtype=int), rank_by=pd.Series(dtype=str))
    return pd.concat(parts, ignore_index=True)


def concept_sets(top: pd.DataFrame, key: str = "concept") -> Dict[str, set]:
    """``{niche: set of concept ids}`` of a top-concepts table."""
    return {str(n): set(g[key].astype(str).str.strip().str.lower()) for n, g in top.groupby("niche", sort=True)}


def niche_composition(niches: Sequence, labels: Sequence) -> pd.DataFrame:
    """Fraction of each label (columns) in each niche (rows)."""
    t = pd.crosstab(pd.Series(np.asarray(niches).astype(str), name="niche"), pd.Series(np.asarray(labels).astype(str), name="label"))
    return t.div(t.sum(1), axis=0)


def match_niches(comp_a: pd.DataFrame, comp_b: pd.DataFrame) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Match niches of two runs by label composition: Hungarian assignment minimising the
    maximum absolute difference of label fractions. Returns the mapping (niche of ``a`` →
    niche of ``b``) and the cost matrix."""
    from scipy.optimize import linear_sum_assignment

    cols = comp_a.columns.union(comp_b.columns)
    A = comp_a.reindex(columns=cols, fill_value=0.0)
    B = comp_b.reindex(columns=cols, fill_value=0.0)
    cost = np.array([[float((A.loc[i] - B.loc[j]).abs().max()) for j in B.index] for i in A.index])
    r, c = linear_sum_assignment(cost)
    mapping = {str(A.index[i]): str(B.index[j]) for i, j in zip(r, c)}
    return mapping, pd.DataFrame(cost, index=A.index.astype(str), columns=B.index.astype(str))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def cross_model_jaccard(top_a: pd.DataFrame, top_b: pd.DataFrame, matching: Optional[Dict[str, str]] = None,
                        key: str = "concept") -> pd.DataFrame:
    """Jaccard index of the top-concept sets for every niche pair of two runs; ``matched`` marks
    the pairs of ``matching`` (niche of ``a`` → niche of ``b``; default: equal niche ids)."""
    sa, sb = concept_sets(top_a, key), concept_sets(top_b, key)
    if matching is None:
        matching = {n: n for n in sa if n in sb}
    rows = []
    for i in sorted(sa):
        for j in sorted(sb):
            shared = sorted(sa[i] & sb[j])
            rows.append({"niche_a": i, "niche_b": j, "matched": matching.get(i) == j, "n_a": len(sa[i]), "n_b": len(sb[j]),
                         "n_shared": len(shared), "jaccard": jaccard(sa[i], sb[j]), "shared": ";".join(shared)})
    return pd.DataFrame(rows)


def summarize_jaccard(pairs: pd.DataFrame) -> Dict[str, Any]:
    """Mean Jaccard over matched and unmatched niche pairs."""
    m, u = pairs[pairs["matched"]], pairs[~pairs["matched"]]
    return {"n_matched": int(len(m)), "matched_mean_jaccard": float(m["jaccard"].mean()) if len(m) else None,
            "matched_n_shared": [int(v) for v in m["n_shared"]], "matched_jaccard": [float(v) for v in m["jaccard"]],
            "unmatched_mean_jaccard": float(u["jaccard"].mean()) if len(u) else None}


# --- cross-model recipe ---------------------------------------------------------------------------


def read_niche_run(run_dir, layer: str, config: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """The top-concept table and the niche labels (indexed by cell) of a finished ``niches`` stage:
    the sweep winner, or the fixed configuration ``config`` (``r18um/xm/kmeans/k3``: the
    ``niche_at_<config>`` column of ``labels.csv`` and ``top_concepts_at_<config>.csv``)."""
    from pathlib import Path

    from interpretfm.runner import stage_dir

    ldir = Path(stage_dir(run_dir, "niches")) / layer
    labels = pd.read_csv(ldir / "labels.csv", dtype={"cell": str}).set_index("cell")
    if config is None:
        return pd.read_csv(ldir / "top_concepts.csv"), labels["niche"].astype(str)
    col = f"niche_at_{config}"
    if col not in labels.columns:
        raise KeyError(f"{ldir / 'labels.csv'}: no column {col!r} (fixed configurations: "
                       f"{[c for c in labels.columns if c.startswith('niche_at_')]})")
    return pd.read_csv(ldir / f"top_concepts_at_{config.replace('/', '_')}.csv"), labels[col].astype(str)


def cross_model_concepts(runs: Dict[str, Tuple[pd.DataFrame, pd.Series]], ref_labels: pd.Series, key: str = "concept",
                         ) -> pd.DataFrame:
    """The manuscript's cross-model comparison of niche concept labels for every pair of runs
    (Supplementary Note S1): niches are matched by label composition on the cells both runs
    labelled (:func:`match_niches` with ``ref_labels``, a cell-indexed Series of the dataset's
    labels), and the top-concept sets of matched and unmatched niche pairs are compared
    (:func:`cross_model_jaccard`). One row per pair of runs: the matching, the shared labels and
    Jaccard of every matched niche (in the order of the first run's niches), and the matched and
    unmatched means. ``runs`` maps a name to ``(top_concepts, niche_labels)`` as
    :func:`read_niche_run` returns them."""
    names = list(runs)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            top_a, lab_a = runs[a]
            top_b, lab_b = runs[b]
            cells = lab_a.index.intersection(lab_b.index)
            ref = ref_labels.reindex(cells)
            ok = ref.notna().to_numpy()
            mapping, _cost = match_niches(niche_composition(lab_a.reindex(cells)[ok], ref[ok]),
                                          niche_composition(lab_b.reindex(cells)[ok], ref[ok]))
            pairs = cross_model_jaccard(top_a, top_b, mapping, key=key)
            summ = summarize_jaccard(pairs)
            matched = pairs[pairs["matched"]].sort_values("niche_a", key=lambda s: s.astype(str).map(lambda v: (len(v), v)))
            rows.append({"model_a": a, "model_b": b, "n_cells": int(ok.sum()),
                         "matching": ";".join(f"{k}->{v}" for k, v in sorted(mapping.items(), key=lambda kv: (len(kv[0]), kv[0]))),
                         "n_shared": [int(v) for v in matched["n_shared"]], "jaccard": [round(float(v), 4) for v in matched["jaccard"]],
                         "matched_mean_jaccard": summ["matched_mean_jaccard"], "unmatched_mean_jaccard": summ["unmatched_mean_jaccard"],
                         "max_shared_matched": int(matched["n_shared"].max()) if len(matched) else 0})
    return pd.DataFrame(rows)


__all__ = ["RANK_KEYS", "bh_fdr", "concept_sets", "cross_model_concepts", "cross_model_jaccard", "jaccard", "match_niches",
           "niche_composition", "niche_feature_tests", "read_niche_run", "summarize_jaccard", "top_concepts"]
