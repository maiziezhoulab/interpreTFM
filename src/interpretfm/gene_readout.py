"""Gene-level linear readout of concept membership on unseen genes.

Each measured gene is represented by the mean of its token activations over the section
(the mean hidden state, or the mean SAE activation). Genes are split into whole-gene
folds (10% test genes each, the rest train+validation); for every concept with 5-100
panel genes an L2-regularized logistic regression (C = 1, balanced class weights) is
fitted on the train+validation genes and scored by the AUROC on the test genes. Inputs are
standardized on the training genes; a center-only variant is reported as a sensitivity
analysis because standardization inflates the rare activations of sparse features. The
best single dimension (chosen by training AUROC, both signs for non-SAE representations)
is scored in the same way.

Controls built from the expression matrix: each gene's co-expression profile (Pearson
correlation with every panel gene, or its first 50 principal components), its mean
log-normalized expression per cell type (z-scored across types), its mean expression and
detection rate (``trivial``), and random Gaussian vectors of the hidden dimension.

Gene-symbol controls (:func:`symbol_representations`): each gene symbol's binary profile of
character trigrams (``names_tri``) and its first 50 principal components (``names_pc50``).
A representation can also be scored after removing the part linearly predictable from
``names_pc50`` (:func:`remove_component`: least squares with an intercept, fitted on the
split's training genes only, applied to all genes). Further activation stores of the same
model (for example of a randomly re-initialized model) enter as representations of their
own: the mean hidden state of every gene in that store.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from interp_pipeline.get_annotation.f1_alignment import _split_genes, first_copy_row_mask
from interpretfm.genes import TokenResolver
from interpretfm.sae import Encoder
from interpretfm.stores import iter_shards

CONTROLS = ("random", "coexpression", "coexpression_pc50", "celltype", "trivial")
SYMBOL_CONTROLS = ("names_tri", "names_pc50")
SCALINGS = ("standardize", "center")
MODEL_REPRESENTATIONS = ("hidden", "sae", "untrained")
SAE_FAMILIES = {"sae", "untrained"}
SYMBOL_REMOVED = "symbol_removed"


@dataclass
class ReadoutParams:
    """``stages.gene_readout``."""

    n_splits: int = 5
    C: float = 1.0
    scaling: Sequence[str] = SCALINGS
    controls: Sequence[str] = CONTROLS
    concept_size: Tuple[int, int] = (5, 100)
    valid_frac: float = 0.2
    test_frac: float = 0.1
    max_cells: int = 100_000
    min_type_cells: int = 50
    n_pcs: int = 50
    seed: int = 0
    max_iter: int = 3000
    dedup_cells: bool = True
    max_shards: Optional[int] = None
    device: str = "auto"
    representations: Sequence[str] = MODEL_REPRESENTATIONS
    universe: Optional[str] = None
    first_split: Optional[str] = None
    stores: Dict[str, str] = field(default_factory=dict)
    symbol_residuals: Sequence[str] = ()

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# --- representations ----------------------------------------------------------------------------


def symbol_trigrams(symbol: str) -> set:
    """Character trigrams of a gene symbol, upper case, padded with ``^`` and ``$`` (``KRT5``:
    ``^KR``, ``KRT``, ``RT5``, ``T5$``)."""
    s = "^" + str(symbol).upper() + "$"
    return {s[i:i + 3] for i in range(len(s) - 2)}


def symbol_representations(symbols: Sequence[str], n_pcs: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """``(names_tri, names_pc50, trigrams)``: the binary gene x trigram matrix over the trigrams
    that occur in ``symbols`` (sorted), and the first ``n_pcs`` principal components of the
    column-centred matrix (scores ``U * S``)."""
    grams = sorted(set().union(*[symbol_trigrams(s) for s in symbols])) if len(symbols) else []
    index = {g: i for i, g in enumerate(grams)}
    M = np.zeros((len(symbols), len(grams)), np.float32)
    for r, s in enumerate(symbols):
        for g in symbol_trigrams(s):
            M[r, index[g]] = 1
    C = M - M.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(C.astype(np.float64), full_matrices=False)
    k = min(int(n_pcs), len(S))
    return M, (U[:, :k] * S[:k]).astype(np.float32), grams


def remove_component(X: np.ndarray, P: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``X`` minus its least-squares prediction from ``[1, P]``, the coefficients fitted on
    ``rows`` only (the training genes of a split) and applied to every row."""
    A = np.hstack([np.ones((len(P), 1)), P.astype(np.float64)])
    B, *_ = np.linalg.lstsq(A[rows], X[rows].astype(np.float64), rcond=None)
    return (X - A @ B).astype(np.float32)


def read_gene_list(path: Union[str, Path]) -> List[str]:
    """Gene keys from a text file, one per line (``#`` starts a comment line)."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    genes = [ln for ln in lines if ln and not ln.startswith("#")]
    if not genes:
        raise ValueError(f"stages.gene_readout.universe: {path} lists no genes")
    return genes


def read_split_file(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """``(train + validation genes, test genes)`` of a whole-gene held-out split written by a
    concept evaluation (JSON with ``train_genes``, ``valid_genes`` and ``test_genes``)."""
    import json

    with open(path) as fh:
        d = json.load(fh)
    missing = [k for k in ("train_genes", "valid_genes", "test_genes") if k not in d]
    if missing:
        raise ValueError(f"stages.gene_readout.split.first_from: {path} has no {missing}")
    return [str(g) for g in d["train_genes"]] + [str(g) for g in d["valid_genes"]], [str(g) for g in d["test_genes"]]


def gene_means(store_root: Union[str, Path], layer: str, resolver: TokenResolver, keys: Sequence[str],
               encoders: Optional[Dict[str, Encoder]] = None, dedup_cells: bool = True,
               max_shards: Optional[int] = None, batch_size: int = 8192) -> Dict[str, np.ndarray]:
    """Per-gene mean hidden state (``hidden``) and, per encoder, mean latent activation,
    over all stored tokens of the gene; ``count`` is the number of tokens per gene."""
    key_to_row = {k: i for i, k in enumerate(keys)}
    G = len(keys)
    sums: Dict[str, Optional[np.ndarray]] = {"hidden": None}
    sums.update({name: None for name in (encoders or {})})
    counts = np.zeros(G, dtype=np.int64)
    seen: set = set()
    for _name, X, idx in iter_shards(store_root, layer, max_shards=max_shards):
        toks = idx.get("token_ids")
        if idx.get("token_unit", "gene") != "gene" or toks is None or len(toks) != X.shape[0]:
            continue
        rows = resolver.resolve_rows(toks, key_to_row)
        if dedup_cells:
            ex = idx.get("example_ids")
            if ex is not None and len(ex) == len(toks):
                rows[~first_copy_row_mask(ex, seen)] = -1
        m = rows >= 0
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        Xm = X[torch.as_tensor(sel)]
        target = rows[sel]
        np.add.at(counts, target, 1)
        if sums["hidden"] is None:
            sums["hidden"] = np.zeros((G, X.shape[1]), dtype=np.float64)
        np.add.at(sums["hidden"], target, Xm.numpy().astype(np.float64))
        for name, enc in (encoders or {}).items():
            Z = torch.cat([enc.encode(Xm[i:i + batch_size]).cpu() for i in range(0, Xm.shape[0], batch_size)]).numpy()
            if sums[name] is None:
                sums[name] = np.zeros((G, Z.shape[1]), dtype=np.float64)
            np.add.at(sums[name], target, Z.astype(np.float64))
    if sums["hidden"] is None:
        raise RuntimeError(f"no stored token of layer {layer!r} maps to a panel gene")
    out: Dict[str, np.ndarray] = {"count": counts}
    denom = np.maximum(counts, 1)[:, None]
    for name, S in sums.items():
        out[name] = (S / denom).astype(np.float32) if S is not None else np.zeros((G, 0), dtype=np.float32)
    return out


def expression_representations(adata, keys: Sequence[str], labels_column: Optional[str], params: ReadoutParams,
                               controls: Sequence[str]) -> Dict[str, np.ndarray]:
    """The expression-derived controls over ``keys`` (the genes in ``adata.var`` order must
    match ``keys``): ``coexpression``, ``coexpression_pc50``, ``celltype``, ``trivial``."""
    rng = np.random.default_rng(params.seed)
    n = adata.n_obs
    idx = np.sort(rng.choice(n, params.max_cells, replace=False)) if n > params.max_cells else np.arange(n)
    X = adata.X[idx] if n > params.max_cells else adata.X[:]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float32)
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1
    L = np.log1p(X / tot * 1e4)
    out: Dict[str, np.ndarray] = {"n_cells": np.array(len(idx))}
    if "trivial" in controls:
        out["trivial"] = np.stack([L.mean(0), (X > 0).mean(0)], 1).astype(np.float32)
    if "coexpression" in controls or "coexpression_pc50" in controls:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            C = np.nan_to_num(np.corrcoef(L, rowvar=False))
        if "coexpression" in controls:
            out["coexpression"] = C.astype(np.float32)
        if "coexpression_pc50" in controls:
            Cc = C - C.mean(0, keepdims=True)
            U, S, _ = np.linalg.svd(Cc, full_matrices=False)
            k = min(params.n_pcs, len(S))
            out["coexpression_pc50"] = (U[:, :k] * S[:k]).astype(np.float32)
    if "celltype" in controls:
        if labels_column is None or labels_column not in adata.obs.columns:
            out["celltype"] = None  # type: ignore[assignment]
        else:
            ct = adata.obs[labels_column].astype(str).values[idx]
            types, cnt = np.unique(ct, return_counts=True)
            types = types[cnt >= params.min_type_cells]
            if len(types) >= 2:
                prof = np.stack([L[ct == t].mean(0) for t in types], 1)
                prof = (prof - prof.mean(1, keepdims=True)) / np.maximum(prof.std(1, keepdims=True), 1e-8)
                out["celltype"] = prof.astype(np.float32)
                out["celltypes"] = types
            else:
                out["celltype"] = None  # type: ignore[assignment]
    return out


# --- scoring ------------------------------------------------------------------------------------


def _rank_cols(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(X, axis=0, method="average").astype(np.float64)


def single_dim_auroc(Xtv: np.ndarray, Xte: np.ndarray, Ytv: np.ndarray, Yte: np.ndarray, both_signs: bool):
    """Per concept the best training-AUROC dimension (and sign) and its test AUROC:
    ``(dim, sign, train_auroc, test_auroc)``."""
    Rtv = _rank_cols(Xtv)
    p = Ytv.sum(0).astype(np.float64)
    n = Ytv.shape[0] - p
    A = (Ytv.T.astype(np.float64) @ Rtv - (p * (p + 1) / 2)[:, None]) / (p * n)[:, None]
    score = np.abs(A - 0.5) if both_signs else A
    d = score.argmax(1)
    a_tv = A[np.arange(len(d)), d]
    sign = np.where(both_signs & (a_tv < 0.5), -1, 1)
    Rte = _rank_cols(Xte)[:, d]
    pt = Yte.sum(0).astype(np.float64)
    nt = Yte.shape[0] - pt
    a_te = ((Yte * Rte).sum(0) - pt * (pt + 1) / 2) / (pt * nt)
    a_te = np.where(sign < 0, 1 - a_te, a_te)
    a_tv = np.where(sign < 0, 1 - a_tv, a_tv)
    return d, sign, a_tv, a_te


def probe_prepare(Xtv: np.ndarray, Xte: np.ndarray, scaling: str) -> Tuple[np.ndarray, np.ndarray]:
    """Center (and standardize) on the training genes, drop constant dimensions and, when
    there are more dimensions than training genes, project onto the training row space."""
    mu, sd = Xtv.mean(0), Xtv.std(0)
    keep = sd > 1e-8
    scale = sd[keep] if scaling == "standardize" else 1.0
    Ztv = (Xtv[:, keep] - mu[keep]) / scale
    Zte = (Xte[:, keep] - mu[keep]) / scale
    if Ztv.shape[1] > Ztv.shape[0]:
        _, S, Vt = np.linalg.svd(Ztv.astype(np.float64), full_matrices=False)
        r = int((S > S[0] * 1e-10).sum()) if len(S) else 0
        V = Vt[:r].T
        Ztv, Zte = Ztv @ V, Zte @ V
    return np.ascontiguousarray(Ztv, dtype=np.float64), np.ascontiguousarray(Zte, dtype=np.float64)


def probe_auroc(Xtv: np.ndarray, Xte: np.ndarray, Ytv: np.ndarray, Yte: np.ndarray, C: float, scaling: str,
                max_iter: int = 3000) -> np.ndarray:
    """Test AUROC of an L2 logistic regression per concept (columns of ``Y``)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Ztv, Zte = probe_prepare(Xtv, Xte, scaling)
    out = np.full(Ytv.shape[1], np.nan)
    if Ztv.shape[1] == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for c in range(Ytv.shape[1]):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, solver="lbfgs")
            m.fit(Ztv, Ytv[:, c])
            out[c] = roc_auc_score(Yte[:, c], m.decision_function(Zte))
    return out


def make_splits(universe: Sequence[str], n_splits: int, valid_frac: float, test_frac: float, seed: int,
                first: Optional[Tuple[Sequence[str], Sequence[str]]] = None) -> Dict[str, Tuple[List[str], List[str]]]:
    """``{split name: (train+validation genes, test genes)}`` with the whole-gene split rule
    (split ``s`` drawn with seed ``seed + s``). ``first`` (train+validation genes, test genes)
    replaces split ``s0``, restricted to ``universe``."""
    out = {}
    U = set(universe)
    for s in range(int(n_splits)):
        if s == 0 and first is not None:
            out["s0"] = (sorted(set(first[0]) & U), sorted(set(first[1]) & U))
            if set(out["s0"][0]) & set(out["s0"][1]):
                raise ValueError("stages.gene_readout.split.first_from: a gene is both a training and a test gene")
            continue
        tr, va, te = _split_genes(list(universe), valid_frac, test_frac, seed + s)
        out[f"s{s}"] = (sorted(tr | va), sorted(te))
    return out


def readout(reps: Dict[str, Tuple[np.ndarray, str]], Y: np.ndarray, term_ids: Sequence[str], sizes: Sequence[int],
            gene_index: Dict[str, int], splits: Dict[str, Tuple[List[str], List[str]]], params: ReadoutParams, log=None,
            residuals: Optional[Tuple[np.ndarray, Sequence[str]]] = None):
    """Score every representation (``name -> (matrix over the universe genes, family)``) on every
    split; returns the per-concept ``DataFrame``. ``residuals = (P, names)`` also scores each
    representation in ``names`` with the part predictable from ``P`` removed (fitted on the
    split's training genes; :func:`remove_component`), as ``<name>|symbol_removed``."""
    import pandas as pd

    frames = []
    for sname, (tv, te) in splits.items():
        itv = np.array([gene_index[g] for g in tv])
        ite = np.array([gene_index[g] for g in te])
        ytv, yte = Y[itv], Y[ite]
        keep = (yte.sum(0) >= 1) & (ytv.sum(0) >= 2) & (ytv.sum(0) < len(itv))
        if not keep.any():
            continue
        ytv, yte = ytv[:, keep], yte[:, keep]
        tk = np.asarray(term_ids, dtype=object)[keep]
        sz = np.asarray(sizes)[keep]
        todo = dict(reps)
        if residuals is not None:
            P, names = residuals
            for name in names:
                M, fam = reps[name]
                todo[f"{name}|{SYMBOL_REMOVED}"] = (remove_component(M, P, itv), fam)
        for rname, (M, fam) in todo.items():
            Xtv, Xte = M[itv], M[ite]
            d, sign, a_tv, a_te = single_dim_auroc(Xtv, Xte, ytv, yte, both_signs=fam not in SAE_FAMILIES)
            df = pd.DataFrame({"split": sname, "representation": rname, "family": fam, "concept": tk, "n_genes": sz,
                               "n_tv_pos": ytv.sum(0), "n_test_pos": yte.sum(0), "n_tv": len(itv), "n_test": len(ite),
                               "single_dim": d, "single_sign": sign, "single_train_auroc": a_tv, "single_test_auroc": a_te})
            for scaling in params.scaling:
                df[f"probe_test_auroc_{scaling}"] = probe_auroc(Xtv, Xte, ytv, yte, params.C, scaling, params.max_iter)
            frames.append(df)
            if log is not None:
                log.info("gene_readout: %s %s: %d concepts, probe AUROC %.3f", sname, rname, len(tk),
                         float(np.nanmean(df[f"probe_test_auroc_{params.scaling[0]}"])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_readout(per_concept) -> Tuple[Any, Any]:
    """``(per_split, overall)``: mean test AUROC and share >= 0.7 per representation and split,
    and mean +- SD over splits."""
    import pandas as pd

    if per_concept.empty:
        return pd.DataFrame(), pd.DataFrame()
    score_cols = ["single_test_auroc"] + [c for c in per_concept.columns if c.startswith("probe_test_auroc_")]
    g = per_concept.groupby(["representation", "family", "split"])
    per_split = g[score_cols].mean().add_prefix("mean_")
    share = per_concept[score_cols].ge(0.7).groupby([per_concept[c] for c in ("representation", "family", "split")]).mean()
    per_split = per_split.join(share.add_prefix("share07_"))
    per_split["n_concepts"] = g.size()
    per_split = per_split.reset_index()
    agg = per_split.groupby(["representation", "family"])
    overall = agg[[f"mean_{c}" for c in score_cols]].agg(["mean", "std"])
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]
    overall["n_splits"] = agg.size()
    overall["n_concepts_mean"] = agg["n_concepts"].mean()
    return per_split, overall.reset_index()


def params_from_mapping(params: Dict[str, Any], default_seed: int = 0) -> ReadoutParams:
    from interpretfm.params import get_list, get_mapping, get_param

    split = get_mapping(params, "split", "gene_readout")
    size = get_list(params, "concept_size", [5, 100], "gene_readout")
    if len(size) != 2:
        raise ValueError("stages.gene_readout.concept_size: expected [min, max]")
    stores = get_mapping(params, "stores", "gene_readout")
    reserved = set(MODEL_REPRESENTATIONS) | set(CONTROLS) | set(SYMBOL_CONTROLS)
    for name, root in stores.items():
        if name in reserved or "|" in str(name) or not isinstance(root, str):
            raise ValueError(f"stages.gene_readout.stores.{name}: expected a store path under a name other than "
                             f"{sorted(reserved)} (without '|')")
    p = ReadoutParams(
        n_splits=get_param(params, "n_splits", 5, "gene_readout", int),
        C=get_param(params, "C", 1.0, "gene_readout", float),
        scaling=get_list(params, "scaling", list(SCALINGS), "gene_readout", choices=SCALINGS),
        controls=get_list(params, "controls", list(CONTROLS), "gene_readout", choices=CONTROLS + SYMBOL_CONTROLS),
        concept_size=(int(size[0]), int(size[1])),
        valid_frac=float(split.get("valid", 0.2)), test_frac=float(split.get("test", 0.1)),
        max_cells=get_param(params, "max_cells", 100_000, "gene_readout", int),
        min_type_cells=get_param(params, "min_type_cells", 50, "gene_readout", int),
        n_pcs=get_param(params, "n_pcs", 50, "gene_readout", int),
        seed=get_param(params, "seed", default_seed, "gene_readout", int),
        max_iter=get_param(params, "max_iter", 3000, "gene_readout", int),
        dedup_cells=bool(params.get("dedup_cells", True)),
        max_shards=get_param(params, "max_shards", None, "gene_readout", int),
        device=get_param(params, "device", "auto", "gene_readout", str),
        representations=get_list(params, "representations", list(MODEL_REPRESENTATIONS), "gene_readout",
                                 choices=MODEL_REPRESENTATIONS),
        universe=get_param(params, "universe", None, "gene_readout", str),
        first_split=get_param(split, "first_from", None, "gene_readout.split", str),
        stores={str(k): str(v) for k, v in stores.items()},
        symbol_residuals=get_list(params, "symbol_residuals", [], "gene_readout",
                                  choices=MODEL_REPRESENTATIONS + CONTROLS + SYMBOL_CONTROLS + tuple(stores)),
    )
    if p.n_splits <= 0 or p.C <= 0 or not p.scaling:
        raise ValueError("stages.gene_readout: n_splits and C must be > 0 and scaling non-empty")
    return p


def params_dict(p: ReadoutParams) -> Dict[str, Any]:
    d = asdict(p)
    d["scaling"], d["controls"], d["concept_size"] = list(p.scaling), list(p.controls), list(p.concept_size)
    d["representations"], d["symbol_residuals"] = list(p.representations), list(p.symbol_residuals)
    return d
