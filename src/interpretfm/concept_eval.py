"""Held-out concept alignment of a representation's units with the panel concept universe.

Each gene-token row of the store (one gene in one cell) carries its gene's concept
memberships as labels. A unit ``k`` of a representation (an SAE feature, a rectified raw
dimension, an untrained-SAE feature) predicts membership of concept ``c`` at threshold
``τ`` when its activation exceeds ``τ``; the prediction is scored by F1 against the
gene-level label. Rows are partitioned into train/validation/test either per row (a
deterministic hash of seed, shard, row index and token: *row split*, 70/20/10) or per gene
(all rows of a gene follow it: *whole-gene split*). Per concept and threshold the
``top_m`` units by validation F1 are kept and scored on the test partition.

The *concept F1* summary (:func:`summarize_concept_f1`): at ``τ = 0.15``, for every concept
with 5-100 panel genes and positive rows in both held-out partitions, the unit with the
highest validation F1 (ties: lowest index) scored by its test F1; the summary is the mean
over concepts, so each concept counts once.

Duplicated cells (a store that recorded a cell more than once) are reduced to their
first copy before evaluation (:func:`interp_pipeline.get_annotation.f1_alignment.first_copy_row_mask`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import sparse

from interp_pipeline.get_annotation.f1_alignment import (
    _assign_gene_token_row_split,
    _split_genes,
    accumulate_tp_sparse_into_dense,
    compute_f1_from_counts,
    first_copy_row_mask,
)
from interpretfm.genes import TokenResolver
from interpretfm.sae import Encoder
from interpretfm.stores import iter_shards, list_shards, select_shards, torch_load

THRESHOLDS = (0.0, 0.15, 0.3, 0.6)
UNITS = ("row", "gene")
REPRESENTATIONS = ("sae", "raw", "untrained")
SPLIT_NAMES = {0: "valid", 1: "test", -1: "train"}


@dataclass
class HeldoutParams:
    """Settings of one held-out evaluation (``stages.concept_eval``)."""

    thresholds: Sequence[float] = THRESHOLDS
    valid_frac: float = 0.2
    test_frac: float = 0.1
    seed: int = 0
    top_m: int = 200
    batch_size: int = 8192
    max_shards: Optional[int] = None
    dedup_cells: bool = True
    device: str = "auto"

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class RowAssignment:
    """Per shard: the concept-table row of every stored row (-1: not a panel gene / dropped
    duplicate) and the held-out partition of every row (0 valid, 1 test, -1 train)."""

    unit: str
    gt_rows: Dict[str, np.ndarray]
    split: Dict[str, np.ndarray]
    observed_gene_counts: Dict[str, int]
    genes_by_split: Dict[str, List[str]] = field(default_factory=dict)
    rows_dropped_duplicates: int = 0
    n_cells_seen: int = 0

    def meta(self, params: HeldoutParams) -> Dict[str, Any]:
        counts = list(self.observed_gene_counts.values())
        m: Dict[str, Any] = {
            "unit": self.unit, "valid_frac": params.valid_frac, "test_frac": params.test_frac, "seed": params.seed,
            "n_observed_genes": len(self.observed_gene_counts),
            "n_rows_mapped": int(sum(counts)),
            "observed_gene_count_min": int(min(counts)) if counts else 0,
            "observed_gene_count_median": float(np.median(counts)) if counts else 0.0,
            "dedup_cells": params.dedup_cells, "rows_dropped_duplicate_cells": int(self.rows_dropped_duplicates),
            "n_cells": int(self.n_cells_seen),
        }
        if self.unit == "gene":
            m.update({f"n_{k}_genes": len(v) for k, v in self.genes_by_split.items()})
            m.update({f"{k}_genes": sorted(v) for k, v in self.genes_by_split.items()})
        return m


def assign_rows(store_root: Union[str, Path], layer: str, resolver: TokenResolver, key_to_row: Dict[str, int],
                unit: str, params: HeldoutParams) -> RowAssignment:
    """First pass over the shard indexes: resolve tokens to concept-table rows and assign partitions."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    shards = select_shards(list_shards(store_root, layer), params.max_shards)
    gt_rows: Dict[str, np.ndarray] = {}
    tokens_by_shard: Dict[str, List[Any]] = {}
    seen: set = set()
    counts: Dict[str, int] = {}
    dropped = 0
    row_keys = {r: k for k, r in key_to_row.items()}
    for shard in shards:
        idx = torch_load(shard / "index.pt")
        if idx.get("token_unit", "gene") != "gene" or idx.get("token_ids") is None:
            continue
        toks = idx["token_ids"]
        rows = resolver.resolve_rows(toks, key_to_row)
        if params.dedup_cells:
            ex = idx.get("example_ids")
            if ex is None or len(ex) != len(toks):
                raise RuntimeError(f"{shard}/index.pt: de-duplication needs example_ids aligned with token_ids")
            keep = first_copy_row_mask(ex, seen)
            dropped += int((~keep).sum())
            rows[~keep] = -1
        gt_rows[shard.name] = rows
        tokens_by_shard[shard.name] = toks
        for r in rows[rows >= 0]:
            k = row_keys[int(r)]
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        raise RuntimeError(f"no stored gene token of layer {layer!r} maps to a panel gene; check the token vocabulary "
                           "and the gene keys")
    split: Dict[str, np.ndarray] = {}
    genes_by_split: Dict[str, List[str]] = {}
    if unit == "gene":
        train, valid, test = _split_genes(sorted(counts), params.valid_frac, params.test_frac, params.seed)
        genes_by_split = {"train": sorted(train), "valid": sorted(valid), "test": sorted(test)}
        gene_split = {k: 0 for k in valid}
        gene_split.update({k: 1 for k in test})
        for name, rows in gt_rows.items():
            s = np.full(len(rows), -1, dtype=np.int8)
            for i in np.flatnonzero(rows >= 0):
                s[i] = gene_split.get(row_keys[int(rows[i])], -1)
            split[name] = s
    else:
        for name, rows in gt_rows.items():
            toks = tokens_by_shard[name]
            s = np.full(len(rows), -1, dtype=np.int8)
            for i in np.flatnonzero(rows >= 0):
                s[i] = _assign_gene_token_row_split(seed=params.seed, shard_name=name, row_index=int(i),
                                                    token_id=toks[i], valid_frac=params.valid_frac,
                                                    test_frac=params.test_frac)
            split[name] = s
    return RowAssignment(unit=unit, gt_rows=gt_rows, split=split, observed_gene_counts=counts,
                         genes_by_split=genes_by_split, rows_dropped_duplicates=dropped, n_cells_seen=len(seen))


@dataclass
class HeldoutResult:
    valid: Any  #: DataFrame: concept, feature, threshold, precision, recall, f1, tp, fp, fn, pred_pos, true_pos
    test: Any  #: DataFrame, restricted to the validation-selected (concept, feature, threshold)
    support: Any  #: DataFrame: split, concept, true_pos
    counts: Dict[str, Any]
    split_meta: Dict[str, Any]

    def write(self, out_dir: Union[str, Path]) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.valid.to_csv(out_dir / "valid_concept_f1.csv", index=False)
        self.test.to_csv(out_dir / "test_concept_f1.csv", index=False)
        self.support.to_csv(out_dir / "support.csv", index=False)
        with open(out_dir / "counts.json", "w") as fh:
            json.dump(self.counts, fh, indent=2)
        with open(out_dir / "split.json", "w") as fh:
            json.dump(self.split_meta, fh, indent=2)
        return out_dir

    @classmethod
    def read(cls, out_dir: Union[str, Path]) -> "HeldoutResult":
        import pandas as pd

        out_dir = Path(out_dir)
        with open(out_dir / "counts.json") as fh:
            counts = json.load(fh)
        with open(out_dir / "split.json") as fh:
            meta = json.load(fh)
        return cls(valid=pd.read_csv(out_dir / "valid_concept_f1.csv"), test=pd.read_csv(out_dir / "test_concept_f1.csv"),
                   support=pd.read_csv(out_dir / "support.csv"), counts=counts, split_meta=meta)


def heldout_concept_f1(store_root: Union[str, Path], layer: str, encoder: Encoder, membership: np.ndarray,
                       term_ids: Sequence[str], assignment: RowAssignment, params: HeldoutParams) -> HeldoutResult:
    """Second pass: encode the held-out rows and count true/predicted positives per unit,
    concept and threshold; returns the validation and test tables."""
    GT = np.asarray(membership, dtype=np.int8)
    K, T = encoder.n_latents, GT.shape[1]
    thresholds = [float(t) for t in params.thresholds]
    splits = ("valid", "test")
    tp = {sp: {t: np.zeros((K, T), dtype=np.int64) for t in thresholds} for sp in splits}
    pred_pos = {sp: {t: np.zeros(K, dtype=np.int64) for t in thresholds} for sp in splits}
    true_pos = {sp: np.zeros(T, dtype=np.int64) for sp in splits}
    rows_by_split = {sp: 0 for sp in splits}
    rows_total = rows_mapped = 0
    for name, X, _idx in iter_shards(store_root, layer, max_shards=params.max_shards):
        if name not in assignment.gt_rows:
            continue
        gt_rows, split = assignment.gt_rows[name], assignment.split[name]
        if len(gt_rows) != X.shape[0]:
            raise RuntimeError(f"shard {name}: {X.shape[0]} rows but {len(gt_rows)} index entries")
        if X.shape[1] != encoder.d_in:
            raise RuntimeError(f"shard {name}: hidden dim {X.shape[1]} but the encoder expects {encoder.d_in}")
        rows_total += int(X.shape[0])
        mapped = gt_rows >= 0
        rows_mapped += int(mapped.sum())
        for sp, sp_id in (("valid", 0), ("test", 1)):
            m = mapped & (split == sp_id)
            if not m.any():
                continue
            rows_by_split[sp] += int(m.sum())
            G = sparse.csr_matrix(GT[gt_rows[m]])
            true_pos[sp] += np.asarray(G.sum(axis=0)).ravel().astype(np.int64)
            Xm = X[torch.as_tensor(np.flatnonzero(m))]
            Z = torch.cat([encoder.encode(Xm[i:i + params.batch_size]).cpu() for i in range(0, Xm.shape[0], params.batch_size)])
            Z = Z.numpy()
            for t in thresholds:
                A = Z > t
                pred_pos[sp][t] += A.sum(axis=0).astype(np.int64)
                accumulate_tp_sparse_into_dense(tp[sp][t], A, G)
    frames = {}
    selected: Dict[float, np.ndarray] = {}
    for sp in splits:
        rows = []
        for t in thresholds:
            precision, recall, f1, fp, fn = compute_f1_from_counts(tp[sp][t], pred_pos[sp][t], true_pos[sp])
            if sp == "valid":
                key = f1.astype(np.float64) + 1e-9 * tp[sp][t] / (tp[sp][t].max() + 1.0)
                order = np.argsort(-key, axis=0, kind="stable")[: int(params.top_m)]  # top_m x T
                selected[t] = order
            else:
                order = selected[t]
            feats = order.ravel(order="F")
            concepts = np.repeat(np.arange(T), order.shape[0])
            rows.append({"concept": np.asarray(term_ids, dtype=object)[concepts], "feature": feats,
                         "threshold": np.full(len(feats), t), "precision": precision[feats, concepts],
                         "recall": recall[feats, concepts], "f1": f1[feats, concepts],
                         "tp": tp[sp][t][feats, concepts], "fp": fp[feats, concepts], "fn": fn[feats, concepts],
                         "pred_pos": pred_pos[sp][t][feats], "true_pos": true_pos[sp][concepts]})
        frames[sp] = _concat_rows(rows)
    support = _concat_rows([{"split": np.full(T, sp, dtype=object), "concept": np.asarray(term_ids, dtype=object),
                             "true_pos": true_pos[sp]} for sp in splits])
    counts = {"layer": layer, "n_units": int(K), "n_concepts": int(T), "thresholds": thresholds,
              "rows_total": rows_total, "rows_mapped": rows_mapped, "rows_by_split": rows_by_split,
              "top_m": int(params.top_m), "max_shards": params.max_shards,
              "concept_support": {sp: {"n_concepts_true_pos_eq_0": int((true_pos[sp] == 0).sum()),
                                       "true_pos_median": float(np.median(true_pos[sp])) if T else 0.0} for sp in splits}}
    return HeldoutResult(valid=frames["valid"], test=frames["test"], support=support, counts=counts,
                         split_meta=assignment.meta(params))


def _concat_rows(parts: List[Dict[str, np.ndarray]]):
    import pandas as pd

    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)


# --- summary ------------------------------------------------------------------------------------


def per_concept_best(valid, test, tau: float = 0.15):
    """Per concept: the unit with the highest validation F1 at ``tau`` (ties: lowest index) and
    its test F1 (``DataFrame``: concept, feature, valid_f1, test_f1)."""
    v = valid[np.isclose(valid["threshold"].astype(float), tau)][["concept", "feature", "f1"]]
    v = v.sort_values(["concept", "f1", "feature"], ascending=[True, False, True]).drop_duplicates("concept")
    t = test[np.isclose(test["threshold"].astype(float), tau)][["concept", "feature", "f1"]].rename(columns={"f1": "test_f1"})
    return v.rename(columns={"f1": "valid_f1"}).merge(t, on=["concept", "feature"], how="inner").reset_index(drop=True)


def supported_concepts(support) -> set:
    """Concepts with positive rows in both held-out partitions."""
    ok = support[support["true_pos"] > 0].groupby("concept")["split"].nunique()
    return set(ok[ok >= 2].index.astype(str))


def summarize_concept_f1(result: HeldoutResult, concept_sizes, tau: float = 0.15,
                         size_range: Tuple[int, int] = (5, 100)) -> Tuple[Dict[str, Any], Any]:
    """The concept F1 summary and its per-concept table.

    ``concept_sizes`` maps concept id to its number of panel genes. Returns
    ``(summary, per_concept)`` where ``summary`` has the mean test F1 over the concepts of
    ``size_range`` with positive rows in both partitions (``concept_f1``), the share of those
    with test F1 >= 0.5, and the means per concept-size bin.
    """
    import pandas as pd

    pc = per_concept_best(result.valid, result.test, tau)
    sizes = pd.Series(concept_sizes).astype(int)
    pc["n_genes"] = pc["concept"].astype(str).map(sizes).fillna(0).astype(int).values
    pc["supported"] = pc["concept"].astype(str).isin(supported_concepts(result.support)).values
    pc["in_size_range"] = (pc["n_genes"] >= size_range[0]) & (pc["n_genes"] <= size_range[1])
    use = pc[pc["supported"] & pc["in_size_range"]]
    bins = [1, 4, 20, 100, 300, np.inf]
    labels = ["2-4", "5-20", "21-100", "101-300", ">300"]
    binned = pc[pc["supported"]].assign(bin=pd.cut(pc.loc[pc["supported"], "n_genes"], bins, labels=labels))
    summary = {"tau": float(tau), "concept_size": list(size_range), "n_concepts": int(len(use)),
               "n_concepts_scored": int(pc["supported"].sum()),
               "concept_f1": float(use["test_f1"].mean()) if len(use) else None,
               "valid_f1_mean": float(use["valid_f1"].mean()) if len(use) else None,
               "share_test_f1_ge_0.5": float((use["test_f1"] >= 0.5).mean()) if len(use) else None,
               "by_size": {str(b): (float(g["test_f1"].mean()) if len(g) else None)
                           for b, g in binned.groupby("bin", observed=False)}}
    return summary, pc


def params_from_mapping(params: Dict[str, Any], default_seed: int = 0) -> HeldoutParams:
    from interpretfm.params import get_list, get_mapping, get_param

    split = get_mapping(params, "split", "concept_eval")
    valid = float(split.get("valid", 0.2))
    test = float(split.get("test", 0.1))
    if not (0 < valid < 1 and 0 < test < 1 and valid + test < 1):
        raise ValueError("stages.concept_eval.split: valid and test fractions must be in (0, 1) and sum below 1")
    dedup = params.get("dedup_cells", "auto")
    if dedup not in ("auto", True, False):
        raise ValueError("stages.concept_eval.dedup_cells: expected auto, true or false")
    p = HeldoutParams(
        thresholds=[float(t) for t in get_list(params, "thresholds", list(THRESHOLDS), "concept_eval")],
        valid_frac=valid, test_frac=test,
        seed=get_param(params, "seed", default_seed, "concept_eval", int),
        top_m=get_param(params, "top_m", 200, "concept_eval", int),
        batch_size=get_param(params, "batch_size", 8192, "concept_eval", int),
        max_shards=get_param(params, "max_shards", None, "concept_eval", int),
        dedup_cells=(dedup is not False), device=get_param(params, "device", "auto", "concept_eval", str),
    )
    if p.top_m <= 0 or p.batch_size <= 0 or not p.thresholds:
        raise ValueError("stages.concept_eval: top_m, batch_size must be > 0 and thresholds non-empty")
    return p


def params_dict(p: HeldoutParams) -> Dict[str, Any]:
    d = asdict(p)
    d["thresholds"] = list(p.thresholds)
    return d
