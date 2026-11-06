"""Input checks for the ``ingest`` stage: counts, gene names, labels, coordinates, QC flags and
gene-to-vocabulary coverage.

Nothing here modifies the dataset. Problems that make the run meaningless (no raw counts, a
missing column, no coordinates where the configuration declares some) raise
:class:`IngestError` with a message naming the configuration key to fix; everything else is
reported in ``dataset_qc.json``. Low-quality cells are *flagged*, never dropped: the flag
travels with the cell table and downstream stages report how many flagged cells they used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from interpretfm.genes import GeneTable, load_token_vocab


class IngestError(ValueError):
    """The dataset does not match the configuration (the message names the key to fix)."""


# --- labels -------------------------------------------------------------------------------------


def dataset_labels(ds, adata) -> Optional[np.ndarray]:
    """Per-cell labels (object array of strings) as ``dataset.labels`` (or the given
    :class:`~interpretfm.config.Labels`) declares them: the ``obs`` column, or the
    ``file_column`` of the CSV ``file`` keyed by cell id (every cell of the dataset must appear in
    it). ``None`` when the configuration has no labels."""
    import pandas as pd

    from interpretfm.config import Labels

    lab = ds if isinstance(ds, Labels) or ds is None else ds.labels
    if lab is None:
        return None
    if lab.file is not None:
        if not Path(lab.file).is_file():
            raise IngestError(f"dataset.labels.file: file not found: {lab.file}")
        table = pd.read_csv(lab.file, index_col=0)
        table.index = table.index.astype(str)
        if lab.file_column not in table.columns:
            raise IngestError(f"dataset.labels.file_column: '{lab.file_column}' not in {lab.file} (columns: {list(table.columns)})")
        values = table[lab.file_column].reindex([str(o) for o in adata.obs_names])
        if values.isna().any():
            missing = [str(o) for o, v in zip(adata.obs_names, values.isna()) if v][:5]
            raise IngestError(f"dataset.labels.file: {int(values.isna().sum())} cells of the dataset are missing from {lab.file} "
                              f"(e.g. {missing})")
        return values.astype(str).to_numpy(dtype=object)
    if lab.column not in adata.obs.columns:
        raise IngestError(f"dataset.labels.column: '{lab.column}' not in adata.obs (columns: {list(adata.obs.columns)})")
    return adata.obs[lab.column].astype(str).to_numpy(dtype=object)


def stage_labels(cfg, params: Dict[str, Any], stage: str):
    """The label set a stage works with: its own ``labels`` override (``{column, file,
    file_column, groups, group_map}``, the keys of ``dataset.labels``; a relative ``file`` /
    ``groups`` resolves against the configuration file) when given, else ``dataset.labels``
    (``None`` without either). The manuscript's Xenium niches, interfaces and screen use the
    marker-derived labels of a per-cell CSV while its cell-level numbers use the dataset's
    ``cell_type`` column, so the spatial stages take the override."""
    from interpretfm.config import ConfigError, parse_labels

    raw = params.get("labels")
    if not isinstance(raw, dict) or not any(k in raw for k in ("column", "file", "groups", "group_map")):
        return cfg.dataset.labels
    errors: List[str] = []
    keys = {k: v for k, v in raw.items() if k in ("column", "file", "file_column", "groups", "group_map")}
    base = Path(cfg.source).parent if cfg.source is not None else Path.cwd()
    lab = parse_labels(keys, base, f"stages.{stage}.labels", errors)
    if errors or lab is None:
        raise ConfigError(errors or [f"stages.{stage}.labels: column is required"])
    return lab


def label_group_map(ds) -> Dict[str, str]:
    """The label → group mapping of ``dataset.labels`` or of a :class:`~interpretfm.config.Labels`
    (``groups`` CSV merged with ``group_map``; empty without either)."""
    from interpretfm.config import Labels
    from interpretfm.interfaces import load_group_map, norm_label

    lab = ds if isinstance(ds, Labels) or ds is None else ds.labels
    if lab is None:
        return {}
    out: Dict[str, str] = {}
    if lab.groups is not None:
        out.update(load_group_map(lab.groups))
    out.update({norm_label(k): norm_label(v) for k, v in (lab.group_map or {}).items()})
    return out


# --- counts -------------------------------------------------------------------------------------


def counts_matrix(adata, counts_layer: Optional[str]):
    """The matrix declared to hold raw counts (``X`` or ``adata.layers[counts_layer]``)."""
    if counts_layer is None:
        return adata.X
    if counts_layer not in adata.layers:
        raise IngestError(f"dataset.counts_layer: '{counts_layer}' not in adata.layers (keys: {list(adata.layers)})")
    return adata.layers[counts_layer]


def check_raw_counts(X, where: str = "dataset.counts_layer", max_cells: int = 5000) -> Dict[str, Any]:
    """Check that a matrix holds raw counts: finite, non-negative and integer-valued.

    The integer check runs on the first ``max_cells`` cells (the dense part is cheap there);
    negative or non-finite values anywhere are an error. Returns a short summary.
    """
    from scipy import sparse

    n = int(X.shape[0])
    head = X[: min(n, max_cells)]
    vals = head.data if sparse.issparse(head) else np.asarray(head)
    vals = np.asarray(vals).ravel()
    if vals.size and not np.isfinite(vals).all():
        raise IngestError(f"{where}: the count matrix holds NaN or infinite values")
    if vals.size and vals.min() < 0:
        raise IngestError(f"{where}: the count matrix holds negative values; raw counts are required "
                          "(point counts_layer at the raw layer, or null when X holds counts)")
    frac_non_integer = float(np.mean(np.abs(vals - np.round(vals)) > 1e-6)) if vals.size else 0.0
    if frac_non_integer > 0:
        raise IngestError(f"{where}: {100 * frac_non_integer:.1f}% of the first {head.shape[0]} cells' values are not "
                          "integers; raw counts are required (normalized or log-transformed data were found). "
                          "Set dataset.counts_layer to the layer holding raw counts, or null when X holds them")
    return {"checked_cells": int(head.shape[0]), "max_value": float(vals.max()) if vals.size else 0.0,
            "dtype": str(np.asarray(vals).dtype), "sparse": bool(sparse.issparse(X))}


def per_cell_counts(X) -> Tuple[np.ndarray, np.ndarray]:
    """``(transcripts per cell, detected genes per cell)``."""
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int64)
    return totals, n_genes


def qc_flags(totals: np.ndarray, n_genes: np.ndarray, min_transcripts: int, min_genes: int) -> np.ndarray:
    """Boolean mask of cells passing both thresholds (``False`` = flagged as low quality)."""
    return (totals >= min_transcripts) & (n_genes >= min_genes)


# --- genes --------------------------------------------------------------------------------------


_ENSEMBL_PREFIXES = ("ENSG", "ENSMUSG", "ENSRNOG", "ENSDARG", "ENSMMUG")


def check_gene_names(genes: GeneTable) -> List[str]:
    """Warnings about the gene names (empty, duplicated, or not matching ``gene_id.kind``)."""
    warnings: List[str] = []
    keys = genes.keys
    empty = sum(1 for k in keys if not k or k in ("nan", "None"))
    if empty:
        raise IngestError(f"dataset.gene_id: {empty} gene names are empty")
    dup = len(keys) - len(set(keys))
    if dup:
        raise IngestError(f"dataset.gene_id: {dup} duplicated gene names (gene names must be unique)")
    looks_ensembl = sum(1 for k in keys if k.upper().startswith(_ENSEMBL_PREFIXES))
    if genes.kind == "symbol" and looks_ensembl > 0.5 * len(keys):
        warnings.append(f"dataset.gene_id.kind is 'symbol' but {looks_ensembl}/{len(keys)} names look like Ensembl ids")
    if genes.kind == "ensembl" and looks_ensembl < 0.5 * len(keys):
        warnings.append(f"dataset.gene_id.kind is 'ensembl' but only {looks_ensembl}/{len(keys)} names look like Ensembl ids")
    return warnings


# --- vocabulary coverage ------------------------------------------------------------------------


def find_vocab(cfg) -> Tuple[Optional[Path], str]:
    """The model's gene vocabulary file for coverage and where it was found.

    Order: ``model.vocab``; ``<model.checkpoint>/vocab.json``; ``stages.<extract | concept_eval |
    gene_readout>.token_vocab``; ``<stages.extract.store_root>/token_vocab.json``. Returns
    ``(None, reason)`` when none exists.
    """
    from interpretfm.params import param_path

    if cfg.model.vocab is not None:
        return Path(cfg.model.vocab), "model.vocab"
    ckpt = cfg.model.checkpoint
    if ckpt is not None and (Path(ckpt) / "vocab.json").is_file():
        return Path(ckpt) / "vocab.json", "model.checkpoint/vocab.json"
    for stage in ("extract", "concept_eval", "gene_readout"):
        given = cfg.stage_params(stage).get("token_vocab")
        if given:
            return param_path(cfg, given, stage, "token_vocab"), f"stages.{stage}.token_vocab"
    root = cfg.stage_params("extract").get("store_root")
    if root:
        p = param_path(cfg, root, "extract", "store_root", must_exist=False) / "token_vocab.json"
        if p.is_file():
            return p, "stages.extract.store_root/token_vocab.json"
    return None, "no vocabulary given (set model.vocab to the model's gene vocabulary JSON)"


def vocab_names(path: Union[str, Path]) -> List[str]:
    """Gene names of a vocabulary file: a JSON object (``{name: id}`` or ``{id: name}``) or a JSON list."""
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return list(load_token_vocab(path).values())


def vocab_coverage(genes: GeneTable, names: Sequence[str]):
    """Per gene: whether any of its names (key, symbol, Ensembl id) is in the vocabulary.

    Returns ``(DataFrame[key, symbol, ensembl, in_vocab, matched_name], summary)``.
    """
    import pandas as pd

    vocab = {str(n) for n in names}
    upper = {n.upper(): n for n in vocab}
    rows = []
    for key, sym, ens in zip(genes.keys, genes.symbols, genes.ensembl):
        match = None
        for cand in (key, sym, ens):
            if cand and cand in vocab:
                match = cand
                break
            if cand and cand.upper() in upper:
                match = upper[cand.upper()]
                break
        rows.append({"key": key, "symbol": sym, "ensembl": ens, "in_vocab": match is not None, "matched_name": match})
    df = pd.DataFrame(rows)
    n_in = int(df["in_vocab"].sum())
    summary = {"n_genes": int(len(df)), "n_in_vocab": n_in, "fraction": (n_in / len(df)) if len(df) else None,
               "n_missing": int(len(df) - n_in), "vocab_size": len(vocab),
               "missing_examples": df.loc[~df["in_vocab"], "key"].head(20).tolist()}
    return df, summary


__all__ = ["IngestError", "check_gene_names", "check_raw_counts", "counts_matrix", "find_vocab", "per_cell_counts",
           "qc_flags", "vocab_coverage", "vocab_names"]
