"""Extraction helpers shared by the ``extract`` stage and its checks (no model code here).

- :func:`spec_from_config` turns ``model`` + ``stages.extract`` into the adapter's
  :class:`~interpretfm.models.base.ExtractionSpec` (re-initialized weights,
  ``stages.extract.weights``, only for the models of :mod:`interpretfm.models.reinit`);
- :func:`select_cells` picks a seeded random subset or the cells listed in a file
  (``stages.extract.cell_subset``);
- :func:`layer_manifest` summarises what was written for one layer (rows, cells, tokens, dtype)
  and checks that the pooling token of ``model.cell_pooling`` is present;
- :func:`compare_stores` and :func:`compare_projections` compare a fresh store with another
  store of the same layer on shared cells (token order, per-token cosine, absolute and
  relative differences, and the cell-level projection through an SAE), the acceptance check of
  a fresh extraction against a cached one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from interpretfm.models.base import ExtractionSpec
from interpretfm.params import get_mapping, get_param
from interpretfm.stores import list_shards, torch_load

DEFAULTS = {"batch_size": 8, "max_length": 512, "dtype": "float32", "rows_per_shard": 200_000, "device": "cuda"}


def spec_from_config(cfg) -> ExtractionSpec:
    """The :class:`ExtractionSpec` of a configuration (``model`` and ``stages.extract``)."""
    p = cfg.stage_params("extract")
    name = cfg.model.name
    options = get_mapping(p, name, "extract")
    device = get_param(p, "device", DEFAULTS["device"], "extract", str)
    if not (device == "cpu" or device.startswith("cuda")):
        raise ValueError(f"stages.extract.device: expected 'cuda', 'cuda:<n>' or 'cpu', got {device!r}")
    from interpretfm.models.reinit import WEIGHTS, check_weights

    weights = get_param(p, "weights", "trained", "extract", str, choices=WEIGHTS)
    check_weights(name, weights)
    return ExtractionSpec(
        model=name, layers=list(cfg.model.layers), checkpoint=cfg.model.checkpoint, device=device,
        batch_size=get_param(p, "batch_size", DEFAULTS["batch_size"], "extract", int),
        max_length=get_param(p, "max_length", DEFAULTS["max_length"], "extract", int),
        dtype=get_param(p, "dtype", DEFAULTS["dtype"], "extract", str, choices=("float32", "float16")),
        rows_per_shard=get_param(p, "rows_per_shard", DEFAULTS["rows_per_shard"], "extract", int),
        pooling_kind=cfg.model.cell_pooling.kind, pooling_token=cfg.model.cell_pooling.token,
        special_tokens=[str(t) for t in cfg.model.special_tokens], seed=int(cfg.run.seed),
        weights=weights, weights_seed=get_param(p, "weights_seed", 0, "extract", int), options=options,
    )


def read_cell_list(path: Union[str, Path]) -> List[str]:
    """Cell ids from a text file (one per line, ``#`` comments) or a CSV with a ``cell`` column;
    either may be gzip-compressed (``.gz``)."""
    path = Path(path)
    if path.suffix.lower() == ".gz":
        import gzip

        with gzip.open(path, "rt") as fh:
            text = fh.read()
        path = path.with_suffix("")
    else:
        text = path.read_text()
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"stages.extract.cell_subset: {path} lists no cells")
    header = [h.strip() for h in lines[0].split(",")]
    if "cell" in header and (len(header) > 1 or path.suffix.lower() == ".csv"):
        col = header.index("cell")
        return [ln.split(",")[col].strip() for ln in lines[1:]]
    return lines


def select_cells(obs_names: Sequence[str], cell_subset: Any, seed: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Row indices of the cells to extract (``None`` = every cell) and a description.

    ``cell_subset`` is ``None``, a number of cells (a seeded draw without replacement, kept in
    dataset order) or the path of a file listing cell ids (unknown ids are an error).
    """
    n = len(obs_names)
    if cell_subset is None:
        return None, {"mode": "all", "n_cells": n}
    if isinstance(cell_subset, bool) or isinstance(cell_subset, float):
        raise ValueError(f"stages.extract.cell_subset: expected an integer or a file path, got {cell_subset!r}")
    if isinstance(cell_subset, int):
        if cell_subset <= 0:
            raise ValueError("stages.extract.cell_subset: must be > 0")
        if cell_subset >= n:
            return None, {"mode": "all", "n_cells": n, "requested": cell_subset}
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=cell_subset, replace=False))
        return idx, {"mode": "random", "n_cells": int(cell_subset), "seed": seed}
    path = Path(str(cell_subset))
    if not path.is_file():
        raise FileNotFoundError(f"stages.extract.cell_subset: file not found: {path}")
    wanted = read_cell_list(path)
    import pandas as pd

    pos = pd.Index([str(o) for o in obs_names]).get_indexer(wanted)
    missing = [w for w, p in zip(wanted, pos) if p < 0]
    if missing:
        raise ValueError(f"stages.extract.cell_subset: {len(missing)} ids of {path} are not in the dataset (e.g. {missing[:5]})")
    idx = np.unique(pos)
    return idx, {"mode": "file", "file": str(path), "n_cells": int(idx.size), "n_listed": len(wanted)}


# --- manifests ----------------------------------------------------------------------------------


def _load_index(shard: Path) -> Dict[str, Any]:
    idx = torch_load(shard / "index.pt")
    if not isinstance(idx, dict):
        raise RuntimeError(f"{shard}/index.pt: expected a dict")
    return idx


def _load_acts(shard: Path):
    import torch

    try:
        return torch.load(str(shard / "activations.pt"), map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        return torch_load(shard / "activations.pt")


def layer_manifest(store_root: Union[str, Path], layer: str, pooling_token: Optional[str] = None,
                   special_tokens: Sequence[str] = ()) -> Dict[str, Any]:
    """Row, cell and token counts of one layer of a store, from the shard metadata and indices."""
    shards = list_shards(store_root, layer)
    special = {str(t) for t in special_tokens}
    n_rows = n_special = n_pool = 0
    cells: set = set()
    tokens: set = set()
    hidden_dim = dtype = unit = None
    cells_with_pool: set = set()
    for shard in shards:
        meta_path = shard / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            hidden_dim = hidden_dim or meta.get("hidden_dim")
            dtype = dtype or meta.get("dtype")
        idx = _load_index(shard)
        ex = [str(e) for e in idx.get("example_ids", [])]
        tok = [str(t) for t in (idx.get("token_ids") or [])]
        unit = unit or idx.get("token_unit")
        n_rows += len(ex)
        cells.update(ex)
        tokens.update(tok)
        for e, t in zip(ex, tok):
            if t in special:
                n_special += 1
            if pooling_token is not None and t == str(pooling_token):
                n_pool += 1
                cells_with_pool.add(e)
    m = {"layer": layer, "n_shards": len(shards), "n_rows": n_rows, "n_cells": len(cells), "hidden_dim": hidden_dim,
         "dtype": dtype, "token_unit": unit, "n_unique_tokens": len(tokens), "n_rows_special": n_special,
         "n_rows_gene": n_rows - n_special, "pooling_token": pooling_token, "n_rows_pooling_token": n_pool,
         "n_cells_with_pooling_token": len(cells_with_pool)}
    return m


# --- comparisons ----------------------------------------------------------------------------------


def rows_by_cell(store_root: Union[str, Path], layer: str, cells: Optional[Iterable[str]] = None,
                 exclude_tokens: Sequence[str] = ()) -> Dict[str, Tuple[List[str], Any]]:
    """``{cell: (token ids, activations[n_tokens, H] float32)}`` for ``cells`` (``None``: every cell),
    in stored row order, skipping rows whose token is in ``exclude_tokens`` (e.g. padding)."""
    import torch

    wanted = None if cells is None else {str(c) for c in cells}
    excl = {str(t) for t in exclude_tokens}
    out: Dict[str, Tuple[List[str], Any]] = {}
    parts: Dict[str, List[Tuple[List[str], Any]]] = {}
    for shard in list_shards(store_root, layer):
        idx = _load_index(shard)
        ex = [str(e) for e in idx.get("example_ids", [])]
        tok = [str(t) for t in (idx.get("token_ids") or [""] * len(ex))]
        rows = [i for i, (e, t) in enumerate(zip(ex, tok)) if (wanted is None or e in wanted) and t not in excl]
        if not rows:
            continue
        X = _load_acts(shard)
        sel = X[torch.as_tensor(rows)].float().clone()
        by_cell: Dict[str, List[int]] = {}
        for j, i in enumerate(rows):
            by_cell.setdefault(ex[i], []).append(j)
        for cell, js in by_cell.items():
            parts.setdefault(cell, []).append(([tok[rows[j]] for j in js], sel[torch.as_tensor(js)]))
        if wanted is not None and len(parts) >= len(wanted) and all(c in parts for c in wanted):
            pass  # keep scanning: a cell may continue in the next shard
    for cell, chunks in parts.items():
        toks = [t for c in chunks for t in c[0]]
        acts = torch.cat([c[1] for c in chunks], 0) if len(chunks) > 1 else chunks[0][1]
        out[cell] = (toks, acts)
    return out


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not len(values):
        return {"min": None, "p01": None, "median": None, "max": None}
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "p01": float(np.quantile(v, 0.01)), "median": float(np.median(v)), "max": float(v.max())}


def compare_stores(new_root: Union[str, Path], ref_root: Union[str, Path], layer: str, cells: Optional[Iterable[str]] = None,
                   exclude_tokens: Sequence[str] = (), ref_exclude_tokens: Sequence[str] = ()) -> Dict[str, Any]:
    """Compare the rows of ``cells`` in two stores of the same layer.

    Per cell the token sequences must match (same ids, same order); on matching cells the
    per-token cosine, the absolute and the relative differences are summarised. Returns a
    JSON-serialisable report (``cells_compared``, ``token_sequence_identical``, ``cosine``,
    ``max_abs_diff``, ``max_rel_diff`` ...).
    """
    import torch

    new = rows_by_cell(new_root, layer, cells, exclude_tokens)
    ids = list(new) if cells is None else [str(c) for c in cells]
    ref = rows_by_cell(ref_root, layer, ids, ref_exclude_tokens or exclude_tokens)
    missing_new = [c for c in ids if c not in new]
    missing_ref = [c for c in ids if c not in ref]
    shared = [c for c in ids if c in new and c in ref]
    cos_all: List[float] = []
    abs_all: List[float] = []
    rel_all: List[float] = []
    identical_rows = 0
    n_rows = 0
    n_seq_ok = 0
    mismatches: List[Dict[str, Any]] = []
    for c in shared:
        t_new, a_new = new[c]
        t_ref, a_ref = ref[c]
        if t_new != t_ref:
            mismatches.append({"cell": c, "n_new": len(t_new), "n_ref": len(t_ref),
                               "first_difference": next((i for i, (x, y) in enumerate(zip(t_new, t_ref)) if x != y), min(len(t_new), len(t_ref)))})
            continue
        n_seq_ok += 1
        a, b = a_new, a_ref
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        d = (a - b).abs()
        cos_all.extend(cos.tolist())
        abs_all.append(float(d.max()))
        rel_all.extend((d.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)).tolist())
        identical_rows += int((d.max(dim=1).values == 0).sum())
        n_rows += int(a.shape[0])
    ref_scale = float(max((ref[c][1].abs().max().item() for c in shared), default=0.0))
    max_abs = float(max(abs_all)) if abs_all else None
    return {
        "layer": layer, "n_cells_requested": len(ids), "n_cells_compared": len(shared), "n_cells_token_sequence_identical": n_seq_ok,
        "n_cells_missing_in_new": len(missing_new), "n_cells_missing_in_ref": len(missing_ref),
        "token_sequence_identical": n_seq_ok == len(shared) and not missing_new and not missing_ref,
        "token_mismatch_examples": mismatches[:5], "n_rows_compared": n_rows, "n_rows_identical": identical_rows,
        "cosine": _summary(cos_all), "n_rows_cosine_below_0.999": int(sum(1 for v in cos_all if v < 0.999)),
        "max_abs_diff": max_abs, "ref_max_abs_value": ref_scale,
        "max_abs_diff_over_ref_scale": None if max_abs is None or ref_scale == 0 else max_abs / ref_scale,
        "row_relative_l2_diff": _summary(rel_all),
    }


def project_cells(rows: Dict[str, Tuple[List[str], Any]], encoder, special_tokens: Sequence[str] = (),
                  feature_idx: Optional[Sequence[int]] = None) -> Tuple[List[str], np.ndarray]:
    """Mean SAE latents over the gene-token rows of every cell (``rows`` from :func:`rows_by_cell`)."""
    import torch

    special = {str(t) for t in special_tokens}
    fidx = None if feature_idx is None else torch.as_tensor(np.asarray(feature_idx, dtype=np.int64))
    cells, mats = [], []
    for cell, (toks, acts) in rows.items():
        keep = torch.as_tensor([t not in special for t in toks], dtype=torch.bool)
        if not bool(keep.any()):
            continue
        z = encoder.encode(acts[keep]).cpu()
        if fidx is not None:
            z = z[:, fidx]
        cells.append(cell)
        mats.append(z.double().mean(dim=0).float().numpy())
    return cells, (np.stack(mats) if mats else np.zeros((0, 0), dtype=np.float32))


def compare_projections(new_root: Union[str, Path], ref_root: Union[str, Path], layer: str, checkpoint: Union[str, Path],
                        cells: Iterable[str], special_tokens: Sequence[str] = (), exclude_tokens: Sequence[str] = (),
                        ref_exclude_tokens: Sequence[str] = (), cells_h5ad: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Cell-level projected features (mean SAE latents over gene-token rows) from two stores, on
    ``cells``; optionally also against a ``cells.h5ad`` written by the ``cells`` stage (its
    ``var['feature']`` column selects the retained features)."""
    from interpretfm.sae import Encoder, load_checkpoint

    ids = [str(c) for c in cells]
    enc = Encoder(load_checkpoint(checkpoint), "cpu")
    new = rows_by_cell(new_root, layer, ids, exclude_tokens)
    ref = rows_by_cell(ref_root, layer, ids, ref_exclude_tokens or exclude_tokens)
    shared = [c for c in ids if c in new and c in ref]
    c_new, z_new = project_cells({c: new[c] for c in shared}, enc, special_tokens)
    c_ref, z_ref = project_cells({c: ref[c] for c in shared}, enc, special_tokens)
    assert c_new == c_ref
    out: Dict[str, Any] = {"layer": layer, "checkpoint": str(checkpoint), "n_cells": len(c_new), "n_features": int(z_new.shape[1]) if z_new.size else 0}
    out["store_vs_store"] = _vector_agreement(z_new, z_ref)
    if cells_h5ad is not None:
        import anndata as ad

        h = ad.read_h5ad(cells_h5ad)
        if "feature" not in h.var.columns:
            raise ValueError(f"{cells_h5ad}: var has no 'feature' column")
        fidx = h.var["feature"].astype(int).tolist()
        pos = h.obs_names.get_indexer(c_new)
        ok = pos >= 0
        X = np.asarray(h.X[pos[ok]], dtype=np.float32)
        _c, z_new_f = project_cells({c: new[c] for c in np.asarray(c_new)[ok]}, enc, special_tokens, feature_idx=fidx)
        _c, z_ref_f = project_cells({c: ref[c] for c in np.asarray(c_new)[ok]}, enc, special_tokens, feature_idx=fidx)
        out["cells_h5ad"] = {"path": str(cells_h5ad), "n_cells": int(ok.sum()), "n_features": len(fidx),
                             "new_store_vs_h5ad": _vector_agreement(z_new_f, X), "ref_store_vs_h5ad": _vector_agreement(z_ref_f, X)}
    return out


def _vector_agreement(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.size == 0 or a.shape != b.shape:
        return {"n": 0, "shape_new": list(a.shape), "shape_ref": list(b.shape)}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a64, axis=1), np.linalg.norm(b64, axis=1)
    both = (na > 0) & (nb > 0)
    cos = (a64[both] * b64[both]).sum(1) / (na[both] * nb[both])
    d = np.abs(a64 - b64)
    active_new, active_ref = a > 0, b > 0
    return {"n": int(a.shape[0]), "cosine": _summary(cos), "max_abs_diff": float(d.max()),
            "max_abs_diff_over_ref_scale": float(d.max() / max(np.abs(b64).max(), 1e-12)),
            "frac_cells_cosine_ge_0.999": float((cos >= 0.999).mean()) if cos.size else None,
            "active_set_jaccard_mean": float(np.mean([
                (x & y).sum() / max((x | y).sum(), 1) for x, y in zip(active_new, active_ref)])),
            "frac_entries_identical": float((d == 0).mean())}
