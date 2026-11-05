"""Reading the sharded activation store from pipeline stages.

The store format is the 0.9 one (:class:`interp_pipeline.io.activation_store.ActivationStore`):
``<root>/activations/<layer>/shard_<i>/{activations.pt,index.pt,metadata.json}``, where
``index.pt`` holds ``example_ids`` (one per row: the cell), ``token_ids`` (one per row: the
gene token or a special token) and ``token_unit``.

Stages find the store through :func:`resolve_store`: a stage parameter ``store_root`` (a
cached store from an earlier extraction, for example the one used for a manuscript) takes
precedence over the ``store.json`` written by the ``extract`` stage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from interpretfm.params import param_path

STORE_SPEC = "store.json"
SHARD_PREFIX = "shard_"


def write_store_spec(stage_dir: Union[str, Path], root: Union[str, Path], layers: Sequence[str],
                     extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<stage_dir>/store.json`` describing where the activation store lives."""
    spec = {"root": str(Path(root).resolve()), "format": "pt_shards", "layers": list(layers)}
    spec.update(extra or {})
    path = Path(stage_dir) / STORE_SPEC
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)
    return path


def read_store_spec(extract_dir: Union[str, Path]) -> Dict[str, Any]:
    path = Path(extract_dir) / STORE_SPEC
    if not path.exists():
        raise FileNotFoundError(f"no {STORE_SPEC} in {extract_dir}; run the extract stage or set store_root")
    with open(path) as fh:
        return json.load(fh)


def resolve_store(ctx, upstream: str = "extract") -> Path:
    """The activation store root for a stage: ``ctx.params['store_root']`` if set, else the
    root recorded in the ``extract`` stage's ``store.json``."""
    override = ctx.params.get("store_root")
    if override:
        root = param_path(ctx.config, override, ctx.stage.name, "store_root")
        if not (root / "activations").is_dir():
            raise FileNotFoundError(f"stages.{ctx.stage.name}.store_root: no 'activations/' under {root}")
        return root
    spec = read_store_spec(ctx.input_dir(upstream))
    return Path(spec["root"])


def layer_root(root: Union[str, Path], layer: str) -> Path:
    return Path(root) / "activations" / layer


def list_shards(root: Union[str, Path], layer: str) -> List[Path]:
    """Shard directories of a layer, in numeric order."""
    d = layer_root(root, layer)
    if not d.is_dir():
        raise FileNotFoundError(f"no activations for layer {layer!r} under {root}")
    shards = [p for p in d.iterdir() if p.is_dir() and p.name.startswith(SHARD_PREFIX)]

    def key(p: Path):
        tail = p.name[len(SHARD_PREFIX):]
        return (0, int(tail)) if tail.isdigit() else (1, tail)

    return sorted(shards, key=key)


def select_shards(shards: Sequence[Path], max_shards: Optional[int] = None, evenly_spaced: bool = False) -> List[Path]:
    """The first ``max_shards`` shards, or ``max_shards`` evenly spaced ones."""
    shards = list(shards)
    if max_shards is None or max_shards >= len(shards):
        return shards
    if evenly_spaced:
        pick = np.unique(np.linspace(0, len(shards) - 1, num=int(max_shards)).astype(int))
        return [shards[i] for i in pick]
    return shards[: int(max_shards)]


def torch_load(path: Union[str, Path]):
    """``torch.load`` of a trusted local file (full unpickling; stores and checkpoints hold
    plain tensors, lists and dicts)."""
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(str(path), map_location="cpu")


def load_shard(shard: Union[str, Path]):
    """``(activations[N, H] float32 tensor, index dict)`` of one shard."""
    shard = Path(shard)
    X = torch_load(shard / "activations.pt")
    idx = torch_load(shard / "index.pt")
    if not isinstance(idx, dict):
        raise RuntimeError(f"{shard}/index.pt: expected a dict")
    return X.float(), idx


def iter_shards(root: Union[str, Path], layer: str, max_shards: Optional[int] = None,
                evenly_spaced: bool = False) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
    """Yield ``(shard_name, activations, index)`` for the selected shards of a layer."""
    for shard in select_shards(list_shards(root, layer), max_shards, evenly_spaced):
        if not (shard / "activations.pt").exists() or not (shard / "index.pt").exists():
            continue
        X, idx = load_shard(shard)
        yield shard.name, X, idx


def hidden_dim(root: Union[str, Path], layer: str) -> int:
    for _name, X, _idx in iter_shards(root, layer, max_shards=1):
        return int(X.shape[1])
    raise RuntimeError(f"no readable shard for layer {layer!r} under {root}")


def sample_rows(root: Union[str, Path], layer: str, max_shards: int = 300, rows_per_shard: int = 512,
                gene_rows_only: bool = False, special_tokens: Sequence[str] = ()):
    """Evenly spaced rows across evenly spaced shards (SAE QC evaluation rows); a float32 tensor."""
    import torch

    special = {str(t) for t in special_tokens}
    xs = []
    for _name, X, idx in iter_shards(root, layer, max_shards=max_shards, evenly_spaced=True):
        if gene_rows_only and idx.get("token_ids") is not None:
            keep = np.array([str(t) not in special for t in idx["token_ids"]], dtype=bool)
            X = X[torch.as_tensor(np.flatnonzero(keep))]
        if X.shape[0] == 0:
            continue
        pick = np.unique(np.linspace(0, X.shape[0] - 1, num=min(int(rows_per_shard), int(X.shape[0]))).astype(int))
        xs.append(X[torch.as_tensor(pick)].clone())
    if not xs:
        raise RuntimeError(f"no rows sampled for layer {layer!r} under {root}")
    return torch.cat(xs)


def resolve_example_rows(example_ids: Sequence[Any], obs_names: Sequence[str]) -> np.ndarray:
    """Row index into ``obs_names`` for each example id (-1 when unmatched).

    Ids that are all digits are taken as row indices; anything else is matched to the
    observation names.
    """
    import pandas as pd

    ids = np.asarray([str(e) for e in example_ids], dtype=object)
    out = np.full(len(ids), -1, dtype=np.int64)
    if len(ids) == 0:
        return out
    numeric = np.array([s.isdigit() for s in ids], dtype=bool)
    n_obs = len(obs_names)
    if numeric.any():
        vals = ids[numeric].astype(np.int64)
        vals[(vals < 0) | (vals >= n_obs)] = -1
        out[numeric] = vals
    if (~numeric).any():
        index = pd.Index([str(o) for o in obs_names])
        out[~numeric] = index.get_indexer(ids[~numeric])
    return out


def cell_matrix(root: Union[str, Path], layer: str, obs_names: Sequence[str], pooling: str,
                token: Optional[str] = None, special_tokens: Sequence[str] = (),
                max_shards: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Cell-level activation matrix aligned to ``obs_names``.

    ``pooling="token"`` takes the row whose ``token_id`` equals ``token`` (the classification
    token); ``pooling="mean"`` averages a cell's rows excluding ``special_tokens``. Returns
    ``(A[n_obs, H] float32, has_row[n_obs] bool, summary)``.
    """
    import torch

    if pooling not in ("token", "mean"):
        raise ValueError(f"pooling must be 'token' or 'mean', got {pooling!r}")
    if pooling == "token" and token is None:
        raise ValueError("pooling='token' needs the token id")
    n_obs = len(obs_names)
    special = {str(t) for t in special_tokens}
    A: Optional[np.ndarray] = None
    counts = np.zeros(n_obs, dtype=np.int64)
    stats = {"n_shards": 0, "n_rows_seen": 0, "n_rows_used": 0, "n_unmatched": 0, "n_duplicate_assignments": 0}
    for _name, X, idx in iter_shards(root, layer, max_shards=max_shards):
        example_ids = idx.get("example_ids")
        token_ids = idx.get("token_ids")
        if example_ids is None or len(example_ids) != X.shape[0]:
            continue
        stats["n_shards"] += 1
        stats["n_rows_seen"] += int(X.shape[0])
        if A is None:
            A = np.zeros((n_obs, int(X.shape[1])), dtype=np.float32)
        if pooling == "token":
            if token_ids is None or len(token_ids) != X.shape[0]:
                continue
            use = np.array([str(t) == str(token) for t in token_ids], dtype=bool)
        elif token_ids is None:
            use = np.ones(X.shape[0], dtype=bool)
        else:
            use = np.array([str(t) not in special for t in token_ids], dtype=bool)
        rows = np.flatnonzero(use)
        if rows.size == 0:
            continue
        cells = resolve_example_rows([example_ids[r] for r in rows], obs_names)
        ok = cells >= 0
        stats["n_unmatched"] += int((~ok).sum())
        rows, cells = rows[ok], cells[ok]
        if rows.size == 0:
            continue
        Xr = X[torch.as_tensor(rows)].numpy().astype(np.float32)
        if pooling == "token":
            stats["n_duplicate_assignments"] += int((counts[cells] > 0).sum())
            A[cells] = Xr
            counts[cells] = 1
        else:
            np.add.at(A, cells, Xr)
            np.add.at(counts, cells, 1)
        stats["n_rows_used"] += int(rows.size)
    if A is None or not (counts > 0).any():
        raise RuntimeError(f"no cell rows aligned for layer {layer!r} under {root} (pooling={pooling}, token={token})")
    has_row = counts > 0
    if pooling == "mean":
        A[has_row] /= counts[has_row, None]
    summary = {"pooling": pooling, "token": None if token is None else str(token), "n_obs": int(n_obs),
               "n_has_row": int(has_row.sum()), "hidden_dim": int(A.shape[1]), **stats}
    return A, has_row, summary
