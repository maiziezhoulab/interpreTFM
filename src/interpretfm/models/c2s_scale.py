"""C2S-Scale adapter: a Hugging Face causal language model (Gemma-2) reading cell sentences.

A cell sentence is the cell's detected genes (count > 0) ordered by descending count, the
first ``max_genes`` of them as space-separated upper-case symbols. Ties are broken by a
shuffle seeded with ``tie_seed`` plus the cell's position within its extraction shard, the
rule of the vendored cell2sentence code (:func:`interp_pipeline.vendor.c2s_local.util.generate_sentences`),
so the sentence of a cell depends on its shard and position in the **whole** dataset, not on
the cell subset being extracted (:meth:`C2SScaleExtractor.prepare`). Per-cell normalisation
is monotonic within a cell and would not change the order; counts are ranked as they are.

The sentence is embedded in the cell2sentence cell-type-prediction prompt (the first template
of ``prompts/single_cell_cell_type_prediction_prompts.json``) with ``num_genes`` =
``max_genes`` and the organism (``obs["organism"]`` when the dataset has it, else ``human``);
the model sees the prompt tokens before the gene tokens, so the template is part of the
representation. The stored rows of ``layer_<i>`` are, for every gene of the sentence, the
output of decoder block *i* (``model.layers[i]``) at the gene's last sub-token
(``token_pooling: last``; ``mean`` and ``max`` over the sub-tokens are available). Rows are
pooled in float32 and stored in ``stages.extract.dtype``. The model runs in evaluation mode
under ``torch.no_grad`` in ``model_dtype`` (float32 by default, as for the manuscript's
stores); ``batch_size`` cells per forward pass, left-padded.

Shards: with ``shard_column`` (an ``obs`` column naming each cell's shard, ``shard_<k>`` or
any label) the store shards follow the column and cells keep their dataset order within a
shard; without it, consecutive blocks of ``cells_per_shard`` cells (derived from
``rows_per_shard`` and the median number of detected genes). Options
(``stages.extract.c2s_scale``): ``max_genes`` (256), ``token_pooling`` (``last``),
``model_dtype`` (``float32``, ``bfloat16``, ``float16``), ``tie_seed`` (42),
``shard_column``, ``cells_per_shard``, ``organism``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from interpretfm.genes import GeneTable
from interpretfm.models.base import ExtractionError, ExtractionSpec, ModelExtractor, layer_index

TOKEN_POOLINGS = ("last", "mean", "max")
MODEL_DTYPES = ("float32", "bfloat16", "float16")
PROMPTS_FILE = "single_cell_cell_type_prediction_prompts.json"
DEFAULT_ORGANISM = "human"  #: what the vendored code uses when the dataset has no ``organism`` column


def prompt_template() -> str:
    """The cell2sentence cell-type-prediction model input (template 0), from the vendored prompts."""
    import interp_pipeline.vendor.c2s_local as vendored

    path = Path(vendored.__file__).parent / "prompts" / PROMPTS_FILE
    with open(path) as fh:
        return str(json.load(fh)["model_input"][0])


def rank_genes(cols: np.ndarray, vals: np.ndarray, seed: int) -> np.ndarray:
    """Column indices of one cell's detected genes by descending value; ties shuffled with ``seed``
    (``sklearn.utils.shuffle`` then a stable argsort, the cell2sentence rule)."""
    from sklearn.utils import shuffle

    if cols.size == 0:
        return cols
    cols, vals = shuffle(np.asarray(cols), np.asarray(vals), random_state=int(seed))
    return cols[np.argsort(-vals, kind="stable")]


def cell_sentences(X, names: Sequence[str], seeds: Sequence[int], max_genes: Optional[int] = None) -> List[List[str]]:
    """The ranked gene names of every row of ``X`` (sparse or dense counts over ``names``), each
    row's ties broken with its own seed, truncated to ``max_genes``."""
    import scipy.sparse as sp

    X = sp.csr_matrix(X)
    X.eliminate_zeros()
    names = np.asarray(names, dtype=object)
    out = []
    for i in range(X.shape[0]):
        cols = X.indices[X.indptr[i]:X.indptr[i + 1]]
        vals = X.data[X.indptr[i]:X.indptr[i + 1]]
        order = rank_genes(cols, vals, seeds[i])
        genes = [str(names[c]) for c in order]
        out.append(genes[:max_genes] if max_genes else genes)
    return out


def render_prompt(template: str, genes: Sequence[str], max_genes: int, organism: str) -> str:
    return template.format(num_genes=str(max_genes), organism=organism, cell_sentence=" ".join(genes))


def shard_positions(shard_labels: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, Dict[int, Any]]:
    """``(shard id, position within shard)`` for every cell, in dataset order.

    Labels ``shard_<k>`` (or integers) keep ``k``; other labels are numbered in order of first
    appearance. Returns the ids, the positions and ``{id: label}``.
    """
    labels = [str(s) for s in shard_labels]
    ids_of: Dict[str, int] = {}
    numbered = all(re.fullmatch(r"shard_(\d+)", s) or s.isdigit() for s in labels)
    for s in labels:
        if s not in ids_of:
            ids_of[s] = int(re.sub(r"^shard_", "", s)) if numbered else len(ids_of)
    ids = np.array([ids_of[s] for s in labels], dtype=np.int64)
    pos = np.zeros(len(labels), dtype=np.int64)
    seen: Dict[int, int] = {}
    for i, k in enumerate(ids):
        pos[i] = seen.get(int(k), 0)
        seen[int(k)] = pos[i] + 1
    return ids, pos, {v: k for k, v in ids_of.items()}


class C2SScaleExtractor(ModelExtractor):
    name = "c2s_scale"
    gene_kind = "symbol"

    def __init__(self) -> None:
        self._positions: Optional[Dict[str, Tuple[int, int]]] = None  #: cell id -> (shard id, position), from prepare()
        self._shard_labels: Dict[int, Any] = {}

    # --- options ------------------------------------------------------------------------------------

    @staticmethod
    def _options(spec: ExtractionSpec) -> Dict[str, Any]:
        o = spec.options or {}
        max_genes = int(o.get("max_genes", 256))
        pooling = str(o.get("token_pooling", "last"))
        dtype = str(o.get("model_dtype", "float32"))
        if max_genes <= 0:
            raise ExtractionError("stages.extract.c2s_scale.max_genes must be > 0")
        if pooling not in TOKEN_POOLINGS:
            raise ExtractionError(f"stages.extract.c2s_scale.token_pooling: expected one of {list(TOKEN_POOLINGS)}, got {pooling!r}")
        if dtype not in MODEL_DTYPES:
            raise ExtractionError(f"stages.extract.c2s_scale.model_dtype: expected one of {list(MODEL_DTYPES)}, got {dtype!r}")
        return {"max_genes": max_genes, "token_pooling": pooling, "model_dtype": dtype, "tie_seed": int(o.get("tie_seed", 42)),
                "shard_column": o.get("shard_column"), "cells_per_shard": o.get("cells_per_shard"), "organism": o.get("organism")}

    # --- check and prepare ----------------------------------------------------------------------------

    def check(self, spec: ExtractionSpec, genes: GeneTable) -> Dict[str, Any]:
        info = super().check(spec, genes)
        opts = self._options(spec)
        ckpt = Path(spec.checkpoint)
        if not (ckpt / "config.json").is_file():
            raise ExtractionError(f"model.checkpoint: {ckpt} must be a Hugging Face model folder with config.json "
                                  "(a local copy of vandijklab/C2S-Scale-Gemma-2-2B)")
        with open(ckpt / "config.json") as fh:
            mcfg = json.load(fh)
        n_layers = int(mcfg.get("num_hidden_layers", 0))
        for layer in spec.layers:
            if "." in layer:
                raise ExtractionError(f"model.layers: {layer!r}; c2s_scale layers are named layer_<i> (decoder block outputs)")
            layer_index(layer, n_layers, model="c2s_scale")
        names = self.gene_names(genes)
        n_named = int(sum(n is not None for n in names))
        if n_named == 0:
            raise ExtractionError("no panel gene has a symbol for the cell sentences; check dataset.gene_id (symbols) or "
                                  "stages.extract.gene_keys.symbol_column")
        info.update({"model_type": mcfg.get("model_type"), "architectures": mcfg.get("architectures"), "n_layers": n_layers,
                     "hidden_size": int(mcfg.get("hidden_size", 0)), "checkpoint_dtype": mcfg.get("torch_dtype") or mcfg.get("dtype"),
                     "coverage": {"n_genes": len(names), "n_with_symbol": n_named, "n_missing": len(names) - n_named,
                                  "fraction": n_named / len(names) if names else None},
                     "options": opts})
        return info

    def prepare(self, adata, spec: ExtractionSpec, log) -> None:
        """Shard id and position of every cell of the whole dataset (the tie-breaking seeds)."""
        opts = self._options(spec)
        obs_names = [str(o) for o in adata.obs_names]
        if opts["shard_column"]:
            col = str(opts["shard_column"])
            if col not in adata.obs.columns:
                raise ExtractionError(f"stages.extract.c2s_scale.shard_column: {col!r} not in adata.obs")
            ids, pos, labels = shard_positions(adata.obs[col].tolist())
        else:
            cps = opts["cells_per_shard"]
            if not cps:
                import scipy.sparse as sp

                X = adata.X
                n_detected = np.asarray((X > 0).sum(axis=1)).ravel() if sp.issparse(X) else (np.asarray(X) > 0).sum(axis=1)
                per_cell = min(int(np.median(n_detected)) if len(n_detected) else 1, opts["max_genes"]) + 1
                cps = max(1, int(spec.rows_per_shard) // per_cell)
            ids = np.arange(adata.n_obs, dtype=np.int64) // int(cps)
            pos = np.arange(adata.n_obs, dtype=np.int64) % int(cps)
            labels = {int(k): f"shard_{int(k)}" for k in np.unique(ids)}
        self._positions = {c: (int(k), int(p)) for c, k, p in zip(obs_names, ids, pos)}
        self._shard_labels = labels
        log.info("c2s_scale: %d cells in %d shards (%s); tie-breaking seed %d + position within shard", adata.n_obs,
                 len(labels), f"column {opts['shard_column']!r}" if opts["shard_column"] else "consecutive blocks", opts["tie_seed"])

    # --- extraction ----------------------------------------------------------------------------------

    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path, log) -> Dict[str, Any]:
        import scipy.sparse as sp
        import torch

        from interp_pipeline.adapters.model_base import ModelSpec
        from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
        from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

        info = self.check(spec, genes)
        opts = info["options"]
        if self._positions is None:
            self.prepare(adata, spec, log)
        names = self.gene_names(genes)
        keep = np.array([n is not None for n in names], dtype=bool)
        if not keep.all():
            log.warning("c2s_scale: %d of %d panel genes have no symbol and are left out of the sentences", int((~keep).sum()), len(names))
        upper = [str(n).upper() for n, k in zip(names, keep) if k]
        if len(set(upper)) != len(upper):
            dup = sorted({u for u in upper if upper.count(u) > 1})
            log.warning("c2s_scale: %d symbols appear more than once in the panel (%s...)", len(dup), dup[:5])
        X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
        X = sp.csr_matrix(X)[:, np.flatnonzero(keep)]
        obs_names = [str(o) for o in adata.obs_names]
        missing = [c for c in obs_names if c not in self._positions]
        if missing:
            raise ExtractionError(f"c2s_scale: {len(missing)} cells were not seen by prepare() (e.g. {missing[:3]})")
        shard_of = np.array([self._positions[c][0] for c in obs_names], dtype=np.int64)
        seeds = [opts["tie_seed"] + self._positions[c][1] for c in obs_names]
        if opts["organism"]:
            organisms = [str(opts["organism"])] * adata.n_obs
        elif "organism" in adata.obs.columns:
            organisms = [str(o) for o in adata.obs["organism"]]
        else:
            organisms = [DEFAULT_ORGANISM] * adata.n_obs
        t0 = time.time()
        sentences = cell_sentences(X, upper, seeds, opts["max_genes"])
        template = prompt_template()
        prompts = [render_prompt(template, s, opts["max_genes"], o) for s, o in zip(sentences, organisms)]
        empty = np.array([len(s) == 0 for s in sentences], dtype=bool)
        if empty.any():
            log.warning("c2s_scale: %d cells have no detected gene and get no rows", int(empty.sum()))
        t1 = time.time()
        log.info("c2s_scale: %d cell sentences (max_genes %d, %.1f s); first prompt: %r", adata.n_obs, opts["max_genes"], t1 - t0,
                 prompts[0][:160] if prompts else "")

        if spec.device.startswith("cuda") and not torch.cuda.is_available():
            raise ExtractionError("stages.extract.device: cuda requested but no GPU is visible to this process")
        cache_dir = work_dir / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        adapter = C2SScaleAdapter()
        handle = adapter.load(ModelSpec(name="c2s_scale", checkpoint=str(spec.checkpoint), device=spec.device,
                                        options={"max_genes": opts["max_genes"], "cache_dir": str(cache_dir),
                                                 "torch_dtype": opts["model_dtype"]}))
        model_dtype = str(next(handle.model._model.parameters()).dtype).replace("torch.", "")
        n_layers = int(handle.model._model.config.num_hidden_layers)
        for layer in spec.layers:
            layer_index(layer, n_layers, model="c2s_scale")
        t2 = time.time()
        log.info("c2s_scale: model loaded from %s on %s (%d layers, %s, %.1f s)", spec.checkpoint, handle.device, n_layers,
                 model_dtype, t2 - t1)

        store = ActivationStore(ActivationStoreSpec(root=str(store_root)))
        save_dtype = "fp16" if spec.dtype == "float16" else "fp32"
        shards_written: Dict[str, List[int]] = {layer: [] for layer in spec.layers}
        n_rows = {layer: 0 for layer in spec.layers}
        n_batches = 0
        n_cells = 0
        order = np.flatnonzero(~empty)
        for shard_id in sorted({int(k) for k in shard_of[order]}):
            rows = [int(i) for i in order if shard_of[i] == shard_id]  # dataset order within the shard
            acts: Dict[str, List[torch.Tensor]] = {layer: [] for layer in spec.layers}
            ex: Dict[str, List[str]] = {layer: [] for layer in spec.layers}
            tok: Dict[str, List[str]] = {layer: [] for layer in spec.layers}
            batches = adapter.batches_from_prompts(handle, [obs_names[i] for i in rows], [prompts[i] for i in rows], spec.batch_size)
            with torch.no_grad():
                for batch in batches:
                    batch.update({"pooling": opts["token_pooling"], "save_dtype": save_dtype, "pool_dtype": "fp32"})
                    captured = adapter.forward_and_capture(model_handle=handle, batch=batch, layers=list(spec.layers), capture_cfg={})
                    processed = adapter.process_captured(captured=captured, batch=batch)
                    for layer, out in processed.items():
                        acts[layer].append(out["acts"])
                        ex[layer].extend(str(x) for x in out["ex"])
                        tok[layer].extend(str(x) for x in out["tok"])
                    n_batches += 1
                    del captured, processed
            n_cells += len(rows)
            for layer in spec.layers:
                if not acts[layer]:
                    continue
                A = torch.cat(acts[layer], dim=0)
                store.write_token_shard(layer=layer, shard_id=shard_id, acts=A, example_ids=ex[layer], token_ids=tok[layer],
                                        token_unit="gene", meta={"source_format": "c2s_scale_extract", "source_layer": layer,
                                                                 "shard_label": str(self._shard_labels.get(shard_id, shard_id)),
                                                                 "n_cells": len(rows), "model_dtype": model_dtype,
                                                                 "token_pooling": opts["token_pooling"], "max_genes": opts["max_genes"]})
                shards_written[layer].append(shard_id)
                n_rows[layer] += int(A.shape[0])
            log.info("c2s_scale: shard %d: %d cells, %d rows (%s), %.0f s elapsed", shard_id, len(rows), n_rows[spec.layers[0]],
                     spec.layers[0], time.time() - t2)
        info.update({"layers_written": [layer for layer in spec.layers if shards_written[layer]], "n_cells_extracted": int(n_cells),
                     "n_cells_without_genes": int(empty.sum()), "n_genes_used": int(keep.sum()), "n_batches": n_batches,
                     "shards": {layer: shards_written[layer] for layer in spec.layers},
                     "tokenization": {"sentence": "detected genes by descending count, ties shuffled with tie_seed + position within shard",
                                      "max_genes": opts["max_genes"], "tie_seed": opts["tie_seed"], "shard_column": opts["shard_column"],
                                      "prompt_template": template, "organisms": sorted(set(organisms)),
                                      "token_pooling": opts["token_pooling"], "pool_dtype": "float32", "padding_side": "left"},
                     "model_dtype": model_dtype, "n_layers": n_layers, "seconds_sentences": round(t1 - t0, 1),
                     "seconds_load": round(t2 - t1, 1), "seconds_forward": round(time.time() - t2, 1)})
        return info
