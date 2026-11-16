"""Stage ``extract``: token-level activations of the configured layers into the sharded store.

Two modes:

- **cached store** (``stages.extract.store_root``): an existing activation store (the 0.9
  sharded format, ``<root>/activations/<layer>/shard_*/``) is checked for every configured
  layer and registered in ``store.json``; no model code runs and the stage stays in the core
  environment.
- **fresh extraction** (no ``store_root``): the adapter of ``model.name``
  (:mod:`interpretfm.models`) loads the weights from ``model.checkpoint`` and writes the store
  under ``02_extract/store/``. The runner starts this mode in the model's environment
  (``model.env``) as a child process ``python -m interpretfm.cli extract --config
  <run_dir>/config.resolved.yaml --run-dir <run_dir> --model <name> --layers ...``; with
  ``env: {}`` it runs in the current interpreter. The child writes one ``<layer>/manifest.json``
  per layer (rows, cells, tokens, dtype, the adapter's summary, the interpreter and the
  installed packages with the torch/CUDA versions) and ``store.json``; the parent reads only
  those files.

Parameters (``stages.extract``): ``batch_size`` (8 cells per forward pass), ``max_length``
(512 tokens per cell, scGPT), ``dtype`` (``float32``; ``float16`` halves the store),
``rows_per_shard`` (200,000), ``device`` (``cuda``, ``cuda:<n>`` or ``cpu``; a missing GPU is an
error unless ``cpu`` is set), ``cell_subset`` (a number of cells drawn with ``run.seed``, or a
file of cell ids; the ids used are written to ``cell_subset.txt``), ``gene_keys``
(``{symbol_column, ensembl_column}`` of ``var``; default: the ``concepts`` stage's keys or
common column names) and one mapping per model with model-specific options
(``scgpt: {}``, ``geneformer: {model_version: V2, nproc: 1}``, ``toy: {hidden_dim: 16}``),
``weights`` (``trained``; ``random_table`` or ``random_all`` re-initialize scGPT or Geneformer
before the extraction, :mod:`interpretfm.models.reinit`) and ``weights_seed`` (0). With a cached
store, ``weights`` states what the store holds and is recorded in ``store.json``.

Cell-level pooling is declared once, in ``model.cell_pooling`` (``token`` with the token id of
the classification row, or ``mean`` over gene rows); the stage records it in ``store.json``
and checks that every extracted cell has the pooling token.
"""
from __future__ import annotations

import shutil
from typing import Any, Dict, Optional

from interpretfm.params import get_mapping, param_path
from interpretfm.runner import StageContext, register_stage
from interpretfm.stores import layer_root, list_shards, write_store_spec

OUTPUTS = ['store.json']
STORE_DIR = "store"
WORK_DIR = "work"


def _needs_model_env(cfg) -> bool:
    return not cfg.stage_params("extract").get("store_root")


@register_stage("extract", inputs=["ingest"], outputs=OUTPUTS, env="model", needs_model_env=_needs_model_env)
def run_extract(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Token-level activation extraction into the sharded store (fresh, in the model environment, or a cached store)."""
    store_root = ctx.params.get("store_root")
    if store_root:
        from interpretfm.models.reinit import check_weights

        weights = ctx.params.get("weights", "trained")
        check_weights(ctx.config.model.name, weights)  # describes the cached store; checked as for a fresh extraction
        root = param_path(ctx.config, store_root, "extract", "store_root")
        n_shards = {}
        for layer in ctx.config.model.layers:
            if not layer_root(root, layer).is_dir():
                raise FileNotFoundError(f"stages.extract.store_root: layer {layer!r} not found under {root}/activations")
            n_shards[layer] = len(list_shards(root, layer))
        write_store_spec(ctx.stage_dir, root, ctx.config.model.layers, {"source": "cached", "n_shards": n_shards,
                                                                        "weights": weights})
        ctx.log.info("extract: using the cached activation store %s (%s, weights %s)", root, n_shards, weights)
        return {"source": "cached", "store_root": str(root), "n_shards": n_shards, "weights": weights}
    return extract_fresh(ctx)


def extract_fresh(ctx: StageContext) -> Dict[str, Any]:
    import anndata as ad

    from interpretfm.envs import child_environment
    from interpretfm.extract import layer_manifest, select_cells, spec_from_config
    from interpretfm.genes import GeneTable
    from interpretfm.manifest import write_json
    from interpretfm.models import get_extractor
    from interpretfm.models.base import ExtractionError

    cfg = ctx.config
    spec = spec_from_config(cfg)
    extractor = get_extractor(cfg.model.name)
    ds = cfg.dataset
    ctx.log.info("extract: %s layers %s from %s (device %s, batch %d, dtype %s)", cfg.model.name, spec.layers, ds.h5ad,
                 spec.device, spec.batch_size, spec.dtype)
    adata = ad.read_h5ad(ds.h5ad)
    if ds.counts_layer:
        if ds.counts_layer not in adata.layers:
            raise ExtractionError(f"dataset.counts_layer: {ds.counts_layer!r} not in adata.layers")
        adata.X = adata.layers[ds.counts_layer]
    gk = get_mapping(ctx.params, "gene_keys", "extract") or (cfg.stage_params("concepts").get("gene_keys") or {})
    genes = GeneTable.from_adata(adata, ds.gene_id, symbol_column=gk.get("symbol_column"), ensembl_column=gk.get("ensembl_column"))
    check = extractor.check(spec, genes)  # weights, layers and vocabulary, before anything heavy loads
    if extractor.uses_device and spec.device.startswith("cuda"):
        import os

        import torch

        if not torch.cuda.is_available():
            raise ExtractionError("stages.extract.device: cuda requested but no GPU is visible to this process "
                                  f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')!r}, torch "
                                  f"{torch.__version__}); set device: cpu to extract on the CPU")
    extractor.prepare(adata, spec, ctx.log)  # dataset-level state (C2S-Scale shard positions) before the subset
    idx, subset = select_cells([str(o) for o in adata.obs_names], ctx.params.get("cell_subset"), spec.seed)
    if idx is not None:
        adata = adata[idx].copy()
        ctx.path("cell_subset.txt").write_text("\n".join(str(o) for o in adata.obs_names) + "\n")
        ctx.log.info("extract: cell subset %s -> %d cells (ids in cell_subset.txt)", subset, adata.n_obs)
    store_root = ctx.path(STORE_DIR, "activations").parent
    for layer in spec.layers:  # a failed earlier attempt leaves a partial layer; start it over
        if layer_root(store_root, layer).exists():
            shutil.rmtree(layer_root(store_root, layer))
    work_dir = ctx.stage_dir / WORK_DIR
    summary = extractor.extract(adata, genes, spec, store_root, work_dir, ctx.log)
    env = child_environment()
    layers_info: Dict[str, Any] = {}
    n_shards: Dict[str, int] = {}
    for layer in spec.layers:
        m = layer_manifest(store_root, layer, spec.pooling_token if spec.pooling_kind == "token" else None, spec.special_tokens)
        if m["n_rows"] == 0:
            raise ExtractionError(f"extract: no rows were written for layer {layer!r}")
        if spec.pooling_kind == "token" and m["n_cells_with_pooling_token"] != m["n_cells"]:
            ctx.log.warning("extract: %s: pooling token %r found for %d of %d cells; check model.cell_pooling.token",
                            layer, spec.pooling_token, m["n_cells_with_pooling_token"], m["n_cells"])
        m.update({"model": cfg.model.name, "checkpoint": None if spec.checkpoint is None else str(spec.checkpoint),
                  "cell_pooling": {"kind": spec.pooling_kind, "token": spec.pooling_token}, "special_tokens": spec.special_tokens,
                  "extraction": spec.to_dict(), "cell_subset": subset, "adapter": summary, "check": check, "environment": env})
        write_json(ctx.path(layer, "manifest.json"), m)
        layers_info[layer] = {k: m[k] for k in ("n_shards", "n_rows", "n_cells", "hidden_dim", "dtype")}
        n_shards[layer] = m["n_shards"]
        ctx.log.info("extract: %s: %d rows, %d cells, %d shards, H=%s, %s", layer, m["n_rows"], m["n_cells"], m["n_shards"],
                     m["hidden_dim"], m["dtype"])
    write_store_spec(ctx.stage_dir, store_root, spec.layers, {
        "source": "extracted", "model": cfg.model.name, "n_shards": n_shards, "n_cells": int(adata.n_obs),
        "cell_pooling": {"kind": spec.pooling_kind, "token": spec.pooling_token}, "special_tokens": spec.special_tokens,
        "dtype": spec.dtype, "cell_subset": subset})
    return {"source": "extracted", "store_root": str(store_root), "layers": layers_info, "cell_subset": subset,
            "torch": env.get("torch"), "python": env.get("python")}
