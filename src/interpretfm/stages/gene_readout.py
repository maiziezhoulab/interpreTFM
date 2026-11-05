"""Stage ``gene_readout``: gene-level linear readout of concept membership on unseen genes.

Each measured gene is represented by its mean hidden state and its mean SAE activation
over the layer's tokens (:func:`interpretfm.gene_readout.gene_means`); the expression
controls come from the dataset (:func:`interpretfm.gene_readout.expression_representations`)
and random Gaussian vectors of the hidden dimension. Every representation is scored on the
same whole-gene splits and concepts (:func:`interpretfm.gene_readout.readout`).

Parameters (``stages.gene_readout``): ``n_splits`` (5), ``C`` (1.0), ``scaling``
([standardize, center]), ``controls`` ([random, coexpression, coexpression_pc50, celltype,
trivial]), ``concept_size`` ([5, 100]), ``split`` (``{valid: 0.2, test: 0.1}``), ``max_cells``
(100000), ``min_type_cells`` (50), ``n_pcs`` (50), ``seed`` (``run.seed``), ``max_iter``
(3000), ``dedup_cells`` (true), ``max_shards``, ``device``, ``token_vocab``, ``store_root``
and ``checkpoints`` (``{<layer>: path}``).

Outputs: ``gene_readout.json`` (mean +- SD over splits per representation), ``gene_readout.csv``
(per representation and split), ``per_concept.csv.gz``, ``splits.json``, ``expression_reps.npz``
and per layer ``<layer>/gene_means.npz`` (``keys``, ``count``, ``hidden``, ``sae``, ``untrained``).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from interpretfm.concepts import Universe, concept_size_mask
from interpretfm.gene_readout import (
    expression_representations,
    gene_means,
    make_splits,
    params_dict,
    params_from_mapping,
    readout,
    summarize_readout,
)
from interpretfm.genes import resolver_for
from interpretfm.runner import StageContext, register_stage
from interpretfm.sae import Encoder, load_checkpoint
from interpretfm.stages.concept_eval import selected_checkpoints
from interpretfm.stages.concepts import load_gene_table
from interpretfm.stores import resolve_store

OUTPUTS = ["gene_readout.json", "gene_readout.csv", "per_concept.csv.gz"]


@register_stage("gene_readout", inputs=["extract", "sae", "concepts"], outputs=OUTPUTS)
def run_gene_readout(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Gene-level linear readout of concept membership on unseen genes, with expression and random controls."""
    import anndata as ad
    import pandas as pd

    cfg = ctx.config
    params = params_from_mapping(ctx.params, default_seed=cfg.run.seed)
    store = resolve_store(ctx)
    universe = Universe.read(ctx.input_dir("concepts") / "universe")
    genes = load_gene_table(cfg, cfg.stage_params("concepts"))
    resolver = resolver_for(cfg, ctx.params, store, genes, stage="gene_readout")
    keys = [str(k) for k in universe.membership.index]
    if keys != list(genes.keys):
        raise RuntimeError("the concept universe is not in the dataset's gene order; rerun the concepts stage")
    device = params.resolved_device()

    # per-gene model representations, per layer
    observed = np.ones(len(keys), dtype=bool)
    layer_reps: Dict[str, Dict[str, np.ndarray]] = {}
    hidden_dims: Dict[str, int] = {}
    for layer in cfg.model.layers:
        ckpts = selected_checkpoints(ctx, layer)
        encoders = {name: Encoder(load_checkpoint(path), device) for name, path in ckpts.items() if path}
        means = gene_means(store, layer, resolver, keys, encoders, dedup_cells=params.dedup_cells,
                           max_shards=params.max_shards)
        np.savez_compressed(ctx.path(layer, "gene_means.npz"), keys=np.array(keys, dtype=object), **means)
        observed &= means["count"] > 0
        layer_reps[layer] = means
        hidden_dims[layer] = int(means["hidden"].shape[1])
        ctx.log.info("gene_readout: %s: %d genes observed, representations %s", layer, int((means["count"] > 0).sum()),
                     sorted(k for k in means if k != "count"))

    # expression controls
    adata = ad.read_h5ad(cfg.dataset.h5ad)
    if cfg.dataset.counts_layer:
        adata.X = adata.layers[cfg.dataset.counts_layer]
    labels = cfg.dataset.labels.column if cfg.dataset.labels is not None else None
    expr = expression_representations(adata, keys, labels, params, params.controls)
    np.savez_compressed(ctx.path("expression_reps.npz"), **{k: v for k, v in expr.items() if v is not None})

    # universe of genes and concepts
    sizes = universe.concept_sizes
    cmask = concept_size_mask(sizes.values, params.concept_size)
    term_ids = [str(c) for c in universe.membership.columns[cmask]]
    Y_all = universe.membership.values[:, cmask].astype(np.int8)
    gene_mask = observed & (Y_all.sum(1) >= 0)
    uni = [k for k, ok in zip(keys, gene_mask) if ok]
    gidx = {g: i for i, g in enumerate(uni)}
    rows = np.flatnonzero(gene_mask)
    Y = Y_all[rows]
    splits = make_splits(uni, params.n_splits, params.valid_frac, params.test_frac, params.seed)
    ctx.write_json("splits.json", {"universe": uni, "n_concepts": len(term_ids),
                                   "splits": {k: {"n_train_valid": len(tv), "n_test": len(te), "test": te}
                                              for k, (tv, te) in splits.items()}})
    ctx.log.info("gene_readout: universe %d genes, %d concepts of size %s, %d splits", len(uni), len(term_ids),
                 list(params.concept_size), params.n_splits)

    reps: Dict[str, Any] = {}
    for layer, means in layer_reps.items():
        reps[f"hidden|{layer}"] = (means["hidden"][rows], "raw")
        for name in ("sae", "untrained"):
            if name in means and means[name].shape[1] > 0:
                reps[f"{name}|{layer}"] = (means[name][rows], name)
    rng = np.random.default_rng(params.seed + 12345)
    if "random" in params.controls:
        for d in sorted(set(hidden_dims.values())):
            reps[f"random_{d}|-"] = (rng.standard_normal((len(uni), d)).astype(np.float32), f"random_{d}")
    for name in ("coexpression", "coexpression_pc50", "celltype", "trivial"):
        if name in params.controls and expr.get(name) is not None:
            reps[f"{name}|-"] = (np.asarray(expr[name])[rows], name)
        elif name in params.controls:
            ctx.log.warning("gene_readout: control %s unavailable (no labels or too few cells per type)", name)

    per_concept = readout(reps, Y, term_ids, sizes.values[cmask], gidx, splits, params, log=ctx.log)
    per_concept.to_csv(ctx.path("per_concept.csv.gz"), index=False)
    per_split, overall = summarize_readout(per_concept)
    per_split.to_csv(ctx.path("gene_readout.csv"), index=False)
    summary = {"params": params_dict(params), "n_genes": len(uni), "n_concepts": len(term_ids),
               "representations": {r["representation"]: {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                                         for k, v in r.items() if k != "representation"}
                                   for r in json.loads(overall.to_json(orient="records"))} if len(overall) else {}}
    ctx.write_json("gene_readout.json", summary)
    key = f"mean_probe_test_auroc_{params.scaling[0]}_mean"
    return {"n_genes": len(uni), "n_concepts": len(term_ids),
            "probe_auroc": {r: v.get(key) for r, v in summary["representations"].items()}}
