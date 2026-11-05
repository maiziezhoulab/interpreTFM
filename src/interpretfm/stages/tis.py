"""Stage ``tis``: Transcriptomic interpretability score per layer, with shuffled and PCA controls.

Parameters (``stages.tis``): ``trials`` (800), ``k_ref`` (32), ``quantiles`` ([0.25, 0.75]),
``judge`` (``log1p_cp10k``), ``controls`` (``[shuffle, pca]``), ``pca_of`` (``activations`` |
``expression``), ``seed`` (42), ``subsample_eval``, ``max_shards``, ``keep_cell_matrix`` (true)
and ``store_root`` (a cached activation store instead of the ``extract`` output).

Outputs: ``tis.json`` (per-layer summaries and the parameters), ``layers.csv`` (one row per
layer), and per layer ``<layer>/tis.json``, ``<layer>/tis_per_dim.csv`` and, when kept,
``<layer>/cell_activations.npy`` with ``<layer>/has_row.npy`` (the aligned cell-level matrix
that later stages can reuse).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from interpretfm.runner import StageContext, register_stage
from interpretfm.stores import cell_matrix, resolve_store
from interpretfm.tis import layer_table, params_dict, params_from_mapping, tis_for_matrix

OUTPUTS = ["tis.json", "layers.csv"]


@register_stage("tis", inputs=["extract"], outputs=OUTPUTS)
def run_tis(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Transcriptomic interpretability score per layer, with shuffled and PCA controls."""
    import anndata as ad
    import pandas as pd

    cfg = ctx.config
    params = params_from_mapping(ctx.params)
    store = resolve_store(ctx)
    adata = ad.read_h5ad(cfg.dataset.h5ad)
    if cfg.dataset.counts_layer:
        adata.X = adata.layers[cfg.dataset.counts_layer]
    pooling = cfg.model.cell_pooling.kind
    token = cfg.model.cell_pooling.token
    results = []
    for layer in cfg.model.layers:
        ctx.log.info("tis: layer %s (%s pooling, %d trials)", layer, pooling, params.trials)
        obs = [str(o) for o in adata.obs_names]
        A, has_row, extraction = cell_matrix(store, layer, obs, pooling, token=token,
                                             special_tokens=cfg.model.special_tokens, max_shards=params.max_shards)
        sub = adata if has_row.all() else adata[has_row]
        res = tis_for_matrix(A[has_row], sub, params, layer=layer, extraction=extraction)
        ldir = ctx.path(layer, "tis.json").parent
        ctx.write_json(f"{layer}/tis.json", res.summary())
        res.per_dim_frame().to_csv(ldir / "tis_per_dim.csv", index=False)
        if params.keep_cell_matrix:
            np.save(ldir / "cell_activations.npy", A.astype(np.float32))
            np.save(ldir / "has_row.npy", has_row)
        ctx.log.info("tis: %s mean %.4f shuffled %s gain %s", layer, res.summary()["model"]["mean"],
                     res.summary().get("shuffled", {}).get("mean"), res.gain)
        results.append(res)
    rows = layer_table(results)
    pd.DataFrame(rows).to_csv(ctx.path("layers.csv"), index=False)
    ctx.write_json("tis.json", {"params": params_dict(params), "pooling": {"kind": pooling, "token": token},
                                "layers": {r.layer: r.summary() for r in results}})
    return {"layers": {r["layer"]: {"tis_mean": r["tis_mean"], "tis_gain": r.get("tis_gain")} for r in rows}}
