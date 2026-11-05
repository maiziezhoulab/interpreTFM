"""Stage ``ingest``: validate the input AnnData against the configuration and normalise units.

Minimal version: checks that the file loads, that the declared gene-id column, label column
and spatial ``obsm`` key exist, converts coordinates to µm in one place
(:func:`interpretfm.units.adata_coords_um`) and writes ``dataset_qc.json`` and, for spatial
data, ``coords_um.csv``. Vocabulary coverage, QC filtering and the cell table are added by
the ingest work package.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from interpretfm.runner import StageContext, register_stage
from interpretfm.units import adata_coords_um

OUTPUTS = ["dataset_qc.json"]


@register_stage("ingest", inputs=[], outputs=OUTPUTS)
def run_ingest(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Validate the dataset and convert spatial coordinates to micrometres."""
    import anndata as ad

    ds = ctx.config.dataset
    ctx.log.info("ingest: reading %s", ds.h5ad)
    adata = ad.read_h5ad(ds.h5ad)
    qc: Dict[str, Any] = {"h5ad": str(ds.h5ad), "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)}

    if ds.gene_id.column != "index" and ds.gene_id.column not in adata.var.columns:
        raise KeyError(f"dataset.gene_id.column: '{ds.gene_id.column}' not in adata.var (columns: {list(adata.var.columns)})")
    if ds.counts_layer is not None and ds.counts_layer not in adata.layers:
        raise KeyError(f"dataset.counts_layer: '{ds.counts_layer}' not in adata.layers (keys: {list(adata.layers)})")
    X = adata.layers[ds.counts_layer] if ds.counts_layer else adata.X
    totals = np.asarray(X.sum(axis=1)).ravel()
    n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    qc["transcripts_per_cell"] = {"median": float(np.median(totals)), "min": float(totals.min()), "max": float(totals.max())}
    qc["genes_per_cell"] = {"median": float(np.median(n_genes_per_cell))}
    qc["cells_passing_qc"] = int(((totals >= ds.qc.min_transcripts) & (n_genes_per_cell >= ds.qc.min_genes)).sum())

    if ds.labels is not None:
        if ds.labels.column not in adata.obs.columns:
            raise KeyError(f"dataset.labels.column: '{ds.labels.column}' not in adata.obs (columns: {list(adata.obs.columns)})")
        counts = adata.obs[ds.labels.column].astype(str).value_counts()
        qc["labels"] = {"column": ds.labels.column, "n_labels": int(len(counts)), "counts": {k: int(v) for k, v in counts.items()}}

    if ds.spatial is not None:
        coords_um = adata_coords_um(adata, ds.spatial)
        np.savetxt(ctx.path("coords_um.csv"), np.column_stack([np.arange(adata.n_obs), coords_um]),
                   delimiter=",", header="cell_index,x_um,y_um", comments="", fmt=["%d", "%.4f", "%.4f"])
        qc["spatial"] = {
            "obsm": ds.spatial.obsm,
            "um_per_unit": ds.spatial.um_per_unit,
            "extent_um": {"x": [float(coords_um[:, 0].min()), float(coords_um[:, 0].max())],
                          "y": [float(coords_um[:, 1].min()), float(coords_um[:, 1].max())]},
        }
    else:
        qc["spatial"] = None
    ctx.write_json("dataset_qc.json", qc)
    return {"n_cells": qc["n_cells"], "n_genes": qc["n_genes"], "spatial": ds.spatial is not None}
