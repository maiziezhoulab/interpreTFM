"""Stage ``ingest``: validate the input AnnData against the configuration, flag low-quality cells,
report gene coverage of the model vocabulary and normalise spatial units.

Checks (:mod:`interpretfm.ingest`): the matrix declared as counts holds raw counts (finite,
non-negative, integer-valued); gene names are unique and match ``dataset.gene_id.kind``; the
label column and the spatial ``obsm`` key exist when declared. Each failure is an
:class:`~interpretfm.ingest.IngestError` naming the configuration key.

Cells with fewer than ``dataset.qc.min_transcripts`` transcripts or ``dataset.qc.min_genes``
detected genes are flagged (``qc_pass = false`` in ``cells.csv``); nothing is dropped, and
later stages report how many flagged cells they used.

Gene coverage: the share of panel genes found in the model's gene vocabulary
(:func:`interpretfm.ingest.find_vocab`), written to ``vocab_coverage.csv`` and the manifest.
Coordinates are converted to micrometres once (:func:`interpretfm.units.adata_coords_um`) and
written to ``coords_um.csv`` and ``cells.csv``.

Labels come from the ``obs`` column or from ``dataset.labels.file`` (a CSV keyed by cell id,
:func:`interpretfm.ingest.dataset_labels`); with ``dataset.labels.groups`` / ``group_map`` the
cell table also carries the coarser ``group`` of every cell and the QC lists labels without a group.

Outputs: ``dataset_qc.json``, ``cells.csv`` (cell, n_transcripts, n_genes, qc_pass, label[, group],
x_um, y_um), ``vocab_coverage.csv`` (when a vocabulary was found) and ``coords_um.csv``
(spatial data).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from interpretfm.genes import GeneTable
from interpretfm.ingest import (
    IngestError,
    check_gene_names,
    check_raw_counts,
    counts_matrix,
    dataset_labels,
    find_vocab,
    label_group_map,
    per_cell_counts,
    qc_flags,
    vocab_coverage,
    vocab_names,
)
from interpretfm.interfaces import group_labels, norm_label
from interpretfm.runner import StageContext, register_stage
from interpretfm.units import adata_coords_um

OUTPUTS = ["dataset_qc.json", "cells.csv"]


@register_stage("ingest", inputs=[], outputs=OUTPUTS, uses_labels=True)
def run_ingest(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Validate the dataset, flag low-quality cells, report vocabulary coverage and convert coordinates to µm."""
    import anndata as ad
    import pandas as pd

    cfg = ctx.config
    ds = cfg.dataset
    ctx.log.info("ingest: reading %s", ds.h5ad)
    adata = ad.read_h5ad(ds.h5ad)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise IngestError(f"dataset.h5ad: empty matrix ({adata.n_obs} cells x {adata.n_vars} genes)")
    qc: Dict[str, Any] = {"h5ad": str(ds.h5ad), "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
                          "warnings": []}

    # counts
    X = counts_matrix(adata, ds.counts_layer)
    qc["counts"] = {"layer": ds.counts_layer or "X", **check_raw_counts(X, "dataset.counts_layer")}
    totals, n_genes_per_cell = per_cell_counts(X)
    passing = qc_flags(totals, n_genes_per_cell, ds.qc.min_transcripts, ds.qc.min_genes)
    qc["transcripts_per_cell"] = {"median": float(np.median(totals)), "min": float(totals.min()), "max": float(totals.max())}
    qc["genes_per_cell"] = {"median": float(np.median(n_genes_per_cell)), "min": int(n_genes_per_cell.min()),
                            "max": int(n_genes_per_cell.max())}
    qc["qc"] = {"min_transcripts": ds.qc.min_transcripts, "min_genes": ds.qc.min_genes,
                "n_pass": int(passing.sum()), "n_flagged": int((~passing).sum()),
                "fraction_flagged": float((~passing).mean()), "note": "flagged cells are kept; later stages report their use"}
    qc["cells_passing_qc"] = int(passing.sum())
    if not passing.any():
        qc["warnings"].append("every cell is flagged by dataset.qc; lower the thresholds or check counts_layer")

    # genes
    if ds.gene_id.column != "index" and ds.gene_id.column not in adata.var.columns:
        raise IngestError(f"dataset.gene_id.column: '{ds.gene_id.column}' not in adata.var (columns: {list(adata.var.columns)})")
    gk = cfg.stage_params("concepts").get("gene_keys") or {}
    genes = GeneTable.from_adata(adata, ds.gene_id, symbol_column=gk.get("symbol_column"),
                                 ensembl_column=gk.get("ensembl_column"))
    qc["warnings"].extend(check_gene_names(genes))
    qc["genes"] = {"column": ds.gene_id.column, "kind": ds.gene_id.kind, "n": len(genes.keys),
                   "n_with_symbol": int(sum(s is not None for s in genes.symbols)),
                   "n_with_ensembl": int(sum(e is not None for e in genes.ensembl)),
                   "n_never_detected": int((np.asarray((X > 0).sum(axis=0)).ravel() == 0).sum())}

    # vocabulary coverage of the model
    vocab_path, source = find_vocab(cfg)
    if vocab_path is not None:
        cov_df, cov = vocab_coverage(genes, vocab_names(vocab_path))
        cov_df.to_csv(ctx.path("vocab_coverage.csv"), index=False)
        cov.update({"model": cfg.model.name, "vocab": str(vocab_path), "source": source})
        ctx.log.info("ingest: %d/%d panel genes in the %s vocabulary (%s)", cov["n_in_vocab"], cov["n_genes"],
                     cfg.model.name, source)
        if cov["fraction"] is not None and cov["fraction"] < 0.5:
            qc["warnings"].append(f"only {100 * cov['fraction']:.0f}% of the panel genes are in the model vocabulary")
    else:
        cov = {"model": cfg.model.name, "vocab": None, "source": None, "fraction": None, "note": source}
        ctx.log.warning("ingest: vocabulary coverage not computed: %s", source)
    qc["vocab_coverage"] = cov

    # labels
    labels = dataset_labels(ds, adata)
    groups = None
    if labels is not None:
        counts = pd.Series(labels).value_counts()
        qc["labels"] = {"column": ds.labels.column, "file": str(ds.labels.file) if ds.labels.file else None,
                        "n_labels": int(len(counts)), "n_labels_ge_30_cells": int((counts >= 30).sum()),
                        "counts": {str(k): int(v) for k, v in counts.items()}}
        gmap = label_group_map(ds)
        if gmap:
            _exact, groups = group_labels(labels, gmap)
            gcounts = pd.Series(groups).value_counts()
            unmapped = sorted(set(norm_label(x) for x in counts.index) - set(gmap))
            qc["labels"]["groups"] = {"n_groups": int(len(gcounts)), "counts": {str(k): int(v) for k, v in gcounts.items()},
                                      "labels_without_group": unmapped}
            if unmapped:
                qc["warnings"].append(f"{len(unmapped)} label(s) are not in dataset.labels.groups/group_map and form their own group: {unmapped[:10]}")
    else:
        qc["labels"] = None
        qc["warnings"].append("no dataset.labels: population specificity and label-based summaries are skipped")

    # coordinates
    coords_um = None
    if ds.spatial is not None:
        try:
            coords_um = adata_coords_um(adata, ds.spatial)
        except (KeyError, ValueError) as e:
            raise IngestError(str(e)) from e
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

    # cell table
    cells = pd.DataFrame({"cell": [str(o) for o in adata.obs_names], "n_transcripts": totals,
                          "n_genes": n_genes_per_cell, "qc_pass": passing})
    cells["label"] = labels if labels is not None else ""
    if groups is not None:
        cells["group"] = groups
    if coords_um is not None:
        cells["x_um"], cells["y_um"] = coords_um[:, 0], coords_um[:, 1]
    cells.to_csv(ctx.path("cells.csv"), index=False)

    ctx.write_json("dataset_qc.json", qc)
    for w in qc["warnings"]:
        ctx.log.warning("ingest: %s", w)
    return {"n_cells": qc["n_cells"], "n_genes": qc["n_genes"], "n_flagged": qc["qc"]["n_flagged"],
            "vocab_coverage": cov.get("fraction"), "spatial": ds.spatial is not None}
