"""Stage ``layers``: per-layer summary table with every criterion reported separately, and
per-criterion best-layer notes (no composite score).

Joins ``tis`` (TIS mean, shuffled and PCA controls, gain), ``sae`` (selected λ and its QC),
``concept_eval`` (concept F1 under the row and whole-gene splits for the SAE, raw and
untrained units), ``gene_readout`` (probe AUROC of the hidden state, the SAE features, the
random control and the best expression control) and ``cells`` (retained-feature count, GSI
excess and MHR for SAE features, raw dimensions and single genes).

Parameters (``stages.layers``): ``readout_scaling`` (``standardize`` | ``center``: which
gene-readout column to report).

Outputs: ``layer_table.csv`` (one row per layer), ``recommendations.json`` (per criterion the
best layer, its value, the baselines at that layer and the per-layer values) and ``notes.txt``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.layers import layer_table, notes_text, recommendations
from interpretfm.params import get_param
from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["layer_table.csv", "recommendations.json"]
UPSTREAM = ["tis", "sae", "concept_eval", "gene_readout", "cells"]


@register_stage("layers", inputs=UPSTREAM, outputs=OUTPUTS, uses_labels=True)
def run_layers(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Per-criterion layer summary and best-layer notes (no composite score)."""
    cfg = ctx.config
    scaling = get_param(ctx.params, "readout_scaling", "standardize", "layers", str, choices=("standardize", "center"))
    dirs = {name: ctx.input_dir(name) for name in UPSTREAM}
    table = layer_table(ctx.run_dir, cfg.model.layers, dirs, readout_scaling=scaling)
    table.to_csv(ctx.path("layer_table.csv"), index=False)
    rec = recommendations(table, cfg.model.representative_layer)
    rec["readout_scaling"] = scaling
    ctx.write_json("recommendations.json", rec)
    lines = notes_text(rec)
    ctx.path("notes.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        ctx.log.info("layers: %s", line)
    return {"n_layers": int(len(table)), "best_layer_counts": rec["best_layer_counts"]}
