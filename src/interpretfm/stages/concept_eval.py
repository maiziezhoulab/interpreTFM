"""Stage ``concept_eval``: held-out concept alignment of SAE features, raw dimensions and the untrained SAE.

For every layer with a selected SAE, every representation (``sae``: the selected
checkpoint; ``raw``: each hidden dimension split into its positive and negative part;
``untrained``: the untrained SAE of the same width) and every held-out unit (``row``:
hash-based row split; ``gene``: whole-gene split) the stage runs
:func:`interpretfm.concept_eval.heldout_concept_f1` and summarizes it with
:func:`interpretfm.concept_eval.summarize_concept_f1`.

Parameters (``stages.concept_eval``): ``thresholds`` ([0, 0.15, 0.3, 0.6]), ``split``
(``{valid: 0.2, test: 0.1}``), ``units`` ([row, gene]), ``representations`` ([sae, raw,
untrained]), ``summary`` (``{tau: 0.15, concept_size: [5, 100]}``), ``dedup_cells`` (auto),
``seed`` (``run.seed``), ``top_m`` (200), ``batch_size`` (8192), ``max_shards``, ``device``
(auto), ``token_vocab`` (JSON ``{gene name: token id}`` when the store holds vocabulary
ids), ``store_root`` and ``checkpoints`` (``{<layer>: path}``: SAE checkpoints to score
instead of the ones selected by the ``sae`` stage).

Outputs: ``concept_f1.json`` and ``concept_f1.csv`` (one row per layer x unit x
representation), per layer ``<layer>/concept_f1.json`` and per evaluation
``<layer>/f1_<unit>_<representation>/{valid_concept_f1.csv, test_concept_f1.csv,
per_concept.csv, support.csv, split.json, counts.json}``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from interpretfm.concept_eval import (
    REPRESENTATIONS,
    UNITS,
    assign_rows,
    heldout_concept_f1,
    params_dict,
    params_from_mapping,
    summarize_concept_f1,
)
from interpretfm.concepts import Universe
from interpretfm.genes import resolver_for
from interpretfm.params import get_list, get_mapping, param_path
from interpretfm.runner import StageContext, register_stage
from interpretfm.sae import Encoder, load_checkpoint, raw_dims_checkpoint
from interpretfm.stages.concepts import load_gene_table
from interpretfm.stores import hidden_dim, resolve_store

OUTPUTS = ["concept_f1.json", "concept_f1.csv"]


def selected_checkpoints(ctx: StageContext, layer: str) -> Dict[str, Optional[str]]:
    """``{'sae': path, 'untrained': path}`` for a layer: the stage's ``checkpoints`` override
    or the ``sae`` stage's ``selected.json`` (``None`` when the λ rule selected nothing)."""
    override = get_mapping(ctx.params, "checkpoints", ctx.stage.name)
    with open(ctx.input_dir("sae") / "selected.json") as fh:
        sel = json.load(fh)["layers"].get(layer, {})
    sae = override.get(layer)
    sae = str(param_path(ctx.config, sae, ctx.stage.name, f"checkpoints.{layer}")) if sae else sel.get("checkpoint")
    untrained = (sel.get("untrained") or {}).get("checkpoint")
    return {"sae": sae, "untrained": untrained}


@register_stage("concept_eval", inputs=["extract", "sae", "concepts"], outputs=OUTPUTS)
def run_concept_eval(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Held-out concept alignment of SAE features, raw dimensions and the untrained SAE."""
    import pandas as pd

    cfg = ctx.config
    params = params_from_mapping(ctx.params, default_seed=cfg.run.seed)
    units = get_list(ctx.params, "units", list(UNITS), "concept_eval", choices=UNITS)
    reps = get_list(ctx.params, "representations", list(REPRESENTATIONS), "concept_eval", choices=REPRESENTATIONS)
    summ = get_mapping(ctx.params, "summary", "concept_eval")
    tau = float(summ.get("tau", 0.15))
    size_range = tuple(int(x) for x in summ.get("concept_size", (5, 100)))
    store = resolve_store(ctx)
    universe = Universe.read(ctx.input_dir("concepts") / "universe")
    genes = load_gene_table(cfg, ctx.config.stage_params("concepts"))
    resolver = resolver_for(cfg, ctx.params, store, genes)
    keys = [str(k) for k in universe.membership.index]
    key_to_row = {k: i for i, k in enumerate(keys)}
    membership = universe.membership.values
    term_ids = [str(c) for c in universe.membership.columns]
    sizes = universe.concept_sizes
    device = params.resolved_device()

    table = []
    per_layer: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        ckpts = selected_checkpoints(ctx, layer)
        d_in = hidden_dim(store, layer)
        encoders: Dict[str, Encoder] = {}
        for rep in reps:
            if rep == "raw":
                encoders[rep] = Encoder(raw_dims_checkpoint(d_in), device)
            elif ckpts.get(rep):
                encoders[rep] = Encoder(load_checkpoint(ckpts[rep]), device)
            else:
                ctx.log.warning("concept_eval: %s: no %s checkpoint; skipped", layer, rep)
        layer_summary: Dict[str, Any] = {"checkpoints": ckpts, "evaluations": {}}
        for unit in units:
            assignment = assign_rows(store, layer, resolver, key_to_row, unit, params)
            ctx.log.info("concept_eval: %s %s split: %d genes, %d mapped rows, %d duplicate rows dropped", layer, unit,
                         len(assignment.observed_gene_counts), assignment.meta(params)["n_rows_mapped"],
                         assignment.rows_dropped_duplicates)
            for rep, enc in encoders.items():
                res = heldout_concept_f1(store, layer, enc, membership, term_ids, assignment, params)
                out = ctx.path(layer, f"f1_{unit}_{rep}", "counts.json").parent
                res.write(out)
                summary, pc = summarize_concept_f1(res, sizes, tau=tau, size_range=size_range)
                pc.to_csv(out / "per_concept.csv", index=False)
                summary.update({"representation": rep, "unit": unit, "n_units": res.counts["n_units"],
                                "rows_by_split": res.counts["rows_by_split"], "checkpoint": ckpts.get(rep)})
                layer_summary["evaluations"][f"{unit}_{rep}"] = summary
                table.append({"layer": layer, "unit": unit, "representation": rep, "n_units": summary["n_units"],
                              "n_concepts": summary["n_concepts"], "concept_f1": summary["concept_f1"],
                              "share_test_f1_ge_0.5": summary["share_test_f1_ge_0.5"]})
                ctx.log.info("concept_eval: %s %s %s: concept F1 %s over %d concepts", layer, unit, rep,
                             None if summary["concept_f1"] is None else round(summary["concept_f1"], 4), summary["n_concepts"])
        ctx.write_json(f"{layer}/concept_f1.json", layer_summary)
        per_layer[layer] = layer_summary
    pd.DataFrame(table).to_csv(ctx.path("concept_f1.csv"), index=False)
    ctx.write_json("concept_f1.json", {"params": params_dict(params), "units": units, "representations": reps,
                                       "summary": {"tau": tau, "concept_size": list(size_range)}, "layers": per_layer})
    return {"evaluations": {f"{r['layer']}/{r['unit']}/{r['representation']}": r["concept_f1"] for r in table}}
