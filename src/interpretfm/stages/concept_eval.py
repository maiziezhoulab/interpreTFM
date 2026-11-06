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
instead of the ones selected by the ``sae`` stage), ``n_jobs`` (1: the evaluations one after
the other in this process; > 1: that many evaluations at a time, each in its own process) and
``threads_per_job`` (torch threads of every evaluation process; default: torch's own default,
which follows ``OMP_NUM_THREADS``). The evaluations are independent, so ``n_jobs`` and
``threads_per_job`` change the wall time only, never a number; both are left out of the stage's
configuration hash.

Outputs: ``concept_f1.json`` and ``concept_f1.csv`` (one row per layer x unit x
representation), per layer ``<layer>/concept_f1.json`` and per evaluation
``<layer>/f1_<unit>_<representation>/{valid_concept_f1.csv, test_concept_f1.csv,
per_concept.csv, support.csv, split.json, counts.json}``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from interpretfm.concept_eval import (
    REPRESENTATIONS,
    UNITS,
    assign_rows,
    heldout_concept_f1,
    params_dict,
    params_from_mapping,
    summarize_concept_f1,
)
from interpretfm.params import get_list, get_mapping, get_param, param_path
from interpretfm.runner import StageContext, register_stage
from interpretfm.stores import hidden_dim, resolve_store

OUTPUTS = ["concept_f1.json", "concept_f1.csv"]
HASH_IGNORE = ("n_jobs", "threads_per_job")


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


def evaluate_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """One held-out evaluation (a layer, a unit and a representation), as the stage runs it in
    process or in a worker process: assigns the rows, encodes the held-out rows, writes the
    evaluation directory and returns the summary. ``job`` holds the configuration (``cfg``), the
    stage parameters (``params``), ``store``, ``layer``, ``unit``, ``representation``,
    ``checkpoint`` (``None`` for the raw dimensions), ``universe_dir``, ``out_dir``, ``tau``,
    ``size_range`` and ``threads``."""
    import torch

    from interpretfm.concepts import Universe
    from interpretfm.genes import resolver_for
    from interpretfm.sae import Encoder, load_checkpoint, raw_dims_checkpoint
    from interpretfm.stages.concepts import load_gene_table

    t0 = time.time()
    if job.get("threads"):
        torch.set_num_threads(int(job["threads"]))
    cfg, stage_params, store, layer = job["cfg"], job["params"], Path(job["store"]), job["layer"]
    unit, rep = job["unit"], job["representation"]
    params = params_from_mapping(stage_params, default_seed=cfg.run.seed)
    universe = Universe.read(job["universe_dir"])
    genes = load_gene_table(cfg, cfg.stage_params("concepts"))
    resolver = resolver_for(cfg, stage_params, store, genes)
    keys = [str(k) for k in universe.membership.index]
    key_to_row = {k: i for i, k in enumerate(keys)}
    term_ids = [str(c) for c in universe.membership.columns]
    device = params.resolved_device()
    if rep == "raw":
        encoder = Encoder(raw_dims_checkpoint(hidden_dim(store, layer)), device)
    else:
        encoder = Encoder(load_checkpoint(job["checkpoint"]), device)
    assignment = assign_rows(store, layer, resolver, key_to_row, unit, params)
    meta = assignment.meta(params)
    res = heldout_concept_f1(store, layer, encoder, universe.membership.values, term_ids, assignment, params)
    out = Path(job["out_dir"])
    res.write(out)
    summary, pc = summarize_concept_f1(res, universe.concept_sizes, tau=float(job["tau"]), size_range=tuple(job["size_range"]))
    pc.to_csv(out / "per_concept.csv", index=False)
    summary.update({"representation": rep, "unit": unit, "n_units": res.counts["n_units"],
                    "rows_by_split": res.counts["rows_by_split"], "checkpoint": job["checkpoint"]})
    return {"layer": layer, "unit": unit, "representation": rep, "summary": summary, "seconds": time.time() - t0,
            "split": {"n_genes": len(assignment.observed_gene_counts), "n_rows_mapped": meta["n_rows_mapped"],
                      "rows_dropped_duplicates": assignment.rows_dropped_duplicates}}


def _run_jobs(jobs: List[Dict[str, Any]], n_jobs: int, log) -> List[Dict[str, Any]]:
    """Run the evaluations in order (``n_jobs`` = 1) or ``n_jobs`` at a time in spawned worker
    processes; results come back in job order either way."""
    if n_jobs <= 1 or len(jobs) <= 1:
        results = []
        for job in jobs:
            r = evaluate_one(job)
            _log_result(log, r)
            results.append(r)
        return results
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    log.info("concept_eval: %d evaluations, %d at a time in separate processes", len(jobs), n_jobs)
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_one, job) for job in jobs]
        results = []
        for f in futures:
            r = f.result()
            _log_result(log, r)
            results.append(r)
    return results


def _log_result(log, r: Dict[str, Any]) -> None:
    s = r["summary"]
    log.info("concept_eval: %s %s split: %d genes, %d mapped rows, %d duplicate rows dropped", r["layer"], r["unit"],
             r["split"]["n_genes"], r["split"]["n_rows_mapped"], r["split"]["rows_dropped_duplicates"])
    log.info("concept_eval: %s %s %s: concept F1 %s over %d concepts (%.0f s)", r["layer"], r["unit"], r["representation"],
             None if s["concept_f1"] is None else round(s["concept_f1"], 4), s["n_concepts"], r["seconds"])


@register_stage("concept_eval", inputs=["extract", "sae", "concepts"], outputs=OUTPUTS, hash_ignore=HASH_IGNORE)
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
    n_jobs = get_param(ctx.params, "n_jobs", 1, "concept_eval", int)
    threads = get_param(ctx.params, "threads_per_job", None, "concept_eval", int)
    if n_jobs < 1 or (threads is not None and threads < 1):
        raise ValueError("stages.concept_eval: n_jobs and threads_per_job must be >= 1")
    store = resolve_store(ctx)
    universe_dir = ctx.input_dir("concepts") / "universe"

    jobs: List[Dict[str, Any]] = []
    ckpts_by_layer: Dict[str, Dict[str, Optional[str]]] = {}
    for layer in cfg.model.layers:
        ckpts = selected_checkpoints(ctx, layer)
        ckpts_by_layer[layer] = ckpts
        for unit in units:
            for rep in reps:
                if rep != "raw" and not ckpts.get(rep):
                    ctx.log.warning("concept_eval: %s: no %s checkpoint; skipped", layer, rep)
                    continue
                jobs.append({"cfg": cfg, "params": dict(ctx.params), "store": str(store), "layer": layer, "unit": unit,
                             "representation": rep, "checkpoint": None if rep == "raw" else ckpts[rep],
                             "universe_dir": str(universe_dir), "out_dir": str(ctx.path(layer, f"f1_{unit}_{rep}", "counts.json").parent),
                             "tau": tau, "size_range": list(size_range), "threads": threads})
    results = _run_jobs(jobs, n_jobs, ctx.log)

    table = []
    per_layer: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        layer_summary: Dict[str, Any] = {"checkpoints": ckpts_by_layer[layer], "evaluations": {}}
        for r in results:
            if r["layer"] != layer:
                continue
            summary = r["summary"]
            layer_summary["evaluations"][f"{r['unit']}_{r['representation']}"] = summary
            table.append({"layer": layer, "unit": r["unit"], "representation": r["representation"], "n_units": summary["n_units"],
                          "n_concepts": summary["n_concepts"], "concept_f1": summary["concept_f1"],
                          "share_test_f1_ge_0.5": summary["share_test_f1_ge_0.5"]})
        ctx.write_json(f"{layer}/concept_f1.json", layer_summary)
        per_layer[layer] = layer_summary
    pd.DataFrame(table).to_csv(ctx.path("concept_f1.csv"), index=False)
    ctx.write_json("concept_f1.json", {"params": params_dict(params), "units": units, "representations": reps,
                                       "summary": {"tau": tau, "concept_size": list(size_range)},
                                       "n_jobs": n_jobs, "threads_per_job": threads, "layers": per_layer})
    return {"evaluations": {f"{r['layer']}/{r['unit']}/{r['representation']}": r["concept_f1"] for r in table}}
