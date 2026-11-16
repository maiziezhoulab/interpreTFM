"""Post hoc controls: randomly re-initialized models (concept F1 and the gene-level readout) and the
gene-symbol baseline of the gene-level readout (Methods; Results; Supplementary Tables S1 and S3).

Data: the configurations ``reinit_cosmx_<model>_<weights>.yaml`` and ``readout_controls_<layer>.yaml``
need ``INTERPRETFM_REINIT_CACHE`` (the stores, SAE checkpoints, λ evaluations and held-out tables of the
re-initialized-model run); the cached results of the post hoc readout are read from
``INTERPRETFM_POSTHOC_CACHE`` when it is set (``PB-RANDTWIN/results``, ``PB-NAMEPROBE/results``).

- ``*_from_cache``: the λ rule on the cached evaluations of the five λ, and the concept-F1 summary rule
  on the cached held-out tables (seconds); ``test_symbol_removal_from_cached_gene_means``: the readout
  code on the manuscript's cached per-gene mean hidden states (minutes per layer).
- run-dir tests: the ``sae`` and ``concept_eval`` stages of the six ``reinit_*`` runs, and the
  ``gene_readout`` stage of the six ``readout_controls_*`` runs (``make repro CONFIG=<name>``).
- ``test_reinit_extraction_matches_cached_store`` (GPU, ``INTERPRETFM_REPRO_GPU=1``): re-initializes the
  model and extracts the first 40 cells of the cached store's first shard in the model environment.

Tolerances: concept F1 ±0.0005 from the cached tables and ±0.005 from the store, counts exact from the
tables and ±2 from the store (as for the six representative layers). The readout controls use the
manuscript's gene universe and splits, so the means over splits are compared at ±0.001 with the
printed values (±0.0005 rounding) and ±0.0005 with the cached results, every split at ±0.0005 and the
SD over splits at ±0.002. The stage averages every token of a gene; the manuscript's mean hidden states
of the trained models averaged the tokens of every k-th shard of the store (CosMx scGPT every 4th,
CosMx Geneformer and C2S-Scale every 2nd, Xenium about 25 shards), so for the rows built from them the
run-dir test reports the cached values, and the cached-gene-means test checks them at ±0.0005.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from expected import EXPECTED, READOUT_CONTROLS, REINIT, SYMBOL_CONTROLS, SYMBOL_DROP
from repro_support import cached_path, check, dev_table, report
from test_extraction import gpu

from interpretfm.concept_eval import HeldoutResult, summarize_concept_f1
from interpretfm.config import ConfigError
from interpretfm.sae import select_lambda

FORMS = ("trained", "random_table", "random_all")
REINIT_DIR = {"trained": "trained", "random_table": "rand_table", "random_all": "rand_all"}  # cache layout
REINIT_TAG = {"scgpt": "layer_1_norm2", "geneformer": "layer_1"}
UNIT_TAG = {"row": "gene_token_row", "gene": "gene_identity"}
REINIT_CONFIGS = [f"reinit_cosmx_{m}_{f}" for m in REINIT for f in FORMS]
READOUT_LAYERS = list(READOUT_CONTROLS)
READOUT_STAGES = ["ingest", "extract", "sae", "concepts", "gene_readout"]
REINIT_STAGES = ["ingest", "extract", "sae", "concepts", "concept_eval"]
CACHE_MODEL = {"scgpt": "scgpt", "geneformer": "geneformer", "c2s_scale": "c2sscale"}
N_EXTRACT = 40


def _split_name(name: str):
    model, form = name[len("reinit_cosmx_"):].split("_", 1)
    return model, form


def reinit_cache(*parts: str) -> Path:
    root = os.environ.get("INTERPRETFM_REINIT_CACHE")
    if not root:
        pytest.skip("INTERPRETFM_REINIT_CACHE is not set (the cached run of the randomly re-initialized models)")
    p = Path(root).joinpath(*parts)
    if not p.exists():
        pytest.skip(f"not in the re-initialized-model cache: {p}")
    return p


def posthoc_cached(*parts: str) -> Optional[Path]:
    root = os.environ.get("INTERPRETFM_POSTHOC_CACHE")
    p = Path(root).joinpath(*parts) if root else None
    return p if p is not None and p.exists() else None


def config_or_skip(paper, name: str):
    try:
        return paper.config(name)
    except ConfigError as e:
        pytest.skip(f"{name}: {str(e).splitlines()[0]} (set INTERPRETFM_REINIT_CACHE in the env file)")


def _concept_sizes():
    p = cached_path(EXPECTED["cosmx_scgpt_layer2"]["gt_dir"], "gprofiler_binary_gene_by_term.csv")
    if p is None:
        pytest.skip("cached concept table not available")
    return pd.read_csv(p, index_col=0).sum(0).astype(int)


def _cached_reinit_summary(model: str, form: str, unit: str):
    t = pd.read_csv(reinit_cache("summary_long.csv"))
    r = t[(t.model == model) & (t.form == REINIT_DIR[form]) & (t.source == "sparse") & (t.split == UNIT_TAG[unit])]
    return r.iloc[0] if len(r) else None


def _cached_f1_dir(model: str, form: str, unit: str) -> Path:
    return reinit_cache("runs", f"{model}__{REINIT_DIR[form]}", model, REINIT_TAG[model], "f1_heldout",
                        f"heldout_sparse__{UNIT_TAG[unit]}")


# --- randomly re-initialized models: rules on the cached evaluations -------------------------------


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model", list(REINIT))
def test_reinit_lambda_rule_from_cache(model, form):
    """The fixed λ rule over the cached QC of the five λ chooses the printed λ (Methods)."""
    exp = REINIT[model][form]
    pick = json.load(open(reinit_cache("picks", f"{model}__{REINIT_DIR[form]}.json")))
    qc = {float(k): {"fvu": e["FVU"], "l0_frac": e["L0_pct"] / 100.0, "dead_frac": e["DFR"]} for k, e in pick["evals"].items()}
    assert len(qc) == 5
    chosen, rule = select_lambda(qc, l0_max=0.02, dead_max=0.25)
    report(f"reinit {model} {form} λ (rule on the cached QC)", chosen, exp["lambda"], pick["l1"])
    assert chosen == exp["lambda"] == float(pick["l1"]), rule


@pytest.mark.parametrize("unit", ["row", "gene"])
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model", list(REINIT))
def test_reinit_concept_f1_summary_rule_from_cache(model, form, unit):
    """The concept-F1 summary rule on the cached held-out tables of the selected sparse SAE gives the printed
    concept F1 and, on the row split, the printed number of concepts with a test F1 of at least 0.5."""
    exp = REINIT[model][form]
    d = _cached_f1_dir(model, form, unit)
    valid = dev_table(d / "valid_concept_f1_scores.csv", 0.15)
    test = dev_table(d / "test_concept_f1_scores.csv", 0.15)
    support = pd.read_csv(d / "concept_support_by_split.csv")
    res = HeldoutResult(valid=valid, test=test, support=support, counts={}, split_meta={})
    summary, pc = summarize_concept_f1(res, _concept_sizes(), tau=0.15, size_range=(5, 100))
    src = _cached_reinit_summary(model, form, unit)
    check(f"reinit {model} {form} {unit} concept F1 (sparse, cached tables)", summary["concept_f1"], exp[f"f1_{unit}"],
          None if src is None else float(src.specific_mean), tol=0.0005)
    if src is not None:
        assert summary["n_concepts"] == int(src.n_specific)
    if unit == "row":
        marked = int(((pc["test_f1"] >= 0.5) & pc["supported"] & pc["in_size_range"]).sum())
        cached = None if src is None else int(round(float(src.specific_ge0p5) * int(src.n_specific)))
        report(f"reinit {model} {form} concepts with test F1 >= 0.5 (cached tables)", marked, exp["marked"], cached)
        assert marked == exp["marked"] and (cached is None or marked == cached)


# --- randomly re-initialized models: the pipeline run ---------------------------------------------


@pytest.mark.parametrize("name", REINIT_CONFIGS)
def test_reinit_sae_stage(paper, name):
    """The ``sae`` stage on the cached checkpoints of the five λ: QC as cached, the printed λ."""
    model, form = _split_name(name)
    exp = REINIT[model][form]
    config_or_skip(paper, name)
    rd = paper.run_dir(name, stages=["ingest", "extract", "sae"])
    sel = json.load(open(rd / "04_sae" / "selected.json"))["layers"][REINIT[model]["layer"]]
    report(f"{name} λ (sae stage)", sel["lambda"], exp["lambda"])
    assert sel["lambda"] == exp["lambda"]
    pick = json.load(open(reinit_cache("picks", f"{model}__{REINIT_DIR[form]}.json")))
    for lam, e in pick["evals"].items():
        qc = sel["qc"][f"{float(lam):g}"]
        assert sel["n_qc_rows"] == e["eval_rows"], (sel["n_qc_rows"], e["eval_rows"])
        check(f"{name} FVU at λ={lam}", qc["fvu"], None, e["FVU"], tol=0.005)
        check(f"{name} L0 fraction at λ={lam}", qc["l0_frac"], None, e["L0_pct"] / 100.0, tol=0.001)
        check(f"{name} dead-feature rate at λ={lam}", qc["dead_frac"], None, e["DFR"], tol=0.001)


@pytest.mark.parametrize("unit", ["row", "gene"])
@pytest.mark.parametrize("name", REINIT_CONFIGS)
def test_reinit_concept_f1(paper, name, unit):
    """Concept F1 of the selected sparse SAE recomputed from the store (both held-out units), the concepts
    marked on the row split, and the manuscript run's partitions."""
    model, form = _split_name(name)
    exp, layer = REINIT[model][form], REINIT[model]["layer"]
    config_or_skip(paper, name)
    rd = paper.run_dir(name, stages=REINIT_STAGES)
    table = pd.read_csv(rd / "06_concept_eval" / "concept_f1.csv")
    row = table[(table.layer == layer) & (table.unit == unit) & (table.representation == "sae")]
    assert len(row) == 1, f"{name}: no {unit}/sae evaluation"
    row = row.iloc[0]
    src = _cached_reinit_summary(model, form, unit)
    check(f"{name} {unit} concept F1 (sparse)", row["concept_f1"], exp[f"f1_{unit}"],
          None if src is None else float(src.specific_mean), tol=0.005)
    if src is not None:
        report(f"{name} {unit} concepts scored", int(row["n_concepts"]), None, int(src.n_specific), 2)
        assert abs(int(row["n_concepts"]) - int(src.n_specific)) <= 2
    ev = rd / "06_concept_eval" / layer / f"f1_{unit}_sae"
    if unit == "row":
        pc = pd.read_csv(ev / "per_concept.csv")
        marked = int(((pc["test_f1"] >= 0.5) & pc["supported"] & pc["in_size_range"]).sum())
        cached = None if src is None else int(round(float(src.specific_ge0p5) * int(src.n_specific)))
        check(f"{name} concepts with test F1 >= 0.5 (row split)", marked, exp["marked"], cached, tol=2)
    d = _cached_f1_dir(model, form, unit)
    if unit == "gene":
        ours, theirs = json.load(open(ev / "split.json")), json.load(open(d / "heldout_gene_split.json"))
        for part in ("test_genes", "valid_genes"):
            report(f"{name} whole-gene split: {part} identical to the manuscript run", set(ours[part]) == set(theirs[part]), True)
            assert set(ours[part]) == set(theirs[part])
    else:
        ours, theirs = json.load(open(ev / "counts.json")), json.load(open(d / "counts_summary.json"))
        report(f"{name} row split rows (valid, test)", ours["rows_by_split"], None, theirs["rows_by_split"])
        assert ours["rows_by_split"] == theirs["rows_by_split"]


@pytest.mark.parametrize("model", list(REINIT))
def test_reinit_sample_cells_are_the_cached_stores_cells(model):
    """The shipped cell list of the re-initialized runs holds exactly the cells of the cached stores."""
    from interpretfm.extract import read_cell_list
    from interpretfm.stores import list_shards, torch_load

    exp = REINIT[model]
    listed = read_cell_list(Path(__file__).resolve().parents[2] / "configs" / "paper" / f"cosmx_{model}_sample_cells.txt.gz")
    assert len(listed) == len(set(listed)) == exp["n_cells"]
    for form in FORMS:
        store = reinit_cache("stores", model, REINIT_DIR[form])
        cells = set()
        for shard in list_shards(store, exp["layer"]):
            cells.update(str(e) for e in torch_load(shard / "index.pt")["example_ids"])
        report(f"reinit {model} {form}: cells in the cached store", len(cells), exp["n_cells"])
        assert cells == set(listed)


@gpu
@pytest.mark.parametrize("name", REINIT_CONFIGS)
def test_reinit_extraction_matches_cached_store(paper, name):
    """Re-initialize the model (stages.extract.weights, seed 0) in its environment on the GPU, extract the first
    40 cells of the cached store's first shard and compare them with the cached store: identical token
    sequences and a cosine of at least 0.999 per token."""
    from interpretfm.extract import compare_stores
    from interpretfm.runner import run, stage_dir
    from interpretfm.stores import list_shards, torch_load

    model, form = _split_name(name)
    exp = REINIT[model]
    cfg = config_or_skip(paper, name)
    ext = dict(cfg.stage_params("extract"))
    reference = Path(ext.pop("store_root"))
    if not reference.is_dir():
        pytest.skip(f"cached store {reference} not found")
    layer = exp["layer"]
    first = list_shards(reference, layer)[0]
    cells = list(dict.fromkeys(str(e) for e in torch_load(first / "index.pt")["example_ids"]))[:N_EXTRACT]
    out = Path(os.environ["INTERPRETFM_RUNS"]) / f"{name}_extract{N_EXTRACT}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells.txt").write_text("\n".join(cells) + "\n")
    ext["cell_subset"] = str(out / "cells.txt")
    cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, name=f"{name}_extract{N_EXTRACT}", out_dir=out / "run"),
                              stages={**cfg.stages, "extract": ext})
    assert cfg.model.env.kind != "current", "the check must cross the environment boundary"
    rows = run(cfg, stages=["ingest", "extract"])
    assert all(r["status"] in ("done", "cached") for r in rows), rows
    sdir = stage_dir(cfg.run.out_dir, "extract")
    manifest = json.load(open(sdir / layer / "manifest.json"))
    assert manifest["n_cells"] == N_EXTRACT and manifest["environment"]["torch"]["cuda_available"]
    w = manifest["adapter"]["weights"]
    report(f"{name} weights", {k: w.get(k) for k in ("weights", "seed", "n_params", "n_changed")}, form)
    assert w["weights"] == form
    if form != "trained":
        assert w["seed"] == 0
        expected_unchanged = [w["embedding_table"]] if form == "random_table" else []
        assert w["unchanged"] == expected_unchanged and w["n_changed"] == w["n_params"] - len(expected_unchanged)
    cmp = compare_stores(sdir / "store", reference, layer, cells, ref_exclude_tokens=["60694"] if model == "scgpt" else [])
    json.dump(cmp, open(sdir / "check_rows_vs_cached.json", "w"), indent=1)
    report(f"{name} token sequences identical", cmp["n_cells_token_sequence_identical"], N_EXTRACT)
    report(f"{name} per-token cosine min / median", (cmp["cosine"]["min"], cmp["cosine"]["median"]), tol=">= 0.999")
    report(f"{name} max abs diff / relative to the cached scale", (cmp["max_abs_diff"], cmp["max_abs_diff_over_ref_scale"]))
    assert cmp["token_sequence_identical"] and cmp["n_cells_compared"] == N_EXTRACT
    assert cmp["n_rows_compared"] > 0 and cmp["n_rows_cosine_below_0.999"] == 0 and cmp["cosine"]["min"] >= 0.999


# --- gene-level readout controls ------------------------------------------------------------------


def _readout(rd: Path):
    out = json.load(open(rd / "07_gene_readout" / "gene_readout.json"))
    per_split = pd.read_csv(rd / "07_gene_readout" / "gene_readout.csv")
    return out["representations"], per_split


def _mean_sd(reps, rep: str):
    assert rep in reps, f"representation {rep} missing from the readout"
    r = reps[rep]
    assert r["n_splits"] == 5
    return r["mean_probe_test_auroc_standardize_mean"], r["mean_probe_test_auroc_standardize_std"]


def _cached_summary(kind: str, dataset: str, rep: str):
    """``(mean, sd)`` of a cached post hoc readout row (``kind``: names or randtwin), else ``None``."""
    if kind == "names":
        p = posthoc_cached("PB-NAMEPROBE", "results", "summary.csv")
        t = None if p is None else pd.read_csv(p)
        r = None if t is None else t[(t.dataset == dataset) & (t.rep == rep)]
    else:
        p = posthoc_cached("PB-RANDTWIN", "results", "summary.csv")
        t = None if p is None else pd.read_csv(p)
        r = None if t is None else t[t.rep == rep]
    return None if r is None or not len(r) else (float(r.iloc[0]["C1_mean"]), float(r.iloc[0]["C1_sd"]))


def _cached_per_split(kind: str, dataset: str, rep: str):
    p = posthoc_cached("PB-NAMEPROBE" if kind == "names" else "PB-RANDTWIN", "results", "per_split.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    t = t[(t.rep == rep) & ((t.dataset == dataset) if "dataset" in t.columns else True)]
    return {("s0" if s == "s0_c2s" else s): float(v) for s, v in zip(t["split"], t["probe_C1"])} or None


def _cached_drop(dataset: str, rep: str):
    p = posthoc_cached("PB-NAMEPROBE", "results", "drops.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    r = t[(t.dataset == dataset) & (t.rep == rep)]
    return float(r.iloc[0]["drop_mean"]) if len(r) else None


def _cached_hidden(exp):
    p = cached_path("revision_metrics", "gene_split_ceiling", "summary.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    r = t[(t.dataset == exp["dataset"]) & (t.rep == f"raw|{exp['cache_model']}|{exp['layer']}")]
    return None if not len(r) else (float(r.iloc[0]["mean_probe_test_auroc_C1__mean"]), float(r.iloc[0]["mean_probe_test_auroc_C1__std"]))


def _compare_values(label, mean, sd, ours_by_split, printed, cached, cached_splits, same_inputs=True, tol=0.001):
    """Mean and SD over splits against the printed (``tol``) and the cached values, and every split against the
    cached run. ``same_inputs=False`` (rows built from the hidden-state means, which the manuscript's run took
    from every k-th shard): the cached values are reported, not asserted."""
    tol_cached = 0.0005 if same_inputs else None
    check(f"{label} probe AUROC (standardized)", mean, None if printed is None else printed[0],
          None if cached is None or not same_inputs else cached[0], tol=tol, tol_cached=tol_cached)
    check(f"{label} SD over splits", sd, None if printed is None else printed[1],
          None if cached is None or not same_inputs else cached[1], tol=0.002, tol_cached=tol_cached)
    if cached is not None and not same_inputs:
        report(f"{label} cached run (every k-th shard of the store), mean and SD", (round(mean, 6), round(sd, 6)), None, cached)
    if cached_splits:
        diff = max(abs(float(ours_by_split[s]) - v) for s, v in cached_splits.items())
        assert set(ours_by_split.index) == set(cached_splits)
        report(f"{label} largest per-split difference to the cached run", round(diff, 6), None, 0.0, 0.0005 if same_inputs else None)
        if same_inputs:
            assert diff <= 0.0005
    return mean


def _compare(label, reps, per_split, rep, printed, cached, cached_splits, same_inputs=True):
    mean, sd = _mean_sd(reps, rep)
    ours = per_split[per_split.representation == rep].set_index("split")["mean_probe_test_auroc_standardize"]
    return _compare_values(label, mean, sd, ours, printed, cached, cached_splits, same_inputs)


@pytest.mark.parametrize("name", READOUT_LAYERS)
def test_readout_controls(paper, name):
    """The gene-level readout on the manuscript's genes and splits: the raw hidden state (Table S3), the hidden
    state with its symbol component removed, the randomly re-initialized models (CosMx scGPT and Geneformer),
    the gene-symbol controls and co-expression with its symbol component removed (post hoc rows of Table S3)."""
    exp = EXPECTED[name]
    ds, layer = exp["dataset"], exp["layer"]
    cache_rep = f"raw|{exp['cache_model']}|{layer}"
    cfg_name = f"readout_controls_{name}"
    config_or_skip(paper, cfg_name)
    rd = paper.run_dir(cfg_name, stages=READOUT_STAGES)
    reps, per_split = _readout(rd)

    # the manuscript's gene universe and splits
    splits = json.load(open(rd / "07_gene_readout" / "splits.json"))
    meta_path = cached_path("revision_metrics", "gene_split_ceiling", f"_meta_{ds}.json")
    if meta_path is not None:
        meta = json.load(open(meta_path))
        report(f"{name} readout genes", len(splits["universe"]), None, len(meta["universe"]))
        assert sorted(splits["universe"]) == sorted(meta["universe"])
        for s in ("s0", "s1", "s2", "s3", "s4"):
            theirs = meta["splits"]["s0_c2s" if s == "s0" else s]
            assert sorted(splits["splits"][s]["test"]) == sorted(theirs["test"]), s
            assert splits["splits"][s]["n_train_valid"] == theirs["n_tv"], s
    assert splits["listed_not_observed"] == []

    hidden = _compare(f"{name} raw hidden state", reps, per_split, f"hidden|{layer}", (exp["gene_readout"]["hidden"][0], None),
                      _cached_hidden(exp), _cached_per_split("names", ds, cache_rep), same_inputs=False)
    removed = _compare(f"{name} raw hidden state, symbol component removed", reps, per_split, f"hidden|{layer}|symbol_removed",
                       READOUT_CONTROLS[name]["symbol_removed"], _cached_summary("names", ds, cache_rep + "|resid"),
                       _cached_per_split("names", ds, cache_rep + "|resid"), same_inputs=False)
    cached_drop = _cached_drop(ds, cache_rep)
    check(f"{name} drop from removing the symbol component", hidden - removed, SYMBOL_DROP["cosmx"].get(name), None, tol=0.001)
    report(f"{name} drop from removing the symbol component, cached run", round(hidden - removed, 6), None, cached_drop)
    for form in ("random_table", "random_all"):
        if form in READOUT_CONTROLS[name]:
            model = exp["model"]
            _compare(f"{name} re-initialized ({form})", reps, per_split, f"{form}|{layer}", READOUT_CONTROLS[name][form],
                     _cached_summary("randtwin", ds, f"d6|{model}|{REINIT_DIR[form]}"),
                     _cached_per_split("randtwin", ds, f"d6|{model}|{REINIT_DIR[form]}"))
    for rep, key, cache_key in (("names_tri|-", "names_tri", "names_tri|-|-"), ("names_pc50|-", "names_pc50", "names_pc50|-|-"),
                                ("coexpression_pc50|-|symbol_removed", "coexpression_pc50_symbol_removed", "coexp_pc50|-|-|resid")):
        _compare(f"{name} {key}", reps, per_split, rep, SYMBOL_CONTROLS[ds][key], _cached_summary("names", ds, cache_key),
                 _cached_per_split("names", ds, cache_key))


LEXICAL_MEANS = {"cosmx": "gene_means", "xenium": "gene_means_xenium_v1"}  # cache layout of the manuscript's gene means


@pytest.mark.parametrize("name", READOUT_LAYERS)
def test_symbol_removal_from_cached_gene_means(paper, name):
    """The readout code (gene-symbol components, the symbol-removed representation, the given first split, the
    probe) applied to the manuscript's cached per-gene mean hidden states (every k-th shard of the store) gives the
    printed raw and symbol-removed values (Table S3) and the drop (Results), split by split as the cached run."""
    from interpretfm.gene_readout import (ReadoutParams, make_splits, read_gene_list, read_split_file, readout,
                                          summarize_readout, symbol_representations)
    from interpretfm.genes import GeneTable

    import anndata as ad

    exp = EXPECTED[name]
    ds, layer, cache_rep = exp["dataset"], exp["layer"], f"raw|{exp['cache_model']}|{exp['layer']}"
    means = cached_path("revision_metrics", "lexical_probe", LEXICAL_MEANS[ds], f"{exp['cache_model']}__{layer}.npz")
    split_file = cached_path("sae_v2", ds, "c2sscale", "layer_13", "f1_heldout", "heldout_raw__gene_identity", "heldout_gene_split.json")
    gt = cached_path(exp["gt_dir"], "gprofiler_binary_gene_by_term.csv")
    if means is None or split_file is None or gt is None:
        pytest.skip("cached gene means, split file or concept table not available")
    cfg = paper.config(name)
    gk = cfg.stage_params("concepts").get("gene_keys") or {}
    genes = GeneTable.from_adata(ad.read_h5ad(cfg.dataset.h5ad, backed="r"), cfg.dataset.gene_id,
                                 symbol_column=gk.get("symbol_column"), ensembl_column=gk.get("ensembl_column"))
    universe = sorted(read_gene_list(Path(__file__).resolve().parents[2] / "configs" / "paper" / f"{ds}_readout_genes.txt"))
    z = np.load(means)
    pos = {k: i for i, k in enumerate(genes.resolve(z["genes"].astype(str)))}
    H = z["hidden_mean"][[pos[k] for k in universe]].astype(np.float32)
    sym = dict(zip(genes.keys, genes.symbols))
    _tri, P, _grams = symbol_representations([sym.get(k) or k for k in universe])
    table = pd.read_csv(gt, index_col=0)
    table.index = table.index.astype(str)
    sizes = table.sum(0)
    terms = sizes.index[(sizes >= 5) & (sizes <= 100)]
    splits = make_splits(universe, 5, 0.2, 0.1, 0, first=read_split_file(split_file))
    pc = readout({"hidden": (H, "raw")}, table.loc[universe, terms].values.astype(np.int8), list(terms), sizes[terms].values,
                 {g: i for i, g in enumerate(universe)}, splits, ReadoutParams(scaling=["standardize"]), residuals=(P, ["hidden"]))
    per_split, overall = summarize_readout(pc)
    by = overall.set_index("representation")
    values = {}
    for rep, label, printed, cached, cached_splits in (
            ("hidden", "raw hidden state", (exp["gene_readout"]["hidden"][0], None), _cached_hidden(exp),
             _cached_per_split("names", ds, cache_rep)),
            ("hidden|symbol_removed", "raw hidden state, symbol component removed", READOUT_CONTROLS[name]["symbol_removed"],
             _cached_summary("names", ds, cache_rep + "|resid"), _cached_per_split("names", ds, cache_rep + "|resid"))):
        ours = per_split[per_split.representation == rep].set_index("split")["mean_probe_test_auroc_standardize"]
        values[rep] = _compare_values(f"{name} {label} (cached gene means)", float(by.loc[rep, "mean_probe_test_auroc_standardize_mean"]),
                                      float(by.loc[rep, "mean_probe_test_auroc_standardize_std"]), ours, printed, cached,
                                      cached_splits, tol=0.0005)
    drop = values["hidden"] - values["hidden|symbol_removed"]
    check(f"{name} drop from removing the symbol component (cached gene means)", drop, SYMBOL_DROP["cosmx"].get(name),
          _cached_drop(ds, cache_rep), tol=0.0005)


def test_symbol_removal_drops(paper):
    """Results: removing the symbol component lowers every model's readout by 0.06-0.09."""
    drops = {}
    for name in READOUT_LAYERS:
        config_or_skip(paper, f"readout_controls_{name}")
        rd = paper.run_dir(f"readout_controls_{name}", stages=READOUT_STAGES)
        reps, _ = _readout(rd)
        layer = EXPECTED[name]["layer"]
        drops[name] = _mean_sd(reps, f"hidden|{layer}")[0] - _mean_sd(reps, f"hidden|{layer}|symbol_removed")[0]
    lo, hi = SYMBOL_DROP["range"]
    report("drops from removing the symbol component (min, max)", (round(min(drops.values()), 4), round(max(drops.values()), 4)),
           f"{lo}-{hi}")
    assert round(min(drops.values()), 2) == lo and round(max(drops.values()), 2) == hi
    cosmx = {k: v for k, v in drops.items() if k.startswith("cosmx")}
    assert max(cosmx, key=cosmx.get) == "cosmx_c2s_layer14"


@pytest.mark.parametrize("dataset", ["cosmx", "xenium"])
def test_readout_genes_are_observed_by_all_three_models(paper, dataset):
    """The shipped gene list of the readout controls is the set of genes observed by all three models: the
    intersection of the gene universes of the three representative layers' gene_readout runs."""
    path = Path(__file__).resolve().parents[2] / "configs" / "paper" / f"{dataset}_readout_genes.txt"
    from interpretfm.gene_readout import read_gene_list

    listed = read_gene_list(path)
    observed = None
    for name, exp in EXPECTED.items():
        if exp["dataset"] != dataset:
            continue
        rd = paper.run_dir(name, stages=["ingest", "extract", "sae", "concepts", "gene_readout"])
        uni = set(json.load(open(rd / "07_gene_readout" / "splits.json"))["universe"])
        observed = uni if observed is None else observed & uni
    report(f"{dataset} genes observed by all three models", len(observed), len(listed))
    assert sorted(observed) == sorted(listed) and len(listed) == len(set(listed))
    meta = cached_path("revision_metrics", "gene_split_ceiling", f"_meta_{dataset}.json")
    if meta is not None:
        assert sorted(json.load(open(meta))["universe"]) == sorted(listed)
