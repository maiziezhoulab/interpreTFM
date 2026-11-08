"""T1-T5 and the concept counts (AUDIT §4) for the model-level stages: TIS gain, the λ rule and the
untrained baseline, concept F1 under both held-out units, and the gene-level readout.

Two kinds of test:

- ``*_from_cache``: seconds; apply the pipeline's rules (λ selection, concept-F1 summary) to the
  manuscript's cached intermediates and compare with the published numbers.
- the others read the finished run directory of a paper configuration (``make repro``).

Tolerances: TIS and SAE QC are deterministic given the seed (±0.0005 on means after the
manuscript's rounding, ±0.001 on QC fractions); concept F1 ±0.005 (counts ±2); the gene readout
is a mean over 5 whole-gene splits whose seeds differ from the manuscript run's (same rule,
different draws; SD over splits is 0.01-0.03), so ±0.03 with the published ordering preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro_support import cached_path, check, known_difference, report
from expected import EXPECTED

from interpretfm.concept_eval import HeldoutResult, summarize_concept_f1
from interpretfm.sae import load_checkpoint, select_lambda

LAYERS = list(EXPECTED)
LAMBDA_TAGS = {0.1: "0p1", 0.3: "0p3", 1.0: "1", 3.0: "3", 10.0: "10"}
REP_TAG = {"sae": "sparse", "raw": "raw", "untrained": "randinit"}
UNIT_TAG = {"row": "gene_token_row", "gene": "gene_identity"}


# --- cached sources -------------------------------------------------------------------------------


def _cached_tis(exp):
    if not exp["tis_cache"]:
        return None
    p = cached_path(exp["tis_cache"], exp["cache_model"], exp["cache_layer"], "summary_row.json")
    return json.load(open(p)) if p else None


def _cached_sae_eval(exp, lam: float):
    p = cached_path("sae_v2", exp["dataset"], exp["cache_model"], exp["cache_layer"], "sae_eval",
                    f"v2biased__steps_8000__lam_{LAMBDA_TAGS[lam]}", "sae_eval.json")
    return json.load(open(p))["best"] if p else None


def _cached_f1_dir(exp, unit: str, rep: str):
    return cached_path("sae_v2", exp["dataset"], exp["cache_model"], exp["cache_layer"], "f1_heldout",
                       f"heldout_{REP_TAG[rep]}__{UNIT_TAG[unit]}")


def _cached_concept_f1_summary(exp):
    p = cached_path("revision_metrics", "concept_f1_v2", "summary_long.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    t = t[(t.dataset == exp["dataset"]) & (t.model == exp["cache_model"]) & (t.layer == exp["cache_layer"])]
    return {(UNIT_TAG_INV[r.split], SRC_INV[r.source]): r for r in t.itertuples() if r.source in SRC_INV}


UNIT_TAG_INV = {v: k for k, v in UNIT_TAG.items()}
SRC_INV = {v: k for k, v in REP_TAG.items()}


def _cached_readout(exp, scaling: str):
    """``{representation: (mean, sd, single)}`` from the manuscript's gene-split tables."""
    p = cached_path("revision_metrics", "gene_split_ceiling", "summary.csv" if scaling == "standardize" else "summary_center.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    t = t[t.dataset == exp["dataset"]]
    names = {"hidden": f"raw|{exp['cache_model']}|{exp['layer']}", "sae": f"sparse|{exp['cache_model']}|{exp['layer']}",
             "coexpression": "coexp_full|-|-", "coexpression_pc50": "coexp_pc50|-|-", "celltype": "celltype|-|-",
             "trivial": "trivial2|-|-", "random": f"random_{exp['hidden_dim']}|-|-"}
    out = {}
    for k, rep in names.items():
        r = t[t.rep == rep]
        if len(r):
            r = r.iloc[0]
            out[k] = (float(r["mean_probe_test_auroc_C1__mean"]), float(r["mean_probe_test_auroc_C1__std"]),
                      float(r["mean_single_test_auroc__mean"]))
    return out


def _dev_table(path: Path, tau: float):
    """A cached held-out table restricted to ``tau`` (the dev column is ``threshold_pct``)."""
    cols = ["concept", "feature", "threshold_pct", "f1", "tp"]
    parts = [c[np.isclose(c.threshold_pct, tau)] for c in pd.read_csv(path, usecols=cols, chunksize=2_000_000)]
    return pd.concat(parts, ignore_index=True).rename(columns={"threshold_pct": "threshold"})


def _concept_sizes(exp):
    p = cached_path(exp["gt_dir"], "gprofiler_binary_gene_by_term.csv")
    if p is None:
        pytest.skip("cached concept table not available")
    g = pd.read_csv(p, index_col=0)
    return g.sum(0).astype(int)


# --- T2 and T3/T4 from the cache (seconds to minutes) --------------------------------------------


@pytest.mark.parametrize("name", LAYERS)
def test_t2_lambda_rule_from_cache(name):
    """The fixed rule over the cached QC of the five λ selects λ = 1 (sparse_selection.json)."""
    exp = EXPECTED[name]
    qc = {}
    for lam in LAMBDA_TAGS:
        e = _cached_sae_eval(exp, lam)
        if e is not None:
            qc[lam] = {"fvu": e["FVU"], "l0_frac": e["L0_pct"] / 100.0, "dead_frac": e["DFR"]}
    if len(qc) < 5:
        pytest.skip(f"{name}: cached QC available for {sorted(qc)} only")
    chosen, rule = select_lambda(qc, l0_max=0.02, dead_max=0.25)
    sel = cached_path("sae_v2", "sparse_selection.json")
    recorded = json.load(open(sel))[f"{exp['dataset']}/{exp['cache_model']}/{exp['layer']}"]["l1"] if sel else None
    report(f"{name} T2 λ", chosen, exp["lambda"], recorded)
    assert chosen == exp["lambda"] == (recorded if recorded is not None else exp["lambda"]), rule
    e1 = qc[1.0]
    report(f"{name} T2 QC at λ=1 (fvu, l0_frac, dead_frac)", (round(e1["fvu"], 4), round(e1["l0_frac"], 5), e1["dead_frac"]))
    assert e1["l0_frac"] <= 0.02 and e1["dead_frac"] <= 0.25


@pytest.mark.parametrize("unit", ["row", "gene"])
@pytest.mark.parametrize("name", LAYERS)
def test_t3_t4_concept_f1_summary_rule_from_cache(name, unit):
    """The concept-F1 summary (best validation feature at τ = 0.15, ties lowest index, mean test F1
    over supported concepts with 5-100 genes) applied to the cached held-out tables gives Table S1."""
    exp = EXPECTED[name]
    sizes = _concept_sizes(exp)
    cached = _cached_concept_f1_summary(exp) or {}
    for rep in ("sae", "raw", "untrained"):
        d = _cached_f1_dir(exp, unit, rep)
        if d is None or not (d / "valid_concept_f1_scores.csv").exists():
            pytest.skip(f"{name}: cached {unit}/{rep} tables not available")
        valid = _dev_table(d / "valid_concept_f1_scores.csv", 0.15)
        test = _dev_table(d / "test_concept_f1_scores.csv", 0.15)
        support = pd.read_csv(d / "concept_support_by_split.csv")
        res = HeldoutResult(valid=valid, test=test, support=support, counts={}, split_meta={})
        summary, _pc = summarize_concept_f1(res, sizes, tau=0.15, size_range=(5, 100))
        src = cached.get((unit, rep))
        check(f"{name} T3/T4 {unit} {rep} concept F1", summary["concept_f1"], exp["concept_f1"][unit][rep],
              None if src is None else float(src.specific_mean), tol=0.0005)
        if src is not None:
            report(f"{name} {unit} {rep} n concepts", summary["n_concepts"], None, int(src.n_specific))
            assert summary["n_concepts"] == int(src.n_specific)


# --- from the finished run directory --------------------------------------------------------------


@pytest.mark.parametrize("name", LAYERS)
def test_t1_tis_gain(paper, name):
    exp = EXPECTED[name]
    rd = paper.run_dir(name, stages=["ingest", "extract", "tis"])
    tis = json.load(open(rd / "03_tis" / "tis.json"))
    layer = tis["layers"][exp["layer"]]
    src = _cached_tis(exp)
    assert layer["n_cells"] == exp["n_cells"] and layer["hidden_dim"] == exp["hidden_dim"]
    assert tis["params"]["trials"] == 800 and tis["params"]["k_ref"] == 32 and tis["params"]["seed"] == 42
    check(f"{name} T1 TIS gain", layer["gain"], exp["tis_gain"],
          None if src is None else src["tis_mean"] - src["shuffle_mean"], tol=0.0005)
    if src is not None:
        check(f"{name} T1 mean TIS", layer["model"]["mean"], None, src["tis_mean"], tol=0.0005)
        check(f"{name} T1 shuffled TIS", layer["shuffled"]["mean"], None, src["shuffle_mean"], tol=0.0005)
        if "pca" in layer and src.get("pca_mean") is not None:
            assert tis["params"]["pca_of"] == "activations"
            # The PCA rotation is a full SVD whose component signs follow scikit-learn's convention, which changed
            # in 1.5 (the manuscript run used 1.3.2); flipped components swap the quantile pools of the seeded
            # trials, so the control's mean moves by about 0.002 while TIS and the shuffled control are identical.
            check(f"{name} T1 PCA-rotated TIS", layer["pca"]["mean"], None, src["pca_mean"], tol=0.005)


@pytest.mark.parametrize("name", LAYERS)
def test_t2_lambda_rule_and_untrained_baseline(paper, name):
    """The sae stage with the cached checkpoint: λ = 1 by the rule, QC within tolerance of the
    cached evaluation, and the untrained SAE bit-identical to the manuscript's randinit control."""
    exp = EXPECTED[name]
    rd = paper.run_dir(name, stages=["ingest", "extract", "sae"])
    sel = json.load(open(rd / "04_sae" / "selected.json"))["layers"][exp["layer"]]
    report(f"{name} T2 λ (stage)", sel["lambda"], exp["lambda"])
    assert sel["lambda"] == exp["lambda"]
    qc = sel["qc"][f"{exp['lambda']:g}"]
    src = _cached_sae_eval(exp, exp["lambda"])
    if src is not None:
        assert sel["n_qc_rows"] == src["eval_rows"], (sel["n_qc_rows"], src["eval_rows"])
        check(f"{name} T2 FVU at λ=1", qc["fvu"], None, src["FVU"], tol=0.005)
        check(f"{name} T2 L0 fraction at λ=1", qc["l0_frac"], None, src["L0_pct"] / 100.0, tol=0.001)
        check(f"{name} T2 dead-feature rate at λ=1", qc["dead_frac"], None, src["DFR"], tol=0.001)
    assert qc["l0_frac"] <= 0.02 and qc["dead_frac"] <= 0.25
    ref = cached_path("sae_v2", exp["dataset"], exp["cache_model"], exp["cache_layer"], "sae", "v2randinit__steps_0",
                      f"sae_{exp['layer']}_best.pt")
    if ref is not None:
        ours = load_checkpoint(sel["untrained"]["checkpoint"])["state_dict"]
        theirs = load_checkpoint(ref)["state_dict"]
        diff = max(float((ours[k] - theirs[k]).abs().max()) for k in ("encoder.weight", "decoder.weight"))
        report(f"{name} T2 untrained SAE max |Δ| vs cached randinit", diff, 0.0)
        assert diff == 0.0


@pytest.mark.parametrize("unit", ["row", "gene"])
@pytest.mark.parametrize("name", LAYERS)
def test_t3_t4_concept_f1(paper, name, unit):
    """Concept F1 of the sparse SAE, raw dimensions and untrained SAE recomputed from the store."""
    exp = EXPECTED[name]
    rd = paper.run_dir(name, stages=["ingest", "extract", "sae", "concepts", "concept_eval"])
    table = pd.read_csv(rd / "06_concept_eval" / "concept_f1.csv")
    cached = _cached_concept_f1_summary(exp) or {}
    for rep in ("sae", "raw", "untrained"):
        row = table[(table.layer == exp["layer"]) & (table.unit == unit) & (table.representation == rep)]
        assert len(row) == 1, f"{name}: no {unit}/{rep} evaluation"
        row = row.iloc[0]
        src = cached.get((unit, rep))
        check(f"{name} T3/T4 {unit} {rep} concept F1", row["concept_f1"], exp["concept_f1"][unit][rep],
              None if src is None else float(src.specific_mean), tol=0.005)
        if src is not None:
            report(f"{name} {unit} {rep} n concepts", int(row["n_concepts"]), None, int(src.n_specific), 2)
            assert abs(int(row["n_concepts"]) - int(src.n_specific)) <= 2
    # the held-out partitions are the manuscript's: same row counts (row split) / same test genes (gene split)
    d = _cached_f1_dir(exp, unit, "sae")
    if d is not None:
        ours = json.load(open(rd / "06_concept_eval" / exp["layer"] / f"f1_{unit}_sae" / "split.json"))
        if unit == "gene" and (d / "heldout_gene_split.json").exists():
            theirs = json.load(open(d / "heldout_gene_split.json"))
            for part in ("test_genes", "valid_genes"):
                if part in theirs:
                    report(f"{name} whole-gene split: {part} identical to the manuscript run", set(ours[part]) == set(theirs[part]), True)
                    assert set(ours[part]) == set(theirs[part]), f"whole-gene split differs from the manuscript run ({part})"
        elif unit == "row" and (d / "counts_summary.json").exists():
            counts = json.load(open(rd / "06_concept_eval" / exp["layer"] / "f1_row_sae" / "counts.json"))
            theirs = json.load(open(d / "counts_summary.json"))
            report(f"{name} row split rows (valid, test)", counts["rows_by_split"], None, theirs["rows_by_split"])
            split = json.load(open(rd / "06_concept_eval" / exp["layer"] / "f1_row_sae" / "split.json"))
            if split.get("rows_dropped_duplicate_cells"):
                # The stage drops duplicate cells before splitting (`dedup_cells: auto`); the cached tables of this
                # layer keep them, so each split holds half the cached rows up to the split draw.
                for part, n in theirs["rows_by_split"].items():
                    assert abs(2 * counts["rows_by_split"][part] - n) <= 0.001 * n, (part, counts["rows_by_split"][part], n)
            else:
                assert counts["rows_by_split"] == theirs["rows_by_split"]


@pytest.mark.parametrize("name", LAYERS)
def test_concept_counts(paper, name):
    """Universe size, concepts scored per held-out unit and concepts marked by sparse features (test F1 >= 0.5)."""
    exp = EXPECTED[name]
    rd = paper.run_dir(name, stages=["ingest", "extract", "sae", "concepts", "concept_eval"])
    uni = json.load(open(rd / "05_concepts" / "universe.json"))
    report(f"{name} concepts in the universe", uni["n_concepts"], exp["n_concepts"])
    if exp["n_concepts"] is not None:
        assert uni["n_concepts"] == exp["n_concepts"]
    assert uni["n_genes"] == exp["n_genes"]
    table = pd.read_csv(rd / "06_concept_eval" / "concept_f1.csv")
    cached = _cached_concept_f1_summary(exp) or {}
    for unit in ("row", "gene"):
        n = int(table[(table.unit == unit) & (table.representation == "sae")].iloc[0]["n_concepts"])
        src = cached.get((unit, "sae"))
        report(f"{name} {unit} concepts scored (5-100 genes, supported)", n, None, None if src is None else int(src.n_specific))
        if src is not None:
            assert abs(n - int(src.n_specific)) <= 2
    pc = pd.read_csv(rd / "06_concept_eval" / exp["layer"] / "f1_row_sae" / "per_concept.csv")
    marked = int(((pc["test_f1"] >= 0.5) & pc["supported"] & pc["in_size_range"]).sum())
    src = cached.get(("row", "sae"))
    expected_marked = None if src is None else int(round(float(src.specific_ge0p5) * int(src.n_specific)))
    lo, hi = exp["marked_concepts_range"]
    label = f"{name} concepts marked at the row split (sparse, test F1 >= 0.5)"
    report(label, marked, f"within {lo}-{hi}", expected_marked, 2)
    assert lo <= marked <= hi
    if expected_marked is not None and known_difference(label) is None:
        assert abs(marked - expected_marked) <= 2


@pytest.mark.parametrize("name", LAYERS)
def test_t5_gene_readout(paper, name):
    """Gene-level readout (Table S3): probe AUROC of the raw state and the sparse SAE against the
    expression and random controls, with the published ordering."""
    exp = EXPECTED[name]
    rd = paper.run_dir(name, stages=["ingest", "extract", "sae", "concepts", "gene_readout"])
    out = json.load(open(rd / "07_gene_readout" / "gene_readout.json"))
    reps = out["representations"]
    assert out["params"]["n_splits"] == 5 and out["params"]["C"] == 1.0
    names = {"hidden": f"hidden|{exp['layer']}", "sae": f"sae|{exp['layer']}", "random": f"random_{exp['hidden_dim']}|-",
             "coexpression": "coexpression|-", "coexpression_pc50": "coexpression_pc50|-", "celltype": "celltype|-",
             "trivial": "trivial|-"}
    cached_std = _cached_readout(exp, "standardize") or {}
    cached_ctr = _cached_readout(exp, "center") or {}
    got = {}
    for key, rep in names.items():
        assert rep in reps, f"{name}: representation {rep} missing from the readout"
        r = reps[rep]
        assert r["n_splits"] == 5
        std, ctr, single = r["mean_probe_test_auroc_standardize_mean"], r["mean_probe_test_auroc_center_mean"], r["mean_single_test_auroc_mean"]
        got[key] = std
        m_std, m_ctr, m_single = exp["gene_readout"][key]
        c_std = cached_std.get(key, (None,))[0]
        c_ctr = cached_ctr.get(key, (None,))[0]
        c_single = cached_std.get(key, (None, None, None))[2]
        if key == "random":
            # A chance-level control: the cached rows are another draw of random vectors than the printed one, so
            # they are reported only; the printed value and the 0.47-0.53 band below are asserted.
            report(f"{name} T5 random cached draw (standardized, center-only, single)", (round(std, 4), round(ctr, 4), round(single, 4)),
                   None, (c_std, c_ctr, c_single))
            c_std = c_ctr = c_single = None
        check(f"{name} T5 {key} probe AUROC (standardized)", std, m_std, c_std, tol=0.03)
        check(f"{name} T5 {key} probe AUROC (center-only)", ctr, m_ctr, c_ctr, tol=0.03)
        check(f"{name} T5 {key} best single dimension AUROC", single, m_single, c_single, tol=0.03)
        report(f"{name} T5 {key} SD over splits (standardized)", round(r["mean_probe_test_auroc_standardize_std"], 4),
               None, None if key not in cached_std else round(cached_std[key][1], 4))
    assert got["hidden"] > got["sae"] > got["random"], "published ordering raw > sparse SAE > random not preserved"
    assert got["hidden"] > max(got["coexpression"], got["coexpression_pc50"], got["celltype"]) > got["random"]
    assert 0.47 <= got["random"] <= 0.53
    report(f"{name} T5 concepts per split (mean)", round(reps[names["hidden"]]["n_concepts_mean"], 1))
