"""Cell-level stages (ingest checks, cells, layers, report) on the toy fixture, plus unit tests of the
retention rule, GSI excess, the MHR split logic, the unit conversion and the ingest checks."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from interpretfm.cells import (
    RetentionRule,
    SpecificityParams,
    gini_columns,
    gsi_excess,
    label_shuffles,
    leading_features,
    log_normalized_expression,
    mhr_split_half,
    population_markers,
    select_features,
    specificity,
    split_half,
)
from interpretfm.cli import main as cli_main
from interpretfm.config import load_config
from interpretfm.ingest import IngestError, check_raw_counts, qc_flags, vocab_coverage
from interpretfm.layers import recommendations
from interpretfm.runner import StageError, run, stage_dir
from interpretfm.testing import TOY_UM_PER_UNIT, make_toy_adata, write_toy_run
from interpretfm.units import coords_to_um, native_to_um, um_to_native

ALL_BUT_SPATIAL = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers", "report"]


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """The toy pipeline ingest -> report (spatial stays a stub) with 20 duplicated cells and a vocabulary."""
    root = tmp_path_factory.mktemp("toy")
    yaml_path = write_toy_run(root, store=True, env={"python": sys.executable}, duplicate_cells=20)
    vocab = {f"G{i:03d}": i for i in range(45)}
    vocab["<cls>"] = 60695
    (root / "data" / "vocab.json").write_text(json.dumps(vocab))
    d = yaml.safe_load(open(yaml_path))
    d["model"]["vocab"] = "data/vocab.json"
    yaml.safe_dump(d, open(yaml_path, "w"), sort_keys=False)
    cfg = load_config(yaml_path)
    rows = run(cfg, stages=ALL_BUT_SPATIAL)
    assert [r["status"] for r in rows] == ["done"] * len(ALL_BUT_SPATIAL)
    return cfg, cfg.run.out_dir


# --- end to end ---------------------------------------------------------------------------------


def test_ingest_checks_coverage_flags_and_cell_table(toy_run):
    cfg, run_dir = toy_run
    qc = json.load(open(run_dir / "01_ingest" / "dataset_qc.json"))
    assert qc["counts"]["layer"] == "X" and qc["qc"]["n_flagged"] + qc["qc"]["n_pass"] == 300
    cov = qc["vocab_coverage"]
    assert cov["n_in_vocab"] == 45 and cov["n_genes"] == 50 and cov["source"] == "model.vocab"
    cells = pd.read_csv(run_dir / "01_ingest" / "cells.csv")
    assert list(cells.columns) == ["cell", "n_transcripts", "n_genes", "qc_pass", "label", "x_um", "y_um"]
    assert len(cells) == 300 and cells["qc_pass"].dtype == bool
    coverage = pd.read_csv(run_dir / "01_ingest" / "vocab_coverage.csv")
    assert coverage["in_vocab"].sum() == 45
    import anndata as ad

    native = ad.read_h5ad(cfg.dataset.h5ad).obsm["spatial"]
    assert np.allclose(cells[["x_um", "y_um"]].values, native * TOY_UM_PER_UNIT)


def test_cells_stage_retention_projection_and_specificity(toy_run):
    cfg, run_dir = toy_run
    d = run_dir / "08_cells"
    sel = json.load(open(d / "selection.json"))
    layer = sel["layers"]["layer_1.norm2"]
    assert sel["rule"]["tau"] == 0.15 and sel["rule"]["min_valid_f1"] == 0.4 and sel["rule"]["min_features"] == 3
    assert layer["n_features"] >= 3 and layer["enough_features"] and layer["n_features"] <= layer["n_concepts_kept"]
    feats = pd.read_csv(d / "layer_1.norm2" / "features.csv")
    assert feats["feature"].is_unique and (feats["valid_f1"] >= 0.4).all()
    assert feats["valid_f1"].is_monotonic_decreasing
    import anndata as ad

    a = ad.read_h5ad(d / "layer_1.norm2" / "cells.h5ad")
    assert a.shape == (300, len(feats)) and list(a.var_names) == list(feats["var_name"])
    assert a.obs["has_row"].all() and "qc_pass" in a.obs and "cell_type" in a.obs
    assert a.obsm["spatial_um"].shape == (300, 2) and (a.X >= 0).all()
    proj = json.load(open(d / "layer_1.norm2" / "projection.json"))
    assert proj["n_rows_dropped_later_copy"] > 0 and proj["n_rows_special"] == 300  # CLS rows excluded, duplicates dropped
    spec = json.load(open(d / "specificity.json"))["layers"]["layer_1.norm2"]
    assert spec["n_populations"] == 3 and spec["n_cells"] == 300
    for rep in ("sae", "raw", "genes"):
        e = spec["representations"][rep]
        assert -1 <= e["gsi"]["excess_mean"] <= 1 and 0 <= e["mhr"]["mhr"] <= 1 and e["mhr"]["chance"] == 10 / 50
        assert e["mhr"]["null_mean"] is not None
    assert spec["representations"]["raw"]["n_features"] == 32  # 16 hidden dims x (+, -)
    gf = pd.read_csv(d / "layer_1.norm2" / "gsi_features.csv")
    assert len(gf) == 32 and gf["retained"].sum() == len(feats)
    per_pop = pd.read_csv(d / "layer_1.norm2" / "mhr_by_population.csv")
    assert set(per_pop["representation"]) == {"sae", "raw", "genes"} and len(per_pop) == 9


def test_layers_stage_reports_every_criterion_separately(toy_run):
    cfg, run_dir = toy_run
    table = pd.read_csv(run_dir / "09_layers" / "layer_table.csv")
    assert len(table) == 1
    for col in ("tis_gain", "tis_shuffled_mean", "concept_f1_row_sae", "concept_f1_row_raw", "concept_f1_row_untrained",
                "concept_f1_gene_sae", "concept_f1_gene_raw", "readout_auroc_hidden", "readout_auroc_random",
                "readout_auroc_expression_control", "gsi_excess_sae", "gsi_excess_raw", "gsi_excess_genes", "mhr_sae",
                "mhr_raw", "mhr_genes", "mhr_chance", "n_features_retained"):
        assert col in table.columns and table[col].notna().all(), col
    assert not any("composite" in c for c in table.columns)
    rec = json.load(open(run_dir / "09_layers" / "recommendations.json"))
    assert len(rec["criteria"]) == 7 and all(e["best_layer"] == "layer_1.norm2" for e in rec["criteria"])
    assert rec["criteria"][1]["baselines"]["concept_f1_row_raw"] == pytest.approx(table["concept_f1_row_raw"][0])
    notes = (run_dir / "09_layers" / "notes.txt").read_text()
    assert "TIS gain: best at layer_1.norm2" in notes and "no composite" in json.dumps(rec)


def test_report_stage_and_cli(toy_run, capsys):
    cfg, run_dir = toy_run
    out = run_dir / "13_report"
    report = json.load(open(out / "report.json"))
    assert report["spatial"]["status"] == "not run" and report["dataset"]["vocab_coverage"]["fraction"] == 0.9
    assert {m["criterion"] for m in report["metrics"]} == {c["criterion"] for c in report["recommendations"]["criteria"]}
    m = next(x for x in report["metrics"] if x["column"] == "gsi_excess_sae")
    assert set(m["baselines"]) == {"gsi_excess_raw", "gsi_excess_genes"}
    page = (out / "report.html").read_text()
    assert page.startswith("<!doctype html>") and "spatial: not run" in page and "How to read this report" in page
    assert "http://" not in page and "https://" not in page and page.count("data:image/png;base64,") >= 6
    assert "whole-gene split" in page and "single genes" in page
    rc = cli_main(["report", str(run_dir)])
    assert rc == 0 and "report done" in capsys.readouterr().out
    assert cli_main(["report", str(run_dir / "01_ingest")]) == 2


def test_report_needs_layers(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, store=True, env={"python": sys.executable}))
    run(cfg, stages=["ingest"])
    with pytest.raises(StageError):
        run(cfg, stages=["report"])


# --- retention rule -----------------------------------------------------------------------------


def _f1_table(entries, thresholds=(0.0, 0.15)):
    rows = []
    for concept, feature, f1 in entries:
        for t in thresholds:
            rows.append({"concept": concept, "feature": feature, "threshold": t, "f1": f1 if t == 0.15 else 0.0})
    return pd.DataFrame(rows)


def test_select_features_rule_threshold_dedup_and_minimum():
    valid = _f1_table([("C1", 3, 0.9), ("C1", 7, 0.95), ("C2", 7, 0.5), ("C3", 1, 0.39), ("C4", 2, 0.8), ("C5", 9, 0.99)])
    test = _f1_table([("C1", 7, 0.9), ("C2", 7, 0.4), ("C4", 2, 0.7), ("C5", 9, 0.95)])
    sizes = pd.Series({"C1": 10, "C2": 50, "C3": 20, "C4": 5, "C5": 200})  # C5 too large
    best, kept, feats, info = select_features(valid, test, sizes, RetentionRule(min_features=3))
    assert info["n_eligible_concepts"] == 4 and info["n_eligible_with_valid_rows"] == 4
    assert set(kept["concept"]) == {"C1", "C2", "C4"}  # C3 below 0.40, C5 not eligible
    assert best.set_index("concept").loc["C1", "feature"] == 7  # best validation F1 per concept
    assert list(feats["feature"]) == [7, 2] and feats.iloc[0]["concept"] == "C1"  # dedup: labelled by best concept
    assert feats.iloc[0]["n_concepts_sharing_feature"] == 2 and feats.iloc[0]["test_f1"] == 0.9
    assert info["n_features"] == 2 and not info["enough_features"]
    assert list(feats["var_name"]) == ["C1|feat7", "C4|feat2"]
    _, _, feats2, info2 = select_features(valid, test, sizes, RetentionRule(min_features=2), term_names={"C1": "T cell"})
    assert info2["enough_features"] and feats2.iloc[0]["var_name"] == "T cell|C1|feat7"


def test_select_features_ties_take_lowest_feature_index():
    valid = _f1_table([("C1", 5, 0.6), ("C1", 2, 0.6)])
    test = _f1_table([("C1", 2, 0.5), ("C1", 5, 0.5)])
    _, _, feats, _ = select_features(valid, test, pd.Series({"C1": 8}), RetentionRule())
    assert list(feats["feature"]) == [2]


# --- GSI ----------------------------------------------------------------------------------------


def test_gini_and_gsi_excess():
    assert float(gini_columns(torch.tensor([[1.0], [1.0], [1.0], [1.0]]))) == pytest.approx(0.0)
    assert float(gini_columns(torch.tensor([[0.0], [0.0], [0.0], [4.0]]))) == pytest.approx(0.75)
    assert torch.isnan(gini_columns(torch.zeros(3, 1))).all()
    rng = np.random.default_rng(0)
    n_pop, per = 4, 200
    lab = torch.as_tensor(np.repeat(np.arange(n_pop), per))
    specific = np.zeros(n_pop * per, dtype=np.float32)
    specific[lab.numpy() == 2] = 1.0  # fires in one population only
    flat = rng.random(n_pop * per).astype(np.float32)  # independent of the population
    feat = torch.as_tensor(np.column_stack([specific, flat]))
    shuffles = label_shuffles(len(lab), 5, rng)
    res = gsi_excess(feat, lab, n_pop, shuffles)
    assert res["gini"][0] == pytest.approx(0.75) and res["excess"][0] > 0.6
    assert abs(res["excess"][1]) < 0.05 and res["summary"]["n_features"] == 2


# --- MHR split logic ----------------------------------------------------------------------------


def test_mhr_split_half_is_stratified_and_markers_from_half_one():
    rng = np.random.default_rng(0)
    lab = np.repeat(np.arange(3), [40, 60, 31])
    half = split_half(lab, 3, rng)
    for p, n in enumerate([40, 60, 31]):
        assert half[lab == p].sum() == n // 2
    # expression: gene g is a marker of population g % 3 (blocks of 10 genes per population)
    n, G = len(lab), 30
    expr = rng.normal(size=(n, G)).astype(np.float32)
    for p in range(3):
        expr[np.ix_(lab == p, np.arange(p * 10, (p + 1) * 10))] += 3.0
    E = torch.as_tensor(expr)
    L = torch.as_tensor(lab)
    A = torch.as_tensor(half)
    markers = population_markers(E[A], L[A], 3, 10)
    for p in range(3):
        assert set(markers[p].tolist()) == set(range(p * 10, (p + 1) * 10))
    with pytest.raises(ValueError):
        population_markers(E[A], L[A], 3, 31)
    # features = the one-hot populations: leading feature of population p is p, MHR = 1 on half 2
    feat = torch.nn.functional.one_hot(L, 3).float() + 0.01 * torch.rand(n, 3)
    lead = leading_features(feat[A], L[A], 3, 1)
    assert lead.tolist() == [[0, 1, 2]]
    res = mhr_split_half(feat, E, L, 3, half, SpecificityParams(mhr_k=10, n_leading=1, n_assoc=10, n_null=3), rng)
    assert res["mhr"] == pytest.approx(1.0) and res["chance"] == pytest.approx(10 / 30)
    assert 0 <= res["null_mean"] <= 1
    # the single-gene representation scores itself: the leading gene of a population is one of its markers
    res_g = mhr_split_half(E - E.min(), E, L, 3, half, SpecificityParams(mhr_k=10, n_leading=3, n_assoc=10, n_null=0), rng)
    assert res_g["mhr"] > 0.8


def test_specificity_drops_small_populations_and_shares_cells():
    rng = np.random.default_rng(1)
    labels = np.array(["a"] * 50 + ["b"] * 50 + ["c"] * 5)
    expr = rng.random((105, 20)).astype(np.float32)
    feat = rng.random((105, 6)).astype(np.float32)
    feat[:, 5] = 0.0  # inactive feature
    res = specificity({"sae": feat, "genes": expr}, expr, labels, SpecificityParams(min_pop=30, mhr_k=5, n_assoc=5, n_null=2))
    assert res["n_populations"] == 2 and res["n_cells"] == 100 and "c" not in res["populations"]
    assert res["representations"]["sae"]["n_active"] == 5 and np.isnan(res["per_feature"]["sae"]["gini"][5])
    assert len(res["per_population"]) == 4
    res2 = specificity({"sae": feat}, expr, labels, SpecificityParams(min_pop=60, mhr_k=5, n_assoc=5))
    assert "skipped" in res2


def test_log_normalized_expression():
    X = np.array([[1, 1, 2], [0, 0, 0]], dtype=np.float32)
    L = log_normalized_expression(X, target_sum=4)
    assert np.allclose(L[0], np.log1p([1, 1, 2])) and (L[1] == 0).all()


# --- units and ingest checks --------------------------------------------------------------------


def test_unit_conversion_round_trip():
    coords = np.array([[100.0, 200.0], [0.0, 50.0]])
    um = coords_to_um(coords, 0.18)
    assert np.allclose(um, coords * 0.18)
    assert um_to_native(native_to_um(652.8, 0.18), 0.18) == pytest.approx(652.8)
    assert native_to_um(652.8, 0.18) == pytest.approx(117.504)
    with pytest.raises(ValueError):
        coords_to_um(coords, 0)


def test_ingest_rejects_normalized_counts_and_flags_cells():
    from scipy import sparse

    counts = np.array([[3, 0, 2], [1, 1, 0]], dtype=np.float32)
    assert check_raw_counts(counts)["checked_cells"] == 2
    assert check_raw_counts(sparse.csr_matrix(counts))["sparse"]
    with pytest.raises(IngestError, match="not integers"):
        check_raw_counts(np.log1p(counts))
    with pytest.raises(IngestError, match="negative"):
        check_raw_counts(counts - 5)
    passing = qc_flags(np.array([25.0, 5.0, 30.0]), np.array([12, 12, 3]), 20, 10)
    assert passing.tolist() == [True, False, False]


def test_ingest_stage_errors_name_the_key(tmp_path):
    yaml_path = write_toy_run(tmp_path)
    cfg = load_config(yaml_path)
    import anndata as ad

    adata = ad.read_h5ad(cfg.dataset.h5ad)
    adata.X = np.log1p(adata.X)
    adata.write_h5ad(cfg.dataset.h5ad)
    with pytest.raises(StageError, match="counts_layer"):
        run(cfg, stages=["ingest"])
    d = yaml.safe_load(open(yaml_path))
    d["dataset"]["labels"] = {"column": "missing"}
    make_toy_adata().write_h5ad(cfg.dataset.h5ad)
    yaml.safe_dump(d, open(yaml_path, "w"), sort_keys=False)
    with pytest.raises(StageError, match="dataset.labels.column"):
        run(load_config(yaml_path), stages=["ingest"], force=True)


def test_vocab_coverage_matches_symbols_and_aliases():
    from interpretfm.config import GeneId
    from interpretfm.genes import GeneTable

    adata = make_toy_adata(n_cells=5, n_genes=4, spatial=False)
    adata.var["ensembl_id"] = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    genes = GeneTable.from_adata(adata, GeneId(column="index", kind="symbol"))
    df, s = vocab_coverage(genes, ["G000", "g001", "ENSG3", "OTHER"])
    assert df["in_vocab"].tolist() == [True, True, True, False] and s["fraction"] == 0.75
    assert s["missing_examples"] == ["G003"]


def test_ingest_without_vocab_reports_why(tmp_path):
    cfg = load_config(write_toy_run(tmp_path))
    run(cfg, stages=["ingest"])
    qc = json.load(open(stage_dir(cfg.run.out_dir, "ingest") / "dataset_qc.json"))
    assert qc["vocab_coverage"]["fraction"] is None and "model.vocab" in qc["vocab_coverage"]["note"]
