"""The ligand–receptor screen on the toy fixture (end to end with the report), the scRNA-seq skip, and unit
tests of a planted ligand–receptor contact (a hit) versus a label-only pattern (not a hit), complex
resolution, the matched-neighbour radius, the HTML page with and without the spatial stages, and the
label-aware stage hash."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

from interpretfm.config import load_config, parse_config
from interpretfm.lr_screen import (
    ScreenParams,
    load_cellchatdb,
    neighbour_graph,
    niche_placement,
    panel_pairs,
    resolve_genes,
    screen_pairs,
    screen_summary,
    within_type_permutations,
)
from interpretfm.manifest import StageManifest
from interpretfm.neighbours import interpolate_radius, match_radius, mean_other_cells, radius_curve
from interpretfm.report import collect, make_figures, render_html
from interpretfm.runner import run, stage_dir, stage_uses_labels
from interpretfm.stages.lr_screen import resolve_radius, screen_params
from interpretfm.testing import make_toy_adata, make_toy_lr_database, toy_config_dict, write_toy_run

ALL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers", "niches", "interfaces", "lr_screen", "report"]
SECTION_HEADERS = ["How to read this report", "Dataset and input checks", "Layers: every criterion next to its baselines",
                   "Spatial analyses", "Niches", "Interfaces", "Ligand–receptor screen", "Stages"]


# --- fixtures ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("lr_screen")
    cfg = load_config(write_toy_run(root, store=True, env={"python": sys.executable}))
    rows = run(cfg)
    assert {r["stage"]: r["status"] for r in rows} == {s: "done" for s in ALL_STAGES}
    return cfg


def _grid(side: int = 40, spacing: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    ij = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
    coords = ij * spacing + rng.normal(scale=0.3, size=ij.shape)
    types = np.where(ij[:, 0] < side / 2, "A", "B").astype(object)
    return coords, types


def _planted(seed: int = 0, side: int = 40):
    """Two types on a grid. Gene L is expressed in a random half of the A cells; gene R in the B cells that touch an
    L-positive A cell (a planted contact) plus a random background; gene M marks type B only (a label-only pattern:
    every B cell, so its co-localization with L is what the within-type null expects)."""
    coords, types = _grid(side=side, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(coords)
    A = neighbour_graph(coords, 12.0)
    L = (types == "A") & (rng.random(n) < 0.5)
    touch = (A @ L.astype(np.float32)) > 0
    R = (types == "B") & (touch | (rng.random(n) < 0.15))
    M = types == "B"
    X = np.zeros((n, 4), dtype=np.float32)
    X[:, 0] = L * rng.integers(1, 5, size=n)
    X[:, 1] = R * rng.integers(1, 5, size=n)
    X[:, 2] = M * rng.integers(1, 5, size=n)
    X[:, 3] = rng.poisson(0.3, size=n)
    return X, ["L", "R", "M", "N"], types, coords


# --- stage on the toy data --------------------------------------------------------------------------


def test_lr_screen_stage_outputs(toy_run):
    d = stage_dir(toy_run.run.out_dir, "lr_screen")
    assert d.name == "12_lr_screen" and stage_dir(toy_run.run.out_dir, "report").name == "13_report"
    s = json.loads((d / "summary.json").read_text())
    assert s["radius_um"] == 12.0 and s["radius_native"] == 24.0 and s["um_per_unit"] == 0.5 and s["matched"] is None
    assert s["params"]["thresholds"] == [1, 2] and s["params"]["n_perm"] == 20 and s["params"]["seed"] == 0 and s["params"]["min_positive"] == 10
    assert s["database"]["packaged"] is False and s["database"]["n_rows"] == 8 and s["database"]["n_complexes"] == 1
    assert s["n_pairs_on_panel"] == 7 and s["n_cells"] == 300 and s["n_edges"] > 0 and s["mean_other_cells"] == pytest.approx(2 * s["n_edges"] / 300)
    assert s["labels"] == {"column": "cell_type", "file": None, "n": 3} and s["niche_column"] == "niche"
    assert set(s["summary"]["thresholds"]) == {"1", "2"} and all(v["n_tested"] <= 7 for v in s["summary"]["thresholds"].values())
    pairs = pd.read_csv(d / "pairs.csv")
    assert len(pairs) == 7 and "NOTONPANEL" not in set(pairs["ligand"]) and (pairs["receptor"] == "RC").sum() == 1
    assert pairs.loc[pairs["receptor"] == "RC", "receptor_genes"].iloc[0].count(";") == 1   # the complex resolved to two subunits
    sc = pd.read_csv(d / "screen.csv")
    cols = {f"{c}_thr{t}" for t in (1, 2) for c in ("n_lig_pos", "n_rec_pos", "tested", "obs", "null_mean", "null_sd", "fold", "z", "p", "p_emp_upper", "q", "hit")}
    assert cols <= set(sc.columns) and {"robust_hit", "depleted_thr1", "top_celltype_pairs"} <= set(sc.columns) and len(sc) == 7
    assert sc["z_thr1"].is_monotonic_decreasing or sc["z_thr1"].isna().any()
    L = s["layers"]["layer_1.norm2"]
    assert L["niche_column"] == "niche" and L["n_niches"] >= 2 and L["n_cells_with_niche"] == 300 and (d / "layer_1.norm2" / "placement.csv").exists()
    m = StageManifest.read(d)
    assert m.extra["radius_native"] == 24.0 and m.extra["n_pairs"] == 7 and set(m.extra["n_hits"]) == {"1", "2"}


def test_report_includes_lr_screen(toy_run):
    r = json.loads((stage_dir(toy_run.run.out_dir, "report") / "report.json").read_text())
    assert r["spatial"]["note"] == "niches: run; interfaces: run; lr_screen: run"
    e = r["lr_screen"]
    assert e["radius_um"] == 12.0 and e["primary"] == "1" and set(e["tested"]) == {"1", "2"} and len(e["screen_rows"]) == 7
    assert "layer_1.norm2" in e["placement"] and set(r["section_notes"]) >= {"niches", "interfaces", "lr_screen", "layers"}
    page = (stage_dir(toy_run.run.out_dir, "report") / "report.html").read_text()
    assert page.startswith("<!doctype html>") and all(h in page for h in SECTION_HEADERS)
    assert "model-free" in page.lower() and "http://" not in page and "https://" not in page


def test_scrna_config_skips_lr_screen_and_report_says_not_run(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, spatial=False, store=True, env={"python": sys.executable}))
    steps = run(cfg, dry_run=True)
    by = {s["stage"]: s for s in steps}
    assert by["lr_screen"]["action"] == "skip-na" and by["niches"]["action"] == "skip-na" and by["interfaces"]["action"] == "skip-na"
    rows = run(cfg, stages=["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers", "niches", "interfaces",
                            "lr_screen", "report"])
    st = {r["stage"]: r["status"] for r in rows}
    assert st["lr_screen"] == "skipped" and st["report"] == "done"
    report = collect(cfg, cfg.run.out_dir)
    assert report["lr_screen"] == {} and report["spatial"]["note"] == "skipped: the dataset has no spatial coordinates"
    page = render_html(report, make_figures(report))
    assert all(h in page for h in SECTION_HEADERS) and "ligand–receptor screen: skipped: the dataset has no spatial coordinates" in page
    assert "niches: skipped" in page and "interfaces: skipped" in page


def test_html_without_spatial_stages_run(toy_run, tmp_path):
    """A report built from a run whose spatial stages were never run renders the sections as 'not run'."""
    cfg = copy.deepcopy(toy_run)
    rd = tmp_path / "partial"
    rd.mkdir()
    for name in ("01_ingest", "09_layers"):
        (rd / name).symlink_to(toy_run.run.out_dir / name)
    report = collect(cfg, rd)
    assert report["niches"] == {} and report["interfaces"] == {} and report["lr_screen"] == {}
    page = render_html(report, make_figures(report))
    assert all(h in page for h in SECTION_HEADERS) and "ligand–receptor screen: not run" in page and "niches: not run" in page


# --- planted contact versus a label-only pattern --------------------------------------------------------


def test_planted_contact_is_a_hit_and_label_pattern_is_not():
    X, genes, types, coords = _planted()
    db = pd.DataFrame([{"interaction_name": "L_R", "pathway_name": "contact", "ligand": "L", "receptor": "R", "annotation": ""},
                       {"interaction_name": "L_M", "pathway_name": "label", "ligand": "L", "receptor": "M", "annotation": ""},
                       {"interaction_name": "L_N", "pathway_name": "noise", "ligand": "L", "receptor": "N", "annotation": ""}])
    pairs = panel_pairs(db, {}, genes)
    params = ScreenParams(radius=12.0, thresholds=[1, 2], n_perm=50, seed=3, min_positive=20)
    res = screen_pairs(X, genes, types, coords, pairs, params).set_index("interaction_name")
    assert bool(res.at["L_R", "hit_thr1"]) and res.at["L_R", "fold_thr1"] > 1.5 and res.at["L_R", "z_thr1"] > 5 and res.at["L_R", "q_thr1"] < 1e-6
    assert not bool(res.at["L_M", "hit_thr1"]) and abs(res.at["L_M", "fold_thr1"] - 1.0) < 0.02   # every B cell is M-positive: nothing to permute
    assert not bool(res.at["L_N", "hit_thr1"]) and not bool(res.at["L_N", "depleted_thr1"])
    assert bool(res.at["L_R", "robust_hit"]) == bool(res.at["L_R", "hit_thr2"])
    summ = screen_summary(res.reset_index(), params)
    assert summ["thresholds"]["1"]["hits"] == ["L_R"] and summ["n_depleted"] == 0 and summ["n_pairs"] == 3
    # the null keeps every cell within its type
    P_L, P_R = within_type_permutations(types, 5, 0)
    assert P_L.shape == (5, len(types)) and all((types[p] == types).all() and sorted(p) == list(range(len(types))) for p in np.vstack([P_L, P_R]))
    # placement: the contact pairs live where the A/B boundary is; a niche covering the boundary has ratio > 1
    A = neighbour_graph(coords, 12.0)
    niche = np.where(np.abs(coords[:, 0] - coords[:, 0].mean()) < 25, "boundary", "interior").astype(object)
    niche[: len(niche) // 20] = None   # some cells without a niche label stay in the denominators
    P = niche_placement(res.reset_index(), X, genes, A, niche, [0], thr=1)
    assert set(P["niche"]) == {"boundary", "interior"} and P.set_index("niche").at["boundary", "ratio"] > 1.5
    assert P["share_cells"].sum() < 1.0 and abs(P["share_pairs"].sum() - 1.0) < 0.2


def test_screen_is_deterministic_and_rejects_bad_parameters():
    X, genes, types, coords = _planted(seed=4, side=20)
    db = pd.DataFrame([{"interaction_name": "L_R", "pathway_name": "", "ligand": "L", "receptor": "R", "annotation": ""}])
    pairs = panel_pairs(db, {}, genes)
    params = ScreenParams(radius=12.0, thresholds=[1], n_perm=10, seed=7, min_positive=5)
    a = screen_pairs(X, genes, types, coords, pairs, params)
    b = screen_pairs(X, genes, types, coords, pairs, params)
    assert a["z_thr1"].iloc[0] == b["z_thr1"].iloc[0] and "robust_hit" in a.columns
    with pytest.raises(ValueError, match="radius"):
        ScreenParams(radius=0.0)
    with pytest.raises(ValueError, match="fold_min"):
        ScreenParams(radius=1.0, fold_min=1.0)
    with pytest.raises(ValueError, match="exactly one of radius_um"):
        resolve_radius({}, 0.5, coords)
    with pytest.raises(ValueError, match="exactly one of radius_um"):
        resolve_radius({"radius_um": 12.0, "match_neighbors_to": 5.0}, 0.5, coords)
    assert resolve_radius({"radius_um": 12.0}, 0.5, coords) == {"radius_native": 24.0, "radius_um": 12.0, "matched": None}
    p = screen_params({"thresholds": [1, 3], "seed": 7}, 100.0)
    assert p.radius == 100.0 and p.thresholds == [1, 3] and p.seed == 7 and p.n_perm == 100 and p.min_positive == 100


# --- complexes and the database -----------------------------------------------------------------------


def test_complex_resolution_and_panel_pairs():
    assert resolve_genes("ITGA1_ITGB1", {"ITGA1_ITGB1": ["ITGA1", "ITGB1"]}) == ["ITGA1", "ITGB1"]
    assert resolve_genes("A;B", {}) == ["A", "B"] and resolve_genes("CXCL9", {}) == ["CXCL9"]
    db = pd.DataFrame([{"interaction_name": "CXCL9_CXCR3", "pathway_name": "CXCL", "ligand": "CXCL9", "receptor": "CXCR3", "annotation": "Secreted"},
                       {"interaction_name": "CXCL9_CXCR3_dup", "pathway_name": "CXCL", "ligand": "CXCL9", "receptor": "CXCR3", "annotation": "Secreted"},
                       {"interaction_name": "COL4A1_ITGA1_ITGB1", "pathway_name": "Collagen", "ligand": "COL4A1", "receptor": "ITGA1_ITGB1", "annotation": "ECM"},
                       {"interaction_name": "COL4A1_ITGA2_ITGB1", "pathway_name": "Collagen", "ligand": "COL4A1", "receptor": "ITGA2_ITGB1", "annotation": "ECM"}])
    cx = {"ITGA1_ITGB1": ["ITGA1", "ITGB1"], "ITGA2_ITGB1": ["ITGA2", "ITGB1"]}
    pairs = panel_pairs(db, cx, ["CXCL9", "CXCR3", "COL4A1", "ITGA1", "ITGB1"])
    assert list(pairs["interaction_name"]) == ["CXCL9_CXCR3", "COL4A1_ITGA1_ITGB1"]   # duplicate gene set dropped; ITGA2 not on the panel
    assert pairs.set_index("interaction_name").at["COL4A1_ITGA1_ITGB1", "receptor_genes"] == "ITGA1;ITGB1"
    # the packaged CellChatDB export
    dbp, cmap = load_cellchatdb()
    assert len(dbp) == 3233 and len(cmap) == 338 and cmap["TGFbR1_R2"] == ["TGFBR1", "TGFBR2"] and {"interaction_name", "ligand", "receptor", "pathway_name", "annotation"} <= set(dbp.columns)
    assert cmap["Activin AB"] == ["INHBA", "INHBB"]
    dbn, none = load_cellchatdb(complexes=False)
    assert none == {} and len(dbn) == len(dbp)
    adata = make_toy_adata(n_cells=30, n_genes=20)
    tdb, tcx = make_toy_lr_database(adata)
    assert len(tdb) == 8 and list(tcx["complex_name"]) == ["RC"] and tcx["subunit_3"].iloc[0] == ""


# --- matched-neighbour radius ------------------------------------------------------------------------


def test_match_neighbors_inverts_the_radius_curve():
    coords, _types = _grid(side=30, spacing=10.0, seed=2)
    radii = np.round(np.arange(0.5, 60.0 + 1e-9, 0.5), 4)
    curve = radius_curve(coords, radii)
    assert (np.diff(curve) >= 0).all() and curve[-1] > curve[0] >= 0
    for r in (7.5, 15.0, 31.5):
        assert mean_other_cells(coords, r) == pytest.approx(np.interp(r, radii, curve), abs=1e-9)
    target = mean_other_cells(coords, 21.3)
    m = match_radius(coords, target, step=0.5, round_decimals=1)
    full = round(interpolate_radius(radii, curve, target), 1)   # the same interpolation on the full grid
    assert m["radius"] == full and abs(m["radius"] - 21.3) <= 0.5 and m["target"] == target
    assert m["mean_other_cells"] == pytest.approx(mean_other_cells(coords, m["radius"])) and m["n_grid_evaluations"] < len(radii)
    m2 = match_radius(coords, target, step=0.1, round_decimals=1)
    assert abs(m2["radius"] - 21.3) <= 0.15
    with pytest.raises(ValueError, match="above the mean count"):
        match_radius(coords, 1e6, step=0.5, r_max=30.0)
    with pytest.raises(ValueError, match="outside the curve range"):
        interpolate_radius(radii, curve, curve[-1] + 1)
    rec = resolve_radius({"match_neighbors_to": target, "match_grid_step": 0.5}, 0.5, coords)
    assert rec["radius_native"] == full and rec["radius_um"] == pytest.approx(full * 0.5) and rec["matched"]["target"] == target


# --- stage hash: dataset.labels only for the stages that read labels ----------------------------------------


def test_label_change_keeps_label_free_stage_hashes(tmp_path):
    adata = make_toy_adata(n_cells=40, n_genes=10)
    h5 = tmp_path / "toy.h5ad"
    adata.write_h5ad(h5)
    raw = toy_config_dict(h5, tmp_path / "run")
    cfg = parse_config(raw, base=tmp_path)
    before = {s: cfg.stage_hash(s) for s in ALL_STAGES}
    raw2 = copy.deepcopy(raw)
    raw2["dataset"]["labels"] = {"column": "other", "group_map": {"type_0": "immune"}}
    cfg2 = parse_config(raw2, base=tmp_path)
    after = {s: cfg2.stage_hash(s) for s in ALL_STAGES}
    for s in ("extract", "tis", "sae", "concepts", "concept_eval"):
        assert after[s] == before[s] and not stage_uses_labels(s)
    for s in ("ingest", "gene_readout", "cells", "layers", "niches", "interfaces", "lr_screen", "report"):
        assert after[s] != before[s] and stage_uses_labels(s)
    assert stage_uses_labels("__run__") and cfg2.stage_hash("__run__") != cfg.stage_hash("__run__")
    assert cfg.stage_hash("tis", uses_labels=True) != before["tis"] and cfg.stage_hash("cells", uses_labels=False) != before["cells"]
    raw3 = copy.deepcopy(raw)
    raw3["dataset"]["qc"] = {"min_transcripts": 1, "min_genes": 1}
    cfg3 = parse_config(raw3, base=tmp_path)
    assert cfg3.stage_hash("tis") != before["tis"]   # other dataset keys still count for every stage
