"""The interfaces stage on the toy fixture (end to end with the report), the scRNA-seq skip, and unit
tests of the µm conversion, a planted contact signal versus a smooth contact-free field, the group
map and label normalisation, the within-label residualisation, and the cross-model recipes."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from interpretfm.config import load_config, parse_config
from interpretfm.ingest import dataset_labels, label_group_map
from interpretfm.interfaces import (
    DEFAULT_SEEDS,
    InterfaceParams,
    PairTest,
    adjust_score,
    block_bootstrap,
    calibrate,
    covariates,
    group_labels,
    interface_test,
    load_group_map,
    marker_excluded_calls,
    neighbour_corr,
    norm_label,
    partner_marker_tercile,
    partner_markers,
    program_score,
    smooth_field,
)
from interpretfm.manifest import StageManifest
from interpretfm.runner import run, stage_dir
from interpretfm.stages.interfaces import interface_params, read_interface_run
from interpretfm.testing import make_toy_adata, toy_config_dict, write_toy_run

ALL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers", "niches", "interfaces", "lr_screen", "report"]


# --- fixtures ----------------------------------------------------------------------------------


def _two_halves(side: int = 40, spacing: float = 10.0, seed: int = 0):
    """Cells on a grid; group A on the left half, B on the right, so that only the cells along the
    boundary have cross-group edges; exact labels split each group into a top and a bottom half."""
    rng = np.random.default_rng(seed)
    ij = np.array([(i, j) for i in range(side) for j in range(side)])
    coords = ij * spacing + rng.normal(scale=0.5, size=ij.shape)
    group = np.where(ij[:, 0] < side // 2, "A", "B").astype(object)
    exact = np.array([f"{g}{j // (side // 2)}" for g, (_i, j) in zip(group, ij)], dtype=object)
    return coords, group, exact


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("interfaces")
    cfg = load_config(write_toy_run(root, store=True, env={"python": sys.executable}))
    rows = run(cfg)
    assert {r["stage"]: r["status"] for r in rows} == {s: "done" for s in ALL_STAGES}
    return cfg


# --- stage on the toy data --------------------------------------------------------------------------


def test_interface_stage_outputs(toy_run):
    d = stage_dir(toy_run.run.out_dir, "interfaces")
    assert d.name == "11_interfaces"
    s = json.loads((d / "summary.json").read_text())
    p = s["params"]
    assert p["edge_radius"] == 24.0 and p["block"] == 50.0 and p["um"]["edge_radius"] == pytest.approx(12.0) and p["um"]["block"] == pytest.approx(25.0)
    assert p["legacy_tile"] == pytest.approx(50.0 * 10 / 3) and len(p["length_scales"]) == 3 and p["seeds"] == DEFAULT_SEEDS
    assert s["niche_column"] == "niche" and s["labels"]["n_exact"] == 3 and s["labels"]["n_groups"] == 3 and s["covariates"] == ["log_counts", "log_genes"]
    assert s["markers"]["n_labels"] == 3 and (d / "markers.csv").exists() and (d / "markers_by_group.csv").exists()
    L = s["layers"]["layer_1.norm2"]
    assert L["n_cells"] == 300 and L["n_pairs_niche"] >= 1 and L["n_pairs_free"] >= 1 and L["n_niches"] >= 2
    cal = L["calibration"]
    assert cal["zstar_niche"] >= 2.0 and cal["zstar_free"] >= 2.0 and len(cal["fields"]) == 3 and {"v1", "v2n", "v2f", "iid_noise_v2n"} <= set(cal)
    assert set(L["n_supported_niche"]) <= {"enriched", "depleted", "not supported"} and 0 <= L["frac_absz_gt2_v1"] <= 1
    row = L["row"]
    assert row["n_pairs_free"] == L["n_pairs_free"] and row["zstar_niche"] == cal["zstar_niche"]
    ldir = d / "layer_1.norm2"
    niche = pd.read_csv(ldir / "niche_pairs.csv")
    assert {"model", "niche", "group_a", "group_b", "n_edges", "z_v1", "effect_pct_v1", "z_v1_adjusted_score", "z_v2", "effect_pct_v2",
            "p_upper_v2", "p_lower_v2", "zstar_v2", "call_v2", "mean_edge_P"} <= set(niche.columns)
    assert len(niche) == L["n_pairs_niche"] and (niche["n_edges"] >= 5).all() and set(niche["model"]) == {"scgpt"}
    free = pd.read_csv(ldir / "free_pairs.csv")
    assert {"group_a", "group_b", "n_edges", "z_v1_free", "z_v2_free", "effect_pct_v2_free", "zstar_v2_free", "call_v2_free"} <= set(free.columns)
    assert len(free) == L["n_pairs_free"] and "niche" not in free.columns
    sc = pd.read_csv(ldir / "scores.csv")
    assert list(sc.columns) == ["cell", "P", "P_adj", "exact", "group", "niche"] and len(sc) == 300 and (sc["P"] >= 0).all()
    fo = pd.read_csv(ldir / "fired_on.csv")
    assert {"feature", "var_name", "fired_on", "n_fired", "n_rate", "n_mass", "fired_on_symbol"} <= set(fo.columns) and len(fo) == L["n_features"]
    assert L["fired_on"]["n_rows"] > 0 and len(L["fired_on"]["shards_used"]) <= 2
    m = StageManifest.read(d)
    assert m.extra["edge_radius_native"] == 24.0 and m.extra["block_native"] == 50.0 and m.extra["niche_column"] == "niche"
    assert m.extra["layers"]["layer_1.norm2"]["n_pairs_niche"] == L["n_pairs_niche"]


def test_report_includes_interfaces(toy_run):
    r = json.loads((stage_dir(toy_run.run.out_dir, "report") / "report.json").read_text())
    assert r["spatial"]["note"] == "niches: run; interfaces: run; lr_screen: run"
    e = r["interfaces"]["layer_1.norm2"]
    assert set(e["share_abs_z_gt2"]) == {"legacy_tile", "calibrated"} and e["z_star"]["niche"] >= 2.0 and e["z_star"]["free"] >= 2.0
    assert set(e["fpr"]) == {"smooth_field_legacy", "smooth_field_calibrated", "smooth_field_free", "iid_noise"}
    assert set(e["calls"]["niche"]) == {"enriched", "depleted", "not_supported"} and "median_abs" in e["effect_pct"] and "max_abs" in e["effect_pct"]
    assert e["units"]["edge_radius_native"] == 24.0 and e["units"]["edge_radius"] == pytest.approx(12.0)
    page = (stage_dir(toy_run.run.out_dir, "report") / "report.html").read_text()
    assert "Interfaces" in page and "tile null" in page


def test_scrna_config_skips_interfaces(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, spatial=False, store=True, env={"python": sys.executable}))
    steps = run(cfg, dry_run=True)
    by = {s["stage"]: s for s in steps}
    assert by["interfaces"]["action"] == "skip-na" and by["niches"]["action"] == "skip-na" and by["report"]["action"] == "run"


def test_recipes_on_the_toy_run(toy_run):
    """The two cross-model recipes run on one model's outputs (with a call forced so that the
    tercile and the marker-excluded re-test have a pair to work on)."""
    import anndata as ad

    rd = toy_run.run.out_dir
    R = read_interface_run(rd, "layer_1.norm2")
    free = R["free"].copy()
    free.loc[0, "call_v2_free"] = "enriched"
    a = ad.read_h5ad(toy_run.dataset.h5ad)
    cells = R["scores"].index
    a = a[cells].copy()
    exact, group = group_labels(a.obs["cell_type"].astype(str).to_numpy(), label_group_map(toy_run.dataset))
    cov = covariates(a.X)
    params = InterfaceParams.from_dict(R["summary"]["params"])   # the run's own parameters (native units)
    assert params.edge_radius == 24.0 and params.block == 50.0 and params.calib_reps == 1
    mk = {r.group: set(str(r.markers).split(";")) for r in R["markers_by_group"].itertuples()}
    rows, summ = partner_marker_tercile({"toy": free}, {"toy": R["scores"]["P_adj"].to_numpy()}, a.obsm["spatial"], exact, group, a.X,
                                        [str(g) for g in a.var_names], mk, params, raw_scores={"toy": R["scores"]["P"].to_numpy()}, n_perm=20, boot=50)
    assert len(rows) == 1 and rows["pair"].iloc[0] == f"{free.loc[0, 'group_a']}|{free.loc[0, 'group_b']}" and "g" in rows.columns
    assert summ.iloc[0]["primary"] and summ.iloc[0]["n_pairs"] == 1 and "verdict" in summ.columns
    feats = ad.read_h5ad(rd / "08_cells" / "layer_1.norm2" / "cells.h5ad")
    feats = feats[cells].copy()
    fired = [set(str(x).split(";")) if isinstance(x, str) and x else set() for x in R["fired_on"]["fired_on"]]
    res = marker_excluded_calls({"toy": free}, {"toy": feats.X}, {"toy": fired}, a.obsm["spatial"], exact, group, cov, a.X,
                                [str(g) for g in a.var_names], mk, params, n_perm_tercile=20, boot=50, min_models=1)
    assert abs(res["controls"]["toy"]["max_abs_diff_z_v2_free"]) < 1e-9
    assert res["controls"]["toy"]["zstar"] == pytest.approx(res["controls"]["toy"]["zstar_table"])
    V = res["verdicts"]
    assert len(V) == 1 and set(V.columns) >= {"pair", "orig_sign", "n_models_holding", "removed_per_model", "z_red", "zstar_red", "spillover_robust"}
    pm = res["pairs_by_model"]
    assert pm["n_removed"].iloc[0] + pm["n_kept"].iloc[0] == feats.n_vars and {"g_red", "g_full", "holds"} <= set(pm.columns)


# --- units ----------------------------------------------------------------------------------------


def test_um_conversion():
    p = interface_params({"edge_radius_um": 21.6, "block_um": 21.6, "legacy_tile_um": 72.0, "length_scales_um": [10.8, 27.0, 72.0]}, 0.18)
    assert p.edge_radius == 120.0 and p.block == 120.0 and p.legacy_tile == 400.0 and list(p.length_scales) == [60.0, 150.0, 400.0]
    d = p.to_dict()
    assert d["um"]["edge_radius"] == pytest.approx(21.6) and d["um"]["length_scales"] == pytest.approx([10.8, 27.0, 72.0]) and d["um_per_unit"] == 0.18
    q = interface_params({"edge_radius_um": 120.0}, 1.0)   # Xenium-like: coordinates in µm; defaults follow the paper's ratios
    assert q.edge_radius == 120.0 and q.block == 120.0 and q.legacy_tile == 400.0 and list(q.length_scales) == [60.0, 150.0, 400.0]
    assert q.n_perm == 100 and q.calib_perm == 50 and q.calib_reps == 2 and q.legacy_perm == 20 and q.min_edges == 50
    r = interface_params({"edge_radius_um": 12.0, "seeds": {"niche": 7}}, 0.5)
    assert r.edge_radius == 24.0 and r.seed("niche") == 7 and r.seed("free") == DEFAULT_SEEDS["free"]
    with pytest.raises(ValueError, match="edge_radius_um"):
        interface_params({}, 0.18)
    with pytest.raises(ValueError, match="seeds"):
        interface_params({"edge_radius_um": 12.0, "seeds": {"nope": 1}}, 0.5)
    assert InterfaceParams.from_um(0.18, edge_radius_um=21.6).edge_radius == 120.0


# --- planted signal vs smooth field ---------------------------------------------------------------------


def test_planted_contact_signal_is_called_and_smooth_field_is_not():
    coords, group, exact = _two_halves()
    rng = np.random.default_rng(1)
    n = len(coords)
    params = InterfaceParams(edge_radius=12.0, block=40.0, legacy_tile=120.0, length_scales=[20.0, 50.0, 120.0], n_perm=50, calib_perm=20,
                             calib_reps=1, legacy_perm=10, min_edges=20)
    cov = pd.DataFrame({"log_counts": rng.normal(size=n), "log_genes": rng.normal(size=n), "log_area": rng.normal(size=n)})
    niche = np.where(coords[:, 1] > coords[:, 1].mean(), "0", "1")
    # contact signal: cells with a cross-group neighbour (the boundary column of each half) score higher than
    # the interior cells of the same label in the same block, which the within-stratum null redistributes
    T = PairTest(coords, group, exact, niche=niche, radius=params.edge_radius, min_edges=params.min_edges)
    contact = np.zeros(n)
    contact[np.unique(np.concatenate([T.e0, T.e1]))] = 1.0
    X = np.column_stack([contact + rng.normal(scale=0.3, size=n), rng.normal(scale=0.3, size=n)]).astype(np.float32)
    res = interface_test(X, coords, exact, group, niche, cov, params, model="toy")
    niche_t, free_t = res["niche"], res["free"]
    assert set(zip(free_t["group_a"], free_t["group_b"])) == {("A", "B")} and len(niche_t) == 2 and set(niche_t["niche"]) == {"0", "1"}
    assert (free_t["call_v2_free"] == "enriched").all() and (free_t["z_v2_free"] > 3).all()
    assert (niche_t["call_v2"] == "enriched").all() and (niche_t["z_v2"] > 3).all() and (niche_t["effect_pct_v2"] > 0).all()
    # a smooth contact-free field at a scale well above the block is not called above z*
    S = smooth_field(coords, 200.0, np.random.default_rng(3))
    Xs = np.column_stack([np.maximum(S, 0) + 0.05 * rng.normal(size=n), 0.05 * rng.normal(size=n)]).astype(np.float32)
    res2 = interface_test(Xs, coords, exact, group, niche, cov, params, model="toy")
    assert (res2["niche"]["call_v2"] == "not supported").all() and (res2["free"]["call_v2_free"] == "not supported").all()
    cal = res2["summary"]["calibration"]
    assert cal["zstar_niche"] >= 2.0 and 0 <= cal["v2n"]["fpr_abs_z_gt2"] <= 1 and len(cal["fields"]) == 3
    assert abs(neighbour_corr(S, res2["edges"])) > 0.5   # the field is smooth at the edge scale


def test_pair_test_mechanics():
    coords, group, exact = _two_halves(side=12)
    T = PairTest(coords, group, exact, niche=None, radius=12.0, min_edges=5)
    assert T.npairs == 1 and T.pairs.iloc[0]["n_edges"] == len(T.e0) and (group[T.e0] != group[T.e1]).all()
    v = np.random.default_rng(0).normal(size=len(coords))
    sid = T.strata(40.0, use_niche=False)
    perm = T.permutation(sid, np.random.default_rng(0))
    assert (sid[perm] == sid).all() and sorted(perm) == list(range(len(v)))   # a within-stratum permutation
    r1 = T.run(v, tile=40.0, n_perm=10, seed=5)
    r2 = T.null(v, T.permutations(sid, 10, 5))
    assert np.allclose(r1.z, r2.z) and np.allclose(r1.null_mean, r2.null_mean) and r1.obs[0] == pytest.approx(T.stat(v)[0])
    assert 0 < r1.p_upper[0] <= 1 and 0 < r1.p_lower[0] <= 1
    Te = PairTest(coords, np.where(group == "A", "low_quality", "B"), exact, niche=None, radius=12.0, min_edges=5)
    assert Te.npairs == 0


# --- labels ---------------------------------------------------------------------------------------


def test_group_map_and_label_normalization(tmp_path):
    assert norm_label(" T CD8 memory ") == "t_cd8_memory" and norm_label("B-cell/plasma") == "b_cell_plasma"
    csv = tmp_path / "groups.csv"
    csv.write_text("label,group\ntumor 13,tumor\nTumor 5,tumor\nB-cell,b_plasma\n")
    gm = load_group_map(csv)
    assert gm == {"tumor_13": "tumor", "tumor_5": "tumor", "b_cell": "b_plasma"}
    exact, group = group_labels(["tumor 13", "B-cell", "macrophage", "Tumor 5"], gm)
    assert list(exact) == ["tumor_13", "b_cell", "macrophage", "tumor_5"] and list(group) == ["tumor", "b_plasma", "macrophage", "tumor"]
    e2, g2 = group_labels(["T CD4", "x"], None)
    assert list(e2) == ["t_cd4", "x"] and list(g2) == list(e2)
    e3, g3 = group_labels(["T CD4"], {"T CD4": "T"}, normalize=False)
    assert list(e3) == ["T CD4"] and list(g3) == ["T"]
    csv2 = tmp_path / "bad.csv"
    csv2.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="group"):
        load_group_map(csv2)
    # configuration: groups CSV + inline group_map, and a per-cell labels file
    adata = make_toy_adata(n_cells=40, n_genes=10)
    h5 = tmp_path / "toy.h5ad"
    adata.write_h5ad(h5)
    lab_file = tmp_path / "labels.csv"
    pd.DataFrame({"cell": adata.obs_names, "label": ["L" + str(i % 2) for i in range(adata.n_obs)]}).to_csv(lab_file, index=False)
    raw = toy_config_dict(h5, tmp_path / "run")
    raw["dataset"]["labels"] = {"column": "cell_type", "groups": str(csv), "group_map": {"type_0": "immune", "type_1": "immune"}}
    cfg = parse_config(raw, base=tmp_path)
    gm = label_group_map(cfg.dataset)
    assert gm["tumor_13"] == "tumor" and gm["type_0"] == "immune" and cfg.dataset.labels.has_groups
    assert list(dataset_labels(cfg.dataset, adata)[:2]) == list(adata.obs["cell_type"].astype(str)[:2])
    raw["dataset"]["labels"] = {"column": "marker_label", "file": str(lab_file), "file_column": "label"}
    cfg = parse_config(raw, base=tmp_path)
    labs = dataset_labels(cfg.dataset, adata)
    assert list(labs[:3]) == ["L0", "L1", "L0"] and cfg.dataset.labels.file == lab_file
    pd.DataFrame({"cell": adata.obs_names[:-1], "label": "x"}).to_csv(lab_file, index=False)
    with pytest.raises(ValueError, match="missing from"):
        dataset_labels(cfg.dataset, adata)


# --- residualisation --------------------------------------------------------------------------------


def test_residualization_within_exact_label():
    rng = np.random.default_rng(0)
    n = 600
    exact = np.array(["a"] * 300 + ["b"] * 300, dtype=object)
    cov = pd.DataFrame({"log_counts": rng.normal(size=n), "log_genes": rng.normal(size=n), "log_area": rng.normal(size=n)})
    slope = np.where(exact == "a", 2.0, -1.0)
    P = 1.0 + slope * cov["log_counts"].to_numpy() + 0.5 * cov["log_area"].to_numpy() + rng.normal(scale=0.1, size=n)
    R = adjust_score(P, exact, cov)
    for lab in ("a", "b"):
        m = exact == lab
        assert abs(R[m].mean()) < 1e-9
        for c in ("log_counts", "log_genes", "log_area"):
            assert abs(np.corrcoef(R[m], cov[c].to_numpy()[m])[0, 1]) < 1e-6
    assert np.std(R) < 0.2   # the label-specific slopes are removed
    R2 = adjust_score(P, exact, cov[["log_counts", "log_genes"]])   # without an area column: two covariates
    assert np.std(R2) > np.std(R)
    X = rng.gamma(2.0, size=(n, 5)).astype(np.float32)
    X[:, 0] *= 100
    s = program_score(X)
    assert s.shape == (n,) and (s >= 0).all() and s.dtype == np.float64


# --- markers, calibration helpers, bootstrap ------------------------------------------------------------


def test_partner_markers_on_toy_types():
    adata = make_toy_adata(n_cells=240, n_genes=30, n_types=3, seed=2)
    M = partner_markers(adata.X, [str(g) for g in adata.var_names], adata.obs["cell_type"].astype(str).to_numpy(), top_n=4)
    assert set(M["label"]) == {"type_0", "type_1", "type_2"} and (M.groupby("label").size() == 4).all() and list(M.columns)[:4] == ["label", "n_cells", "rank", "gene"]
    for t in range(3):
        top = set(M[M["label"] == f"type_{t}"]["gene"])
        block = {f"G{i:03d}" for i in range(t * 10, (t + 1) * 10)}
        assert len(top & block) >= 3   # each type over-expresses its own gene block


def test_calibration_and_bootstrap_helpers():
    coords, group, exact = _two_halves(side=20)
    params = InterfaceParams(edge_radius=12.0, block=40.0, legacy_tile=120.0, length_scales=[20.0, 60.0], n_perm=10, calib_perm=5, calib_reps=1,
                             legacy_perm=5, min_edges=5)
    Tn = PairTest(coords, group, exact, niche=np.zeros(len(coords), dtype=int), radius=12.0, min_edges=5)
    Tf = PairTest(coords, group, exact, niche=None, radius=12.0, min_edges=5)
    rng = np.random.default_rng(0)
    v = rng.normal(size=len(coords))
    cal = calibrate(Tn, Tf, v, v, Tn.all_edges, params)
    assert len(cal["fields"]) == 2 and all(0 < f["signal_fraction"] <= 1 for f in cal["fields"]) and {"60", "20"} <= set(cal["v2n"]["by_L"])
    assert cal["zstar_niche"] == max(2.0, cal["v2n"]["q95_abs_z"]) and cal["zstar_free"] == max(2.0, cal["v2f"]["q95_abs_z"])
    rows = pd.DataFrame({"pair": ["a|b"] * 3 + ["c|d"] * 3, "g": [0.5, 0.6, 0.4, -0.2, -0.1, -0.3]})
    b = block_bootstrap(rows, "g", B=200, seed=0)
    assert b["n_rows"] == 6 and b["n_pairs"] == 2 and b["ci_lo"] <= b["median_g"] <= b["ci_hi"]
