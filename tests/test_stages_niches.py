"""The niche stage on the toy fixture (end to end with the report), the scRNA-seq skip, and unit
tests of the µm conversion, the selection rule, the label-composition baseline and the concept
labelling."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from interpretfm.config import load_config
from interpretfm.manifest import StageManifest
from interpretfm.niche_concepts import (
    bh_fdr,
    cross_model_jaccard,
    match_niches,
    niche_composition,
    niche_feature_tests,
    summarize_jaccard,
    top_concepts,
)
from interpretfm.niches import (
    NicheGrid,
    cluster_niches,
    config_label,
    expression_svd,
    expression_variant,
    label_agreement,
    label_composition,
    niche_sweep,
    niche_validation,
    radius_to_native,
    relabel_by_size,
    seed_stability,
    select_config,
)
from interpretfm.runner import run, stage_dir
from interpretfm.stages.niches import niche_grid
from interpretfm.testing import make_toy_adata, write_toy_run

ALL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers", "niches", "interfaces", "lr_screen", "report"]


# --- fixtures ----------------------------------------------------------------------------------


def _patches(n_cells: int = 400, n_features: int = 6, seed: int = 0):
    """Cells on a grid in three spatial patches; features are patch indicators plus noise."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_cells)))
    coords = np.array([(i % side, i // side) for i in range(n_cells)], dtype=np.float64) * 10.0
    patch = (coords[:, 0] > 0.33 * coords[:, 0].max()).astype(int) + (coords[:, 0] > 0.66 * coords[:, 0].max()).astype(int)
    X = rng.normal(scale=0.3, size=(n_cells, n_features)).astype(np.float32)
    for p in range(3):
        X[patch == p, p] += 2.0
    X = np.clip(X, 0, None)
    return X, coords, patch


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("niches")
    cfg = load_config(write_toy_run(root, store=True, env={"python": sys.executable}))
    rows = run(cfg)
    assert {r["stage"]: r["status"] for r in rows} == {s: "done" for s in ALL_STAGES}
    return cfg


# --- stage on the toy data --------------------------------------------------------------------------


def test_niche_stage_outputs(toy_run):
    d = stage_dir(toy_run.run.out_dir, "niches")
    sel = json.loads((d / "selected.json").read_text())
    assert sel["grid"]["radii_um"] == [12.0, 25.0] and sel["grid"]["radii_native"] == [24.0, 50.0]
    layer = sel["layers"]["layer_1.norm2"]
    assert layer["n_cells"] == 300 and layer["n_features"] >= 3
    chosen = layer["selected"]
    assert chosen["radius_um"] in (12.0, 25.0) and chosen["radius_native"] == chosen["radius_um"] * 2
    assert chosen["method"] == "kmeans" and chosen["k"] in (2, 3) and chosen["rank"] == 1
    val = layer["validation"]
    assert sum(val["sizes"].values()) == 300 and val["min_cluster_size"] >= 5
    assert "ari" in val and "nmi" in val and val["purity_null_std"] > 0
    assert set(val["sizes"]) == {str(i) for i in range(chosen["k"])}
    base = layer["baselines"]
    assert base["expression_svd50"]["winner"]["selected"]["config"] and "ari" in base["expression_svd50"]["at_model_config"]
    assert "ari" in base["expression_svd50"]["at_fixed_configs"]["r12um/m/kmeans/k2"]
    lc = base["label_composition"]
    assert lc["published"]["radius_um"] == 12.0 and lc["published"]["k"] == 2 and lc["at_model_config"]["k"] == chosen["k"]
    fx = layer["fixed_configs"]["r12um/m/kmeans/k2"]
    assert fx["config"]["radius_native"] == 24.0 and fx["in_sweep"] and fx["rank"] >= 1 and "ari" in fx and 0 <= fx["ari_vs_selected"] <= 1
    assert fx["is_winner"] == (chosen["config"] == "r12um/m/kmeans/k2")
    assert layer["stability"]["k"] == [3] and layer["stability"]["model"]["n_configs"] == 4
    assert layer["concepts"]["top_k"] == 3 and all(len(v) <= 3 for v in layer["concepts"]["top"].values())
    ldir = d / "layer_1.norm2"
    sweep = pd.read_csv(ldir / "sweep.csv")
    assert len(sweep) <= 8 and list(sweep["score"]) == sorted(sweep["score"], reverse=True)
    assert set(sweep["radius_um"]) <= {12.0, 25.0} and set(sweep["radius"]) <= {24.0, 50.0}
    labels = pd.read_csv(ldir / "labels.csv")
    assert len(labels) == 300 and labels["niche"].nunique() == chosen["k"] and labels["niche_at_r12um/m/kmeans/k2"].nunique() == 2
    bl = pd.read_csv(ldir / "baselines" / "labels.csv")
    assert {"expression_winner", "expression_at_model_config", "label_composition_published", "label_composition_at_model_config"} <= set(bl.columns)
    tests = pd.read_csv(ldir / "concepts.csv")
    assert {"niche", "feature", "concept", "effect", "pval", "qval", "rank"} <= set(tests.columns)
    assert len(tests) == chosen["k"] * layer["n_features"]
    top = pd.read_csv(ldir / "top_concepts.csv")
    assert (top.groupby("niche").size() <= 3).all() and (top["effect"] > 0).all()
    ftop = pd.read_csv(ldir / "top_concepts_at_r12um_m_kmeans_k2.csv")
    assert set(ftop["niche"]) == {0, 1} and layer["concepts"]["at_fixed_configs"]["r12um/m/kmeans/k2"]["top"]
    # the manifest records both units
    m = StageManifest.read(d)
    assert m.extra["radii_um"] == [12.0, 25.0] and m.extra["radii_native"] == [24.0, 50.0]


def test_report_includes_niches(toy_run):
    r = json.loads((stage_dir(toy_run.run.out_dir, "report") / "report.json").read_text())
    assert r["spatial"] == {"status": "done", "note": "niches: run; interfaces: run; lr_screen: run"}
    n = r["niches"]["layer_1.norm2"]
    assert n["config"] and "ari" in n["model"] and n["baselines"]["expression_winner"]["config"]
    assert n["baselines"]["label_composition"]["config"].startswith("r12um/composition/kmeans/k2")
    assert n["baselines"]["tile_null"]["n_null"] == 3 and n["seed_stability"]["model"] is not None
    page = (stage_dir(toy_run.run.out_dir, "report") / "report.html").read_text()
    assert "Niches" in page and "label composition" in page


def test_scrna_config_skips_niches(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, spatial=False, store=True, env={"python": sys.executable}))
    rows = run(cfg)
    status = {r["stage"]: r["status"] for r in rows}
    assert status["niches"] == "skipped" and status["report"] == "done"
    assert StageManifest.read(stage_dir(cfg.run.out_dir, "niches")).status == "skipped"
    r = json.loads((stage_dir(cfg.run.out_dir, "report") / "report.json").read_text())
    assert r["spatial"]["status"] == "skipped" and r["niches"] == {}


# --- units and grid -------------------------------------------------------------------------------


def test_radius_conversion():
    assert radius_to_native(18.0, 0.18) == 100.0 and radius_to_native(21.6, 0.18) == 120.0 and radius_to_native(27.0, 0.18) == 150.0
    assert radius_to_native(100.0, 1.0) == 100.0
    grid = NicheGrid(radii_um=[18.0, 21.6, 27.0])
    assert grid.radii_native(0.18) == [100.0, 120.0, 150.0]
    d = grid.to_dict(0.18)
    assert d["radii_um"] == [18.0, 21.6, 27.0] and d["radii_native"] == [100.0, 120.0, 150.0] and d["um_per_unit"] == 0.18
    assert grid.sweep_config(0.18, 194).radii == [100.0, 120.0, 150.0] and grid.pca_components(194) == 50 and grid.pca_components(20) == 20
    assert config_label(18.0, "xm", "kmeans", 3, 100.0) == "r18um/xm/kmeans/k3 (r100 native)"
    with pytest.raises(ValueError, match="radii_um"):
        niche_grid({})
    with pytest.raises(ValueError, match="methods"):
        NicheGrid(radii_um=[10.0], methods=["hdbscan"])
    with pytest.raises(ValueError, match="k must"):
        NicheGrid(radii_um=[10.0], k=[1])
    g = niche_grid({"radii_um": [18, 21.6], "k": [3], "methods": ["kmeans"], "score": {"ari": 2.0}})
    assert g.w_ari == 2.0 and g.w_purity_z == 0.1 and g.k == [3]


# --- selection rule -------------------------------------------------------------------------------


def test_selection_rule():
    t = pd.DataFrame({"radius_um": [18, 18, 21.6], "radius": [100, 100, 120], "space": ["m", "xm", "m"], "method": ["kmeans"] * 3,
                      "n_clusters": [3, 4, 3], "ari_mean": [0.9, 0.95, 0.8], "ari_std": [0.0] * 3, "purity_z": [20.0, 5.0, 50.0],
                      "separability": [1.0, 1.2, 1.1], "pca_components": [50] * 3})
    t["score"] = 1.5 * t["ari_mean"] + 0.1 * np.clip(t["purity_z"], -10, 10) + t["separability"]
    t["config"] = [config_label(r.radius_um, r.space, r.method, r.n_clusters) for r in t.itertuples()]
    t = t.sort_values("score", ascending=False).reset_index(drop=True)
    # scores: m/k3/r100 = 1.35 + 1.0 + 1.0 = 3.35; xm/k4 = 1.425 + 0.5 + 1.2 = 3.125; m/k3/r120 = 1.2 + 1.0 + 1.1 = 3.3
    best = select_config(t)
    assert best["space"] == "m" and best["radius"] == 100 and best["rank"] == 1 and best["n_configs"] == 3
    k4 = select_config(t, k=4)
    assert k4["space"] == "xm" and k4["rank"] == 3
    with pytest.raises(ValueError, match="no configuration"):
        select_config(t, k=5)


def test_sweep_scores_follow_the_rule_and_select_the_patches():
    X, coords, patch = _patches()
    grid = NicheGrid(radii_um=[12.0, 30.0], k=[3], methods=["kmeans"], spaces=["m"], seeds=[0, 1], min_cluster=5, n_pcs=4, n_null=3)
    table = niche_sweep(X, coords, grid, um_per_unit=0.5)
    assert set(table["radius"]) <= {24.0, 60.0} and set(table["radius_um"]) <= {12.0, 30.0}
    for r in table.itertuples():
        assert r.score == pytest.approx(1.5 * r.ari_mean + 0.1 * np.clip(r.purity_z, -10, 10) + r.separability)
    best = select_config(table)
    lab, E, neigh = cluster_niches(X, coords, best["radius"], best["space"], best["method"], best["n_clusters"], seed=0, n_pcs=4)
    assert label_agreement(patch, lab)["ari"] > 0.9
    val = niche_validation(lab, neigh, coords, E, tile_size=4 * best["radius"], n_null=3, ref_labels=patch)
    assert val["ari"] > 0.9 and val["purity"] > val["purity_null_mean"] and val["purity_z"] > 0 and val["n_niches"] == 3
    assert val["purity"] == pytest.approx(best["purity"]) and val["purity_null_mean"] == pytest.approx(best["purity_null_mean"])
    sized = relabel_by_size(lab)
    counts = np.bincount(sized)
    assert list(counts) == sorted(counts, reverse=True) and label_agreement(lab, sized)["ari"] == 1.0
    st = seed_stability(table, ks=[3])
    assert st["n_configs"] == len(table) and st["median_seed_ari"] == pytest.approx(float(table["ari_mean"].median()))
    assert st["n_configs_other_k"] == 0 and st["median_seed_ari_other_k"] is None


# --- baselines ------------------------------------------------------------------------------------


def test_label_composition_baseline():
    X, coords, patch = _patches()
    labels = np.array([f"t{p}" for p in patch])
    rng = np.random.default_rng(1)
    noisy = labels.copy()
    flip = rng.random(len(labels)) < 0.1
    noisy[flip] = rng.choice(["t0", "t1", "t2"], size=int(flip.sum()))
    lab, comp = label_composition(noisy, coords, radius_native=25.0, k=3, n_pcs=2, n_init=5)
    assert comp.shape == (len(labels), 3) and list(comp.columns) == ["t0", "t1", "t2"]
    rows = comp.sum(1).to_numpy()
    assert np.allclose(rows[rows > 0], 1.0)
    # the composition excludes the cell itself: an isolated cell has a zero row
    far = np.vstack([coords, [[1e6, 1e6]]])
    _, comp2 = label_composition(np.append(noisy, "t0"), far, 25.0, 3, n_pcs=2, n_init=2)
    assert comp2.iloc[-1].sum() == 0
    assert label_agreement(patch, lab)["ari"] > 0.8
    assert len(set(lab)) == 3


def test_expression_features():
    adata = make_toy_adata(n_cells=120, n_genes=30)
    F = expression_svd(adata.X, n_components=10, seed=0)
    assert F.shape == (120, 10) and F.dtype == np.float32
    for v in ("cp10k_scaled", "cp10k_unscaled", "cp10k_svd", "median_scaled"):
        G = expression_variant(adata.X, v, n_components=10)
        assert G.shape == (120, 10)
    assert np.allclose(expression_variant(adata.X, "cp10k_svd", 10), F)
    with pytest.raises(ValueError, match="variant"):
        expression_variant(adata.X, "nope")


# --- concept labels -------------------------------------------------------------------------------


def test_bh_fdr():
    q = bh_fdr(np.array([0.01, 0.04, 0.03, np.nan, 0.5]))
    # four valid p-values: 0.01 * 4/1, min(0.03 * 4/2, 0.04 * 4/3), 0.04 * 4/3, 0.5
    assert np.isnan(q[3]) and q[0] == pytest.approx(0.04) and q[1] == pytest.approx(0.04 * 4 / 3)
    assert q[2] == pytest.approx(0.04 * 4 / 3) and q[4] == 0.5
    assert np.isnan(bh_fdr(np.array([np.nan]))).all()


def test_niche_concept_labels_and_cross_model_jaccard():
    X, coords, patch = _patches(n_features=6)
    feats = pd.DataFrame({"feature": range(6), "concept": [f"GO:000{i}" for i in range(6)]})
    tests = niche_feature_tests(X, patch, feats, min_cells=5)
    assert len(tests) == 3 * 6 and set(tests["niche"]) == {0, 1, 2}
    for n in range(3):
        t = tests[tests["niche"] == n].sort_values("rank")
        assert t.iloc[0]["concept"] == f"GO:000{n}" and t.iloc[0]["qval"] < 1e-6 and t.iloc[0]["effect"] > 1.0
        assert list(t["rank"]) == list(range(1, 7))
    top = top_concepts(tests, k=2)
    assert (top.groupby("niche").size() <= 2).all() and (top["effect"] > 0).all() and top["rank"].min() == 1
    assert [top[top["niche"] == n].iloc[0]["concept"] for n in range(3)] == ["GO:0000", "GO:0001", "GO:0002"]
    by_score = top_concepts(tests, k=2, rank_by="score")
    assert by_score.iloc[0]["concept"] == top.iloc[0]["concept"]
    assert len(top_concepts(tests, k=2, max_q=1e-30)) <= 6
    with pytest.raises(ValueError, match="rank_by"):
        top_concepts(tests, rank_by="pval")
    # a second "model": the same patches with permuted niche ids and one concept renamed
    perm = np.array([2, 0, 1])[patch]
    feats_b = feats.copy()
    feats_b.loc[5, "concept"] = "GO:0099"
    top_b = top_concepts(niche_feature_tests(X, perm, feats_b, min_cells=5), k=2)
    ref = np.array([f"t{p}" for p in patch])
    mapping, cost = match_niches(niche_composition(patch, ref), niche_composition(perm, ref))
    assert mapping == {"0": "2", "1": "0", "2": "1"} and cost.shape == (3, 3)
    pairs = cross_model_jaccard(top, top_b, mapping)
    assert len(pairs) == 9 and pairs["matched"].sum() == 3
    s = summarize_jaccard(pairs)
    assert s["n_matched"] == 3 and all(j > 0 for j in s["matched_jaccard"]) and s["matched_mean_jaccard"] >= s["unmatched_mean_jaccard"]
    same = cross_model_jaccard(top, top)
    assert same[same["matched"]]["jaccard"].tolist() == [1.0, 1.0, 1.0]


# --- per-stage label override -------------------------------------------------------------------------


def test_stage_label_override(toy_run, tmp_path):
    """``stages.<niches|interfaces|lr_screen>.labels`` (the keys of ``dataset.labels``) replaces the
    dataset labels for that stage only: the niche stage reports the override's name, its ARI
    changes with the labels, the override enters the stage hash, and the label-free stages keep theirs."""
    import copy
    import shutil

    import anndata as ad

    from interpretfm.ingest import dataset_labels, stage_labels
    from interpretfm.runner import run_stage_inprocess

    cfg = toy_run
    rd = cfg.run.out_dir
    a = ad.read_h5ad(cfg.dataset.h5ad, backed="r")
    rng = np.random.default_rng(3)
    names = [str(o) for o in a.obs_names]
    other = pd.DataFrame({"cell_id": names, "shuffled": rng.permutation(a.obs["cell_type"].astype(str).to_numpy())}).set_index("cell_id")
    a.file.close()
    csv = tmp_path / "other_labels.csv"
    other.to_csv(csv)
    assert stage_labels(cfg, cfg.stage_params("niches"), "niches") is cfg.dataset.labels
    cfg2 = copy.deepcopy(cfg)
    cfg2.stages["niches"] = dict(cfg2.stages.get("niches", {}), labels={"column": "shuffled", "file": str(csv), "file_column": "shuffled"})
    lab = stage_labels(cfg2, cfg2.stage_params("niches"), "niches")
    assert lab.column == "shuffled" and lab.file == csv and lab is not cfg2.dataset.labels
    assert cfg2.stage_hash("niches") != cfg.stage_hash("niches") and cfg2.stage_hash("cells") == cfg.stage_hash("cells")
    values = dataset_labels(lab, ad.read_h5ad(cfg.dataset.h5ad))
    assert list(values) == other["shuffled"].tolist()
    rd2 = tmp_path / "override"
    shutil.copytree(rd, rd2, symlinks=True, ignore=shutil.ignore_patterns("10_niches", "11_interfaces", "12_lr_screen", "13_report"))
    cfg2.run.out_dir = rd2
    run_stage_inprocess(cfg2, "niches")
    layer = cfg.model.layers[0]
    v1 = json.load(open(stage_dir(rd, "niches") / layer / "validation.json"))
    v2 = json.load(open(rd2 / stage_dir(rd, "niches").name / layer / "validation.json"))
    assert v1["reference_labels"] == "cell_type" and v2["reference_labels"] == "shuffled"
    assert v1["selected"] == v2["selected"] and abs(v1["ari"] - v2["ari"]) > 1e-6
    bad = copy.deepcopy(cfg)
    bad.stages["niches"] = dict(bad.stages.get("niches", {}), labels={"file": str(csv)})
    with pytest.raises(Exception, match="column"):
        stage_labels(bad, bad.stage_params("niches"), "niches")


def test_cross_model_recipe_on_a_run(toy_run):
    """`read_niche_run` reads a run's niche labels and top concepts; `cross_model_concepts` on the same run twice
    gives the identity matching and Jaccard 1 for every matched niche."""
    from interpretfm.niche_concepts import cross_model_concepts, read_niche_run

    cfg = toy_run
    layer = cfg.model.layers[0]
    top, lab = read_niche_run(cfg.run.out_dir, layer)
    assert set(top.columns) >= {"niche", "concept", "rank"} and lab.index.name == "cell" and len(lab) > 0
    cells = pd.read_csv(stage_dir(cfg.run.out_dir, "niches").parent / "01_ingest" / "cells.csv", dtype={"cell": str}).set_index("cell")
    table = cross_model_concepts({"a": (top, lab), "b": (top, lab)}, cells["label"].astype(str))
    assert len(table) == 1 and table.iloc[0]["model_a"] == "a" and table.iloc[0]["n_cells"] == len(lab)
    assert all(abs(j - 1.0) < 1e-12 for j in table.iloc[0]["jaccard"]) and table.iloc[0]["matched_mean_jaccard"] == pytest.approx(1.0)
    assert all(f"{n}->{n}" in table.iloc[0]["matching"] for n in lab.unique())
    with pytest.raises(KeyError, match="no column"):
        read_niche_run(cfg.run.out_dir, layer, "r1um/m/kmeans/k9")
