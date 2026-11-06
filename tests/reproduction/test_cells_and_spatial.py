"""T6-T13 (AUDIT §4): the cell-level and spatial numbers of the CosMx scGPT layer.

Two kinds of test, as in ``test_model_stages.py``:

- ``*_from_cache`` / ``*_from_dataset``: seconds to minutes; apply the pipeline's rules
  (feature retention, niche selection, label-composition baseline, niche concept labels) to the
  manuscript's cached intermediates or to the dataset alone and compare with the published numbers;
- the others read the finished run directory of the paper configuration (``make repro`` with
  ``STAGES=...,cells,layers,niches``): T6 features kept, T7 GSI excess and MHR@100, T8 the niche
  sweep winner and its validation, T9 the expression-only and label-composition baselines, T13 the
  niche concept labels shared with the cached Geneformer run.

Tolerances: ±0.005 on F1/GSI/MHR/ARI/NMI, ±1 on counts (±0.01 on purity, ±5 % on the purity z,
whose null has 10 draws). T10-T12 (interface test, LR screen, matched radius) are placeholders for
the next spatial stages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repro_support import cached_path, check, dev_table, report
from expected import EXPECTED, PRIMARY, SPATIAL_COSMX_SCGPT

from interpretfm.cells import RetentionRule, select_features
from interpretfm.niche_concepts import cross_model_jaccard, match_niches, niche_composition, niche_feature_tests, top_concepts
from interpretfm.niches import label_agreement, label_composition, radius_to_native, select_config

CELL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells"]
NICHE_STAGES = CELL_STAGES + ["layers", "niches"]
SPS = ("reconstruction", "spatial_sparse")
SPATIAL_TODO = "TODO: needs the interface and ligand-receptor stages (P2-B/C); expected values are in expected.SPATIAL_COSMX_SCGPT"
GENEFORMER_CACHE = ("geneformer", "layer_1", "r120p0_m_kmeans_k3")


# --- cached sources -------------------------------------------------------------------------------


def _sps_layer(exp):
    return cached_path(*SPS, "runs", exp["dataset"], exp["cache_model"], exp["layer"])


def _cached_validation_dir(model: str, layer: str, cfg_dir: str):
    return cached_path(*SPS, "runs", "cosmx", model, layer, "S", "05_niche_validation", model, layer, cfg_dir)


def _cached_stage_b(exp):
    p = cached_path(*SPS, "tables", "stageB_summary.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    t = t[(t.dataset == exp["dataset"]) & (t.model == exp["cache_model"]) & (t.layer == exp["layer"])]
    return t.iloc[0] if len(t) else None


def _cached_gsi_mhr(exp):
    p = cached_path("revision_metrics", "gsi_mhr_full", f"gsi_mhr_{exp['dataset']}_{exp['cache_model']}_{exp['layer']}_v2biased__steps_8000__lam_1.json")
    return json.load(open(p)) if p else None


def _concept_sizes(exp):
    p = cached_path(exp["gt_dir"], "gprofiler_binary_gene_by_term.csv")
    if p is None:
        pytest.skip("cached concept table not available")
    return pd.read_csv(p, index_col=0).sum(0).astype(int)


def _dataset(exp):
    import anndata as ad
    import os

    p = os.path.join(os.environ["INTERPRETFM_DATA"], "cosmx_human_lung_sec8.h5ad" if exp["dataset"] == "cosmx" else "xenium_rcc_union_filtered_nonzero.h5ad")
    return ad.read_h5ad(p, backed="r")


def _cached_top25(model: str, layer: str):
    p = cached_path(*SPS, "runs", "cosmx", model, layer, "S", "06_niche_terms", "combined_top25_terms_by_model_niche.csv")
    if p is None:
        return None
    t = pd.read_csv(p)
    return pd.DataFrame({"niche": t["niche"].astype(int), "feature": t["feature_id"], "concept": t["term_norm"].astype(str),
                         "rank": t["rank_effect"], "effect": t["effect_size"]})


def _concept_table_from_validation_h5ad(vdir):
    import anndata as ad

    b = ad.read_h5ad(vdir / "adata_with_niche_labels.h5ad")
    feats = pd.DataFrame({"feature": np.arange(b.n_vars), "concept": [str(v).split("|")[0] for v in b.var_names]})
    return b, niche_feature_tests(b.X, b.obs["niche"].astype(int).to_numpy(), feats)


# --- T6 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED))
def test_t6_features_kept_from_cache(name):
    """The retention rule (best validation feature at τ 0.15 ≥ 0.40, concepts with 5-100 genes,
    de-duplicated) on the cached row-split tables gives Table S1 "Features kept"."""
    exp = EXPECTED[name]
    sel = _sps_layer(exp)
    cached = json.load(open(sel / "selection_summary.json")) if sel and (sel / "selection_summary.json").exists() else {}
    d = cached_path("sae_v2", exp["dataset"], exp["cache_model"], exp["cache_layer"], "f1_heldout", "heldout_sparse__gene_token_row")
    if cached.get("f1_dir"):  # the tables the manuscript's selection read (Xenium Geneformer: the de-duplicated rerun)
        parts = Path(cached["f1_dir"]).parts
        d = cached_path(*parts[parts.index("runs") + 1:]) if "runs" in parts else d
    if d is None or not (d / "valid_concept_f1_scores.csv").exists():
        pytest.skip(f"{name}: cached row-split tables not available")
    valid = dev_table(d / "valid_concept_f1_scores.csv", 0.15)
    test = dev_table(d / "test_concept_f1_scores.csv", 0.15)
    _best, kept, feats, info = select_features(valid, test, _concept_sizes(exp), RetentionRule())
    check(f"{name} T6 features kept", info["n_features"], exp["features_kept"], cached.get("n_features"), tol=1)
    if cached:
        check(f"{name} T6 concepts kept", info["n_concepts_kept"], None, cached.get("n_concepts_kept_validF1_ge_0p40"), tol=1)
        report(f"{name} T6 test F1 of the kept pairs (mean)", round(info["test_f1_mean"], 4), None, round(cached["test_f1_label_pairs_mean"], 4))


def test_t6_features_kept(paper):
    exp = EXPECTED[PRIMARY]
    rd = paper.run_dir(PRIMARY, CELL_STAGES)
    sel = json.load(open(rd / "08_cells" / "selection.json"))["layers"][exp["layer"]]
    s = _sps_layer(exp)
    cached = json.load(open(s / "selection_summary.json")) if s and (s / "selection_summary.json").exists() else {}
    check(f"{PRIMARY} T6 features kept (cells stage)", sel["n_features"], exp["features_kept"], cached.get("n_features"), tol=1)
    check(f"{PRIMARY} T6 concepts kept (cells stage)", sel["n_concepts_kept"], 925, cached.get("n_concepts_kept_validF1_ge_0p40"), tol=1)
    assert sel["enough_features"] and (rd / "08_cells" / exp["layer"] / "cells.h5ad").exists()


# --- T7 -------------------------------------------------------------------------------------------


def test_t7_gsi_excess_and_mhr(paper):
    """GSI excess and MHR@100 of sparse features, raw dimensions and single genes (every stored
    cell, populations ≥ 30 cells, 5 label shuffles, seed 0)."""
    exp = EXPECTED[PRIMARY]
    rd = paper.run_dir(PRIMARY, CELL_STAGES)
    spec = json.load(open(rd / "08_cells" / "specificity.json"))["layers"][exp["layer"]]
    reps = spec["representations"]
    cached = _cached_gsi_mhr(exp) or {}
    report(f"{PRIMARY} T7 cells / populations", (spec["n_cells"], spec["n_populations"]), None, (cached.get("n_cells"), cached.get("n_pops")))
    for rep in ("sae", "raw", "genes"):
        check(f"{PRIMARY} T7 GSI excess {rep}", reps[rep]["gsi"]["excess_mean"], exp["gsi_excess"][rep], cached.get(f"GSI_{rep}_excess_mean"))
    for rep in ("sae", "raw", "genes"):
        check(f"{PRIMARY} T7 MHR@100 {rep}", reps[rep]["mhr"]["mhr"], exp["mhr100"][rep], cached.get(f"MHR_K100_{rep}"))
    check(f"{PRIMARY} T7 MHR chance", reps["sae"]["mhr"]["chance"], exp["mhr100"]["chance"], cached.get("MHR_K100_chance"), tol=0.001)


# --- T8 -------------------------------------------------------------------------------------------


def _check_niche_config(label, sel, expected_cfg: str):
    r, space, method, k = expected_cfg.split("/")
    report(f"{label} configuration", sel.get("config_native") or sel.get("config"), expected_cfg)
    assert float(sel["radius_native"]) == float(r[1:]) and sel["space"] == space and sel["method"] == method and int(sel["k"]) == int(k[1:])


def test_t8_niche_selection_rule_from_cache():
    """The score and the selection rule on the cached sweep table pick r100/xm/kmeans/k3 with the
    published seed ARI, purity, tile null and z; the cached labels give the published ARI/NMI."""
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["niche"]
    s = _sps_layer(exp)
    p = s / "S" / "04_niche_sweep" / "combined_niche_sweep_results.csv" if s else None
    if p is None or not p.exists():
        pytest.skip("cached niche sweep table not available")
    t = pd.read_csv(p)
    t["radius_um"] = t["radius"] * 0.18
    t["config"] = [f"r{r.radius_um:g}um/{r.space}/{r.method}/k{r.n_clusters}" for r in t.itertuples()]
    sel = select_config(t)
    chosen = {"radius_native": sel["radius"], "space": sel["space"], "method": sel["method"], "k": sel["n_clusters"],
              "config": sel["config"]}
    _check_niche_config(f"{PRIMARY} T8 (rule on the cached sweep)", chosen, e["config"])
    assert sel["rank"] == 1 and sel["n_configs"] == 36
    check(f"{PRIMARY} T8 seed ARI", sel["ari_mean"], e["seed_ari"])
    check(f"{PRIMARY} T8 purity", sel["purity"], e["purity"], tol=0.01)
    check(f"{PRIMARY} T8 purity null", sel["purity_null_mean"], e["null"], tol=0.01)
    check(f"{PRIMARY} T8 purity z", sel["purity_z"], e["z"], tol=0.05 * e["z"])
    vdir = _cached_validation_dir("scgpt", exp["layer"], "r100p0_xm_kmeans_k3")
    if vdir is not None and (vdir / "global_labels.csv").exists():
        lab = pd.read_csv(vdir / "global_labels.csv", dtype={"cell": str}).set_index("cell")["niche"]
        a = _dataset(exp)
        ref = pd.Series(a.obs["author_cell_type"].astype(str).to_numpy(), index=a.obs_names.astype(str))
        a.file.close()
        agree = label_agreement(ref.reindex(lab.index), lab)
        check(f"{PRIMARY} T8 ARI of the cached labels", agree["ari"], e["ari"])
        check(f"{PRIMARY} T8 NMI of the cached labels", agree["nmi"], e["nmi"])


def test_t8_niche_sweep_winner_and_validation(paper):
    """The ``niches`` stage: the manuscript's configuration (listed in ``fixed_configs``) reproduces
    its seed ARI, purity, tile null, z, ARI and NMI; it is the sweep winner, or within a near-tie
    (score gap below 0.005: the published run's margin over the runner-up was 0.002 and the
    k-means/PCA numerics depend on the scikit-learn version; see docs/reproducing.md)."""
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["niche"]
    rd = paper.run_dir(PRIMARY, NICHE_STAGES)
    L = json.load(open(rd / "10_niches" / "selected.json"))["layers"][exp["layer"]]
    sel = L["selected"]
    stage_b = _cached_stage_b(exp)
    report(f"{PRIMARY} T8 sweep winner (niches stage)", sel["config_native"], e["config"])
    report(f"{PRIMARY} T8 configurations passing min_cluster", sel["n_configs"], 36)
    fixed = [v for v in L["fixed_configs"].values() if v["config"]["radius_um"] == pytest.approx(18.0)
             and v["config"]["space"] == "xm" and v["config"]["method"] == "kmeans" and v["config"]["k"] == 3]
    assert fixed, "the paper configuration must list r18um/xm/kmeans/k3 in stages.niches.fixed_configs"
    f = fixed[0]
    assert f["in_sweep"], "the manuscript configuration was dropped by min_cluster"
    report(f"{PRIMARY} T8 rank and score of the manuscript configuration", (f["rank"], round(f["score"], 4)), (1, None))
    if not f["is_winner"]:
        gap = sel["score"] - f["score"]
        report(f"{PRIMARY} T8 near-tie: winner {sel['config_native']} leads the manuscript configuration by", round(gap, 4), None, None, 0.005)
        assert f["rank"] <= 2 and gap < 0.005, f"the manuscript configuration ranks {f['rank']} (score gap {gap:.4f})"
    check(f"{PRIMARY} T8 ARI", f["ari"], e["ari"], None if stage_b is None else float(stage_b["S_ARI_vs_author"]))
    check(f"{PRIMARY} T8 NMI", f["nmi"], e["nmi"], None if stage_b is None else float(stage_b["S_NMI_vs_author"]))
    check(f"{PRIMARY} T8 seed ARI", f["seed_ari"], e["seed_ari"], None if stage_b is None else float(stage_b["S_winner_seedARI"]))
    check(f"{PRIMARY} T8 purity", f["purity"], e["purity"], tol=0.01)
    check(f"{PRIMARY} T8 purity null", f["purity_null_mean"], e["null"], tol=0.01)
    check(f"{PRIMARY} T8 purity z", f["purity_z"], e["z"], tol=0.05 * e["z"])
    report(f"{PRIMARY} T8 niche sizes", f["sizes"], None, None if stage_b is None else stage_b["S_sizes"])


# --- T9 -------------------------------------------------------------------------------------------


def test_t9_label_composition_from_dataset():
    """The label-composition baseline (author labels, 120 px = 21.6 µm, k = 3, PCA 15, k-means ×20)
    on the dataset alone reproduces its ARI/NMI."""
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["label_composition"]
    a = _dataset(exp)
    labels = a.obs["author_cell_type"].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"])[:, :2]
    a.file.close()
    lab, comp = label_composition(labels, coords, radius_to_native(21.6, 0.18), 3, n_pcs=15, n_init=20, seed=0)
    agree = label_agreement(labels, lab)
    stage_b = _cached_stage_b(exp)
    check(f"{PRIMARY} T9 label composition ARI", agree["ari"], e["ari"], None if stage_b is None else float(stage_b["L_ARI_vs_author"]))
    check(f"{PRIMARY} T9 label composition NMI", agree["nmi"], e["nmi"], None if stage_b is None else float(stage_b["L_NMI_vs_author"]))
    ref = cached_path(*SPS, "runs", "labelcomp", "cosmx_author_r120_k3", "neighborhood_celltype_composition.csv")
    if ref is not None:
        r = pd.read_csv(ref)
        report(f"{PRIMARY} T9 max |Δ| of the composition table vs the cached one", float(np.abs(r[comp.columns].to_numpy() - comp.to_numpy()).max()), None, None, 1e-5)
        assert np.abs(r[comp.columns].to_numpy() - comp.to_numpy()).max() < 1e-5


def test_t9_expression_and_label_composition_baselines(paper):
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["expression_baseline"]
    rd = paper.run_dir(PRIMARY, NICHE_STAGES)
    L = json.load(open(rd / "10_niches" / "selected.json"))["layers"][exp["layer"]]
    stage_b = _cached_stage_b(exp)
    ex = L["baselines"]["expression_svd50"]
    _check_niche_config(f"{PRIMARY} T9 expression-only winner", ex["winner"]["selected"], e["winner"])
    check(f"{PRIMARY} T9 expression-only winner ARI", ex["winner"]["ari"], e["ari"], None if stage_b is None else float(stage_b["E_win_ARI_vs_author"]))
    check(f"{PRIMARY} T9 expression-only winner NMI", ex["winner"]["nmi"], e["nmi"], None if stage_b is None else float(stage_b["E_win_NMI_vs_author"]))
    at = ex["at_model_config"]
    report(f"{PRIMARY} T9 expression at the selected configuration {at['selected']['config_native']} ARI / NMI", (round(at["ari"], 4), round(at["nmi"], 4)))
    at_pub = ex.get("at_fixed_configs", {}).get("r18um/xm/kmeans/k3") or (at if at["selected"]["config"] == "r18um/xm/kmeans/k3" else None)
    assert at_pub is not None, "expression-only niches at the manuscript configuration r18um/xm/kmeans/k3 are missing"
    check(f"{PRIMARY} T9 expression at the scGPT configuration (r100/xm/kmeans/k3) ARI", at_pub["ari"], e["at_scgpt_config"][0],
          None if stage_b is None else float(stage_b["E_match_ARI_vs_author"]))
    check(f"{PRIMARY} T9 expression at the scGPT configuration (r100/xm/kmeans/k3) NMI", at_pub["nmi"], e["at_scgpt_config"][1],
          None if stage_b is None else float(stage_b["E_match_NMI_vs_author"]))
    lc = L["baselines"]["label_composition"]["published"]
    assert lc["radius_native"] == 120.0 and lc["k"] == 3
    el = SPATIAL_COSMX_SCGPT["label_composition"]
    check(f"{PRIMARY} T9 label composition ARI (niches stage)", lc["ari"], el["ari"], None if stage_b is None else float(stage_b["L_ARI_vs_author"]))
    check(f"{PRIMARY} T9 label composition NMI (niches stage)", lc["nmi"], el["nmi"], None if stage_b is None else float(stage_b["L_NMI_vs_author"]))
    lm = L["baselines"]["label_composition"]["at_model_config"]
    report(f"{PRIMARY} T9 label composition at the scGPT configuration ARI / NMI", (round(lm["ari"], 4), round(lm["nmi"], 4)), None,
           None if stage_b is None else (round(float(stage_b["L_match_ARI_vs_author"]), 4), round(float(stage_b["L_match_NMI_vs_author"]), 4)))
    st = L["stability"]
    report(f"{PRIMARY} seed stability (median seed ARI, k 4-5) model vs expression",
           (round(st["model"]["median_seed_ari"], 4), round(st["expression_svd50"]["median_seed_ari"], 4)))


# --- T13 ------------------------------------------------------------------------------------------


def _jaccard_vs_geneformer(top_scgpt, lab_scgpt: pd.Series, ref: pd.Series):
    """Shared top-25 concept ids and Jaccard with the cached Geneformer niches, matched by label composition."""
    gf_model, gf_layer, gf_cfg = GENEFORMER_CACHE
    gf_top = _cached_top25(gf_model, gf_layer)
    gf_dir = _cached_validation_dir(gf_model, gf_layer, gf_cfg)
    if gf_top is None or gf_dir is None or not (gf_dir / "global_labels.csv").exists():
        pytest.skip("cached Geneformer niche run not available")
    lab_gf = pd.read_csv(gf_dir / "global_labels.csv", dtype={"cell": str}).set_index("cell")["niche"].reindex(lab_scgpt.index)
    ok = lab_scgpt.notna() & lab_gf.notna()
    mapping, _cost = match_niches(niche_composition(lab_scgpt[ok], ref[ok]), niche_composition(lab_gf[ok], ref[ok]))
    pairs = cross_model_jaccard(top_scgpt, gf_top, mapping)
    m = pairs[pairs["matched"]].sort_values("n_shared")
    return mapping, m


def test_t13_niche_concept_labels_from_cache():
    """Inside-versus-outside tests on the cached scGPT niches, top 25 by effect, matched to the cached
    Geneformer niches by label composition: 2 / 8 / 8 shared labels, Jaccard 0.042 / 0.190 / 0.190."""
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["niche_concepts"]
    vdir = _cached_validation_dir("scgpt", exp["layer"], "r100p0_xm_kmeans_k3")
    if vdir is None or not (vdir / "adata_with_niche_labels.h5ad").exists():
        pytest.skip("cached scGPT niche run not available")
    b, tests = _concept_table_from_validation_h5ad(vdir)
    top = top_concepts(tests, k=25)
    cached = _cached_top25("scgpt", exp["layer"])
    if cached is not None:
        for n, g in cached.groupby("niche"):
            mine = top[top["niche"] == n].sort_values("rank")["concept"].str.lower().tolist()
            theirs = g.sort_values("rank")["concept"].str.lower().tolist()
            report(f"{PRIMARY} T13 niche {n}: top-25 identical to the cached list (same order)", mine == theirs, None, True)
            assert mine == theirs
    lab = pd.Series(b.obs["niche"].astype(str).to_numpy(), index=b.obs_names.astype(str))
    ref = pd.Series(b.obs["author_cell_type"].astype(str).to_numpy(), index=lab.index)
    mapping, m = _jaccard_vs_geneformer(top, lab, ref)
    report(f"{PRIMARY} T13 niche matching scGPT -> Geneformer (by composition)", mapping)
    shared = tuple(int(v) for v in m["n_shared"])
    report(f"{PRIMARY} T13 shared top-25 labels of the matched niches", shared, tuple(sorted(e["shared_top25"])), None, 1)
    assert all(abs(a - b_) <= 1 for a, b_ in zip(shared, sorted(e["shared_top25"])))
    for j, ej in zip(m["jaccard"], sorted(e["jaccard"])):
        check(f"{PRIMARY} T13 Jaccard", j, ej)


def test_t13_niche_concept_labels(paper):
    exp = EXPECTED[PRIMARY]
    e = SPATIAL_COSMX_SCGPT["niche_concepts"]
    rd = paper.run_dir(PRIMARY, NICHE_STAGES)
    ldir = rd / "10_niches" / exp["layer"]
    L = json.load(open(rd / "10_niches" / "selected.json"))["layers"][exp["layer"]]
    labels = pd.read_csv(ldir / "labels.csv", dtype={"cell": str}).set_index("cell")
    if L["selected"]["config"] == "r18um/xm/kmeans/k3":
        top, lab = pd.read_csv(ldir / "top_concepts.csv"), labels["niche"].astype(str)
    else:  # the sweep picked another configuration (near-tie): use the manuscript configuration's niches
        report(f"{PRIMARY} T13 sweep winner is {L['selected']['config_native']}; using the fixed manuscript configuration", None)
        top = pd.read_csv(ldir / "top_concepts_at_r18um_xm_kmeans_k3.csv")
        lab = labels["niche_at_r18um/xm/kmeans/k3"].astype(str)
    assert (top.groupby("niche").size() == 25).all() and set(top["rank_by"]) == {"effect"}
    cells = pd.read_csv(rd / "01_ingest" / "cells.csv", dtype={"cell": str}, keep_default_na=False).set_index("cell")
    ref = cells["label"].astype(str).reindex(lab.index)
    mapping, m = _jaccard_vs_geneformer(top, lab, ref)
    report(f"{PRIMARY} T13 niche matching scGPT (niches stage) -> cached Geneformer", mapping)
    shared = tuple(int(v) for v in m["n_shared"])
    report(f"{PRIMARY} T13 shared top-25 labels of the matched niches", shared, tuple(sorted(e["shared_top25"])), None, 1)
    assert all(abs(a - b_) <= 1 for a, b_ in zip(shared, sorted(e["shared_top25"])))
    for j, ej in zip(m["jaccard"], sorted(e["jaccard"])):
        check(f"{PRIMARY} T13 Jaccard", j, ej)


# --- T10-T12: placeholders ------------------------------------------------------------------------


@pytest.mark.skip(reason=SPATIAL_TODO)
@pytest.mark.parametrize("key", ["interfaces", "lr_screen", "matched_radius"])
def test_t10_to_t12_spatial(paper, key):
    assert key in SPATIAL_COSMX_SCGPT and PRIMARY in EXPECTED
