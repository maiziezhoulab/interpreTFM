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
whose null has 10 draws). T10 (interface test) runs the library on the cached niche-labelled h5ads of the three
CosMx models and reads the `interfaces` stage of the paper run; the partner-marker tercile and marker-excluded
checks run on the cached calls of the three models. T11-T12 (LR screen, matched radius) are placeholders.
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
from interpretfm.interfaces import (
    InterfaceParams,
    adjust_score,
    cell_area,
    covariates,
    fired_on_sets,
    group_labels,
    interface_test,
    load_group_map,
    marker_excluded_calls,
    markers_by_group,
    partner_marker_tercile,
    partner_markers,
    program_score,
)
from interpretfm.niche_concepts import cross_model_jaccard, match_niches, niche_composition, niche_feature_tests, top_concepts
from interpretfm.niches import label_agreement, label_composition, radius_to_native, select_config

CELL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells"]
NICHE_STAGES = CELL_STAGES + ["layers", "niches"]
SPS = ("reconstruction", "spatial_sparse")
SPATIAL_TODO = "TODO: needs the ligand-receptor screen and matched-radius stage (P2-C); expected values are in expected.SPATIAL_COSMX_SCGPT"
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


# --- T10: calibrated interface test ---------------------------------------------------------------------

INTERFACE_MODELS = {"scgpt": ("layer_1.norm2", "r100p0_xm_kmeans_k3"), "geneformer": ("layer_1", "r120p0_m_kmeans_k3"),
                    "c2sscale": ("layer_13", "r100p0_m_kmeans_k3")}
GROUPS_CSV = Path(__file__).resolve().parents[2] / "configs" / "paper" / "cosmx_author_groups.csv"


def _cosmx_inputs():
    """Dataset-level inputs of the interface test: covariates (bounding-box area), normalised author labels and their groups, symbols."""
    exp = EXPECTED[PRIMARY]
    a = _dataset(exp)
    counts = a.X[:]
    obs = a.obs
    sym = a.var["feature_name"].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"])[:, :2]
    a.file.close()
    cov = covariates(counts, cell_area(obs, {"product": ["Width", "Height"]}))
    exact, group = group_labels(obs["author_cell_type"].astype(str).to_numpy(), load_group_map(GROUPS_CSV))
    return {"counts": counts, "cov": cov, "exact": exact, "group": group, "symbols": sym, "coords": coords,
            "params": InterfaceParams.from_um(0.18, edge_radius_um=21.6, block_um=21.6, legacy_tile_um=72.0, length_scales_um=[10.8, 27.0, 72.0])}


def _cached_validation_adata(model: str):
    layer, cfg_dir = INTERFACE_MODELS[model]
    vdir = _cached_validation_dir(model, layer, cfg_dir)
    if vdir is None or not (vdir / "adata_with_niche_labels.h5ad").exists():
        pytest.skip(f"cached {model} niche-labelled validation h5ad not available")
    import anndata as ad

    return ad.read_h5ad(vdir / "adata_with_niche_labels.h5ad")


def _cached_free_tables():
    out = {}
    for m in INTERFACE_MODELS:
        p = cached_path(*SPS, "tables", f"strict_v2_cosmx_author_{m}_free.csv")
        if p is None:
            pytest.skip("cached strict-v2 niche-free tables not available")
        out[m] = pd.read_csv(p)
    return out


def _expected_interfaces(model: str):
    e = SPATIAL_COSMX_SCGPT["interfaces"]
    return e if model == "scgpt" else SPATIAL_COSMX_SCGPT["interfaces_other_models"][model]


def _check_interface_summary(label: str, s, e, cached=None, tol_pairs: int = 1, tol_share: float = 0.01, tol_effect: float = 0.005):
    """The summary of one model against the manuscript (and the cached summary when given). The effect sizes
    are medians/maxima over the niche-stratified pairs, so on another niche partition they take ``tol_effect``."""
    cal, cs = s["calibration"], (cached or {}).get("calibration") or {}
    check(f"{label} niche pairs", s["n_pairs_niche"], e["n_tests"], (cached or {}).get("n_pairs_niche"), tol=tol_pairs)
    check(f"{label} share |z|>2, legacy tile null", s["frac_absz_gt2_v1"], e["share_abs_z_gt_2_tile"], (cached or {}).get("frac_absz_gt2_v1"), tol=tol_share)
    check(f"{label} share |z|>2, calibrated", s["frac_absz_gt2_v2"], e["share_abs_z_gt_2_calibrated"], (cached or {}).get("frac_absz_gt2_v2"), tol=tol_share)
    check(f"{label} enriched (niche)", s["n_supported_niche"].get("enriched", 0), e["enriched"], ((cached or {}).get("n_supported_niche") or {}).get("enriched", 0), tol=1)
    check(f"{label} depleted (niche)", s["n_supported_niche"].get("depleted", 0), e["depleted"], ((cached or {}).get("n_supported_niche") or {}).get("depleted", 0), tol=1)
    check(f"{label} z*", cal["zstar_niche"], e["z_star"], cs.get("zstar_niche"), tol=0.1)
    nf = e["niche_free"]
    check(f"{label} niche-free pairs", s["n_pairs_free"], nf["n_pairs"], (cached or {}).get("n_pairs_free"), tol=1)
    check(f"{label} enriched (niche-free)", s["n_supported_free"].get("enriched", 0), nf["enriched"], ((cached or {}).get("n_supported_free") or {}).get("enriched", 0), tol=1)
    check(f"{label} depleted (niche-free)", s["n_supported_free"].get("depleted", 0), nf["depleted"], ((cached or {}).get("n_supported_free") or {}).get("depleted", 0), tol=1)
    if "fpr_pct" in e:
        fpr = e["fpr_pct"]
        check(f"{label} FPR smooth field, legacy null (%)", 100 * cal["v1"]["fpr_abs_z_gt2"], fpr[0], 100 * cs["v1"]["fpr_abs_z_gt2"] if cs else None, tol=1.0)
        check(f"{label} FPR smooth field, calibrated null (%)", 100 * cal["v2n"]["fpr_abs_z_gt2"], fpr[1], 100 * cs["v2n"]["fpr_abs_z_gt2"] if cs else None, tol=1.0)
        check(f"{label} FPR i.i.d. noise (%)", 100 * cal["iid_noise_v2n"]["fpr_abs_z_gt2"], fpr[2], 100 * cs["iid_noise_v2n"]["fpr_abs_z_gt2"] if cs else None, tol=1.0)
    if "effect_median_pct" in e:
        check(f"{label} median |effect| %", s["effect_pct_v2"]["median_abs"], e["effect_median_pct"], tol=tol_effect)
        check(f"{label} max |effect| %", s["effect_pct_v2"]["max_abs"], e["effect_max_pct"], tol=tol_effect)
        check(f"{label} neighbour correlation of P", cal["neighbour_corr_P"], 0.318, cs.get("neighbour_corr_P"), tol=0.005)
        check(f"{label} neighbour correlation of P_adj", cal["neighbour_corr_Padj"], 0.071, cs.get("neighbour_corr_Padj"), tol=0.005)


@pytest.mark.parametrize("model", list(INTERFACE_MODELS))
def test_t10_interface_test_from_cache(model):
    """The interface test (`interpretfm.interfaces.interface_test`) on the cached niche-labelled validation
    h5ad of each CosMx model (the reference's input) reproduces the manuscript's summary table and the cached
    pair tables (z within 1e-5, identical calls). About 20 s per model."""
    exp = EXPECTED[PRIMARY]
    inp = _cosmx_inputs()
    b = _cached_validation_adata(model)
    res = interface_test(b.X, np.asarray(b.obsm["spatial"])[:, :2], inp["exact"], inp["group"], b.obs["niche"].astype(str).to_numpy(), inp["cov"],
                         inp["params"], model=model)
    cached = cached_path(*SPS, "tables", "strict_v2_cosmx_author_summary.json")
    cs = json.load(open(cached))[model] if cached else None
    _check_interface_summary(f"{PRIMARY} T10 {model} (cache)", res["summary"], _expected_interfaces(model), cs)
    for kind, name in (("niche", "niche"), ("free", "free")):
        p = cached_path(*SPS, "tables", f"strict_v2_cosmx_author_{model}_{name}.csv")
        if p is None:
            continue
        ref, mine = pd.read_csv(p), res[kind]
        keys = ["niche", "group_a", "group_b", "n_edges"] if kind == "niche" else ["group_a", "group_b", "n_edges"]
        assert len(mine) == len(ref) and (mine[keys].astype(str).to_numpy() == ref[keys].astype(str).to_numpy()).all()
        zc, cc = ("z_v2", "call_v2") if kind == "niche" else ("z_v2_free", "call_v2_free")
        dz = float(np.nanmax(np.abs(mine[zc].to_numpy() - ref[zc].to_numpy())))
        report(f"{PRIMARY} T10 {model} max |Δz| vs the cached {name} table", dz, None, None, 1e-5)
        assert dz < 1e-5 and (mine[cc].to_numpy() == ref[cc].to_numpy()).all()
    if model == "scgpt":
        exp_layer = exp["layer"]
        report(f"{PRIMARY} T10 layer", exp_layer, "layer_1.norm2")


def test_t10_interface_test(paper):
    """The `interfaces` stage of the paper run: the niche-free numbers and the calibration do not depend on
    the niche partition and reproduce exactly; the niche-stratified numbers are computed on the pipeline's own
    r100 niches (ARI 0.4175 vs the cached labels' 0.4187; see docs/reproducing.md), so pairs near the 50-edge
    threshold, the shares and the median/maximum effect over the pairs can move: ±3 pairs, ±2 pp and ±0.05 on the
    effect sizes are accepted and both values are printed."""
    exp = EXPECTED[PRIMARY]
    rd = paper.run_dir(PRIMARY, NICHE_STAGES + ["interfaces"])
    s = json.load(open(rd / "11_interfaces" / exp["layer"] / "summary.json"))
    p = s["params"]
    assert p["edge_radius"] == 120.0 and p["block"] == 120.0 and p["legacy_tile"] == 400.0 and p["length_scales"] == [60.0, 150.0, 400.0]
    assert s["niche_column"] == "niche_at_r18um/xm/kmeans/k3" and s["covariates"] == ["log_counts", "log_genes", "log_area"]
    cached = cached_path(*SPS, "tables", "strict_v2_cosmx_author_summary.json")
    cs = json.load(open(cached))["scgpt"] if cached else None
    _check_interface_summary(f"{PRIMARY} T10 (interfaces stage, pipeline niches)", s, _expected_interfaces("scgpt"), cs, tol_pairs=3, tol_share=0.02,
                             tol_effect=0.05)
    free = pd.read_csv(rd / "11_interfaces" / exp["layer"] / "free_pairs.csv")
    pf = cached_path(*SPS, "tables", "strict_v2_cosmx_author_scgpt_free.csv")
    if pf is not None:
        ref = pd.read_csv(pf)
        dz = float(np.nanmax(np.abs(free["z_v2_free"].to_numpy() - ref["z_v2_free"].to_numpy())))
        report(f"{PRIMARY} T10 max |Δz| of the niche-free pairs vs the cached table", dz, None, None, 1e-5)
        assert dz < 1e-5 and (free["call_v2_free"].to_numpy() == ref["call_v2_free"].to_numpy()).all()
    fo = rd / "11_interfaces" / exp["layer"] / "fired_on.csv"
    pfo = cached_path("reconstruction", "readthrough_20261104", "P6_marker_excluded", "fired_on", "fired_on_cosmx__scgpt__layer_1_norm2.csv.gz")
    if fo.exists() and pfo is not None:
        mine = pd.read_csv(fo).set_index("feature")["fired_on_symbol"]
        ref = pd.read_csv(pfo).set_index("feature")["fired_on_sym"].reindex(mine.index)
        same = sum(set(str(a).split(";")) == set(str(b_).split(";")) for a, b_ in zip(mine, ref))
        report(f"{PRIMARY} fired-on gene sets identical to the cached ones", same, len(mine), None, 0)
        assert same == len(mine)


# --- tercile and marker-excluded checks (three models) -----------------------------------------------------


def _cached_fired_on(model: str, feats):
    layer = INTERFACE_MODELS[model][0].replace(".", "_")
    base = ("reconstruction", "readthrough_20261104", "P6_marker_excluded", "fired_on")
    p = cached_path(*base, f"fired_on_cosmx__{model}__{layer}.csv.gz") or cached_path(*base, f"fired_on_cosmx__{model}__{layer}_all_latents.csv.gz")
    if p is None:
        pytest.skip(f"cached fired-on genes of {model} not available")
    return fired_on_sets(pd.read_csv(p).set_index("feature").reindex(feats), "fired_on_sym")


def test_partner_marker_tercile_and_marker_excluded_from_cache():
    """Partner markers (top-10 Wilcoxon per author type, union per group), the partner-marker tercile over the
    three models' cached niche-free calls (39 rows / 13 pairs, median g 0.071, CI [-0.770, 0.581], includes 0)
    and the marker-excluded re-test (3 spillover-robust pairs; reduced-score median g -1.067, CI [-2.056, 0.079];
    the full score on the same permutations equals the tercile primary row). About one minute."""
    e_t, e_m = SPATIAL_COSMX_SCGPT["tercile"], SPATIAL_COSMX_SCGPT["marker_excluded"]
    inp = _cosmx_inputs()
    free = _cached_free_tables()
    M = partner_markers(inp["counts"], inp["symbols"], inp["exact"], top_n=10)
    by_group = markers_by_group(M, dict(zip(inp["exact"], inp["group"])))
    cm = cached_path("reconstruction", "spatial_paperA", "tables", "a3_partner_markers_by_group.csv")
    if cm is not None:
        ref = pd.read_csv(cm)
        same = sum(set(str(r.markers).split(";")) == by_group.get(r.group, set()) for r in ref.itertuples())
        report(f"{PRIMARY} partner-marker sets identical to the cached table (groups)", same, len(ref), None, 0)
        assert same == len(ref)
    X, Padj, P, fired, concepts = {}, {}, {}, {}, {}
    for m in INTERFACE_MODELS:
        b = _cached_validation_adata(m)
        X[m] = np.asarray(b.X)
        P[m] = program_score(X[m])
        Padj[m] = adjust_score(P[m], inp["exact"], inp["cov"])
        fired[m] = _cached_fired_on(m, b.var["feature"].astype(int).to_numpy())
        concepts[m] = b.var["concept"].astype(str).to_numpy()
    rows, summ = partner_marker_tercile(free, Padj, inp["coords"], inp["exact"], inp["group"], inp["counts"], inp["symbols"], by_group, inp["params"], raw_scores=P)
    prim = summ.iloc[0]
    assert prim["primary"]
    check(f"{PRIMARY} tercile rows", prim["n_rows"], e_t["n_rows"], tol=0)
    check(f"{PRIMARY} tercile pairs", prim["n_pairs"], e_t["n_pairs"], tol=0)
    check(f"{PRIMARY} tercile median g", prim["median_g"], e_t["median_g"])
    check(f"{PRIMARY} tercile CI low", prim["ci_lo"], e_t["ci"][0])
    check(f"{PRIMARY} tercile CI high", prim["ci_hi"], e_t["ci"][1])
    report(f"{PRIMARY} tercile verdict", prim["verdict"], e_t["verdict"])
    assert prim["verdict"] == e_t["verdict"]
    res = marker_excluded_calls(free, X, fired, inp["coords"], inp["exact"], inp["group"], inp["cov"], inp["counts"], inp["symbols"], by_group,
                                inp["params"], concept_names=concepts)
    for m, c in res["controls"].items():
        report(f"{PRIMARY} marker-excluded control {m}: max |Δz| vs the cached table, z* vs table", (c["max_abs_diff_z_v2_free"], c["zstar"], c["zstar_table"]))
        assert c["max_abs_diff_z_v2_free"] < 1e-5 and abs(c["zstar"] - c["zstar_table"]) < 1e-6
    V = res["verdicts"]
    robust = tuple(sorted(V[V["spillover_robust"]]["pair"]))
    check(f"{PRIMARY} marker-excluded pairs", len(V), e_m["n_pairs"], tol=0)
    report(f"{PRIMARY} spillover-robust pairs", robust, tuple(sorted(e_m["robust"])))
    assert robust == tuple(sorted(e_m["robust"]))
    pv = cached_path("reconstruction", "readthrough_20261104", "P6_marker_excluded", "tables", "p6_cosmx_pair_verdicts.csv")
    if pv is not None:
        ref = pd.read_csv(pv).set_index("pair").reindex(V["pair"])
        report(f"{PRIMARY} marker-excluded n_models_holding identical to the cached table", (V["n_models_holding"].to_numpy() == ref["n_models_holding"].to_numpy()).all(), True)
        assert (V["n_models_holding"].to_numpy() == ref["n_models_holding"].to_numpy()).all()
    TS = res["tercile_summary"]
    red = TS[(TS["score"].str.startswith("reduced")) & (TS["set"] == "all called pairs x models")].iloc[0]
    full = TS[(TS["score"].str.startswith("full")) & (TS["set"] == "all called pairs x models")].iloc[0]
    check(f"{PRIMARY} marker-excluded reduced-score median g", red["median_g"], e_m["reduced_median_g"])
    check(f"{PRIMARY} marker-excluded reduced-score CI low", red["ci_lo"], e_m["reduced_ci"][0])
    check(f"{PRIMARY} marker-excluded reduced-score CI high", red["ci_hi"], e_m["reduced_ci"][1])
    check(f"{PRIMARY} full score on the same permutations = tercile primary row (median g)", full["median_g"], prim["median_g"], tol=1e-9)
    check(f"{PRIMARY} full score on the same permutations = tercile primary row (CI low)", full["ci_lo"], prim["ci_lo"], tol=1e-9)
    check(f"{PRIMARY} full score on the same permutations = tercile primary row (CI high)", full["ci_hi"], prim["ci_hi"], tol=1e-9)


# --- T11-T12: placeholders ------------------------------------------------------------------------


@pytest.mark.skip(reason=SPATIAL_TODO)
@pytest.mark.parametrize("key", ["lr_screen", "matched_radius"])
def test_t11_t12_spatial(paper, key):
    assert key in SPATIAL_COSMX_SCGPT and PRIMARY in EXPECTED
