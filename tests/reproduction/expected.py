"""Published numbers of the six representative layers (Supplementary Tables S1 and S3) and where
their cached sources live under ``${INTERPRETFM_PAPER_CACHE}``.

Stored layer ids are zero-based (``layer_1.norm2`` is display Layer 2). Cached-source values are
read from the cache at test time (``cached_*`` helpers in the test modules), so that the
manuscript value is the only number pinned here.
"""
from __future__ import annotations

# Supplementary Table S3: (standardized probe, center-only probe, best single dimension), mean over 5 splits.
_S3_CONTROLS = {
    "cosmx": {"coexpression_pc50": (0.603, 0.638, 0.575), "coexpression": (0.590, 0.634, 0.608),
              "celltype": (0.608, 0.609, 0.593), "trivial": (0.558, 0.537, 0.547), "random": (0.494, 0.493, 0.501)},
    "xenium": {"coexpression_pc50": (0.589, 0.621, 0.564), "coexpression": (0.593, 0.629, 0.595),
               "celltype": (0.597, 0.599, 0.597), "trivial": (0.556, 0.551, 0.557), "random": (0.494, 0.496, 0.503)},
}

_DATASETS = {
    "cosmx": {"n_cells": 81236, "n_genes": 960, "n_concepts": 5650,
              "gt_dir": "f1heldout_gene_token_row_cosmx_all_layers_l1_1e-3/gprofiler", "tis_cache": "tis_layer_screen_3models_cosmx"},
    "xenium": {"n_cells": 465220, "n_genes": 399, "n_concepts": None,
               "gt_dir": "xenium_geneformer_c2sscale_strict_from_extract_to_immune/01_f1_heldout/gprofiler", "tis_cache": None},
}


def _layer(name, dataset, model, cache_model, layer, hidden_dim, tis_gain, f1_row, f1_gene, gsi, mhr, features_kept,
           readout_raw, readout_sae, lr_concepts_range=(262, 593)):
    d = dict(_DATASETS[dataset])
    d.update({
        "name": name, "dataset": dataset, "model": model, "cache_model": cache_model, "layer": layer,
        "cache_layer": layer.replace(".", "_"), "hidden_dim": hidden_dim, "lambda": 1.0,
        "tis_gain": tis_gain,
        # concept F1 (raw, untrained SAE, sparse SAE)
        "concept_f1": {"row": dict(zip(("raw", "untrained", "sae"), f1_row)),
                       "gene": dict(zip(("raw", "untrained", "sae"), f1_gene))},
        # cell level (T6, T7)
        "gsi_excess": dict(zip(("raw", "sae", "genes"), gsi)),
        "mhr100": dict(zip(("raw", "sae", "genes", "chance"), mhr)),
        "features_kept": features_kept,
        # Supplementary Table S3 (standardized, center-only, single dimension)
        "gene_readout": {"hidden": readout_raw, "sae": readout_sae, **_S3_CONTROLS[dataset]},
        # Results: "63-111 features per analysis layer together marked 262-593 biological concepts"
        "marked_concepts_range": lr_concepts_range,
    })
    return d


EXPECTED = {e["name"]: e for e in [
    _layer("cosmx_scgpt_layer2", "cosmx", "scgpt", "scgpt", "layer_1.norm2", 512, 0.176,
           (0.175, 0.230, 0.339), (0.033, 0.027, 0.021), (0.063, 0.128, 0.195), (0.465, 0.579, 0.567, 0.104), 194,
           (0.678, 0.685, 0.588), (0.636, 0.680, 0.571)),
    _layer("cosmx_geneformer_layer2", "cosmx", "geneformer", "geneformer", "layer_1", 1152, 0.171,
           (0.110, 0.166, 0.349), (0.054, 0.040, 0.017), (0.079, 0.146, 0.195), (0.503, 0.624, 0.572, 0.104), 261,
           (0.681, 0.684, 0.581), (0.620, 0.659, 0.560)),
    _layer("cosmx_c2s_layer14", "cosmx", "c2s_scale", "c2sscale", "layer_13", 2304, 0.169,
           (0.107, 0.169, 0.317), (0.067, 0.075, 0.025), (0.091, 0.169, 0.195), (0.553, 0.685, 0.572, 0.104), 170,
           (0.745, 0.740, 0.634), (0.666, 0.732, 0.605)),
    _layer("xenium_scgpt_layer3", "xenium", "scgpt", "scgpt", "layer_2.norm2", 512, 0.214,
           (0.319, 0.400, 0.462), (0.091, 0.075, 0.029), (0.115, 0.256, 0.262), (0.766, 0.826, 0.789, 0.251), 118,
           (0.662, 0.662, 0.562), (0.592, 0.630, 0.536)),
    _layer("xenium_geneformer_layer2", "xenium", "geneformer", "geneformer", "layer_1", 1152, 0.195,
           (0.261, 0.342, 0.450), (0.103, 0.103, 0.006), (0.115, 0.206, 0.267), (0.808, 0.823, 0.793, 0.251), 125,
           (0.666, 0.669, 0.579), (0.608, 0.636, 0.571)),
    _layer("xenium_c2s_layer14", "xenium", "c2s_scale", "c2sscale", "layer_13", 2304, 0.190,
           (0.253, 0.321, 0.411), (0.115, 0.091, 0.032), (0.096, 0.232, 0.267), (0.811, 0.841, 0.793, 0.251), 94,
           (0.678, 0.673, 0.583), (0.613, 0.680, 0.571)),
]}

#: The layer whose caches are complete today; the others are listed for P3.
PRIMARY = "cosmx_scgpt_layer2"

# Spatial numbers of the CosMx scGPT layer (AUDIT §4, T8-T13). Niche configurations are written with the native
# radius (CosMx pixels; r100 = 18 µm, r120 = 21.6 µm) as in the manuscript's tables.
SPATIAL_COSMX_SCGPT = {
    "niche": {"config": "r100/xm/kmeans/k3", "ari": 0.419, "nmi": 0.358, "seed_ari": 0.989, "purity": 0.822, "null": 0.625, "z": 378.3},
    "expression_baseline": {"winner": "r120/m/gmm/k3", "ari": 0.301, "nmi": 0.317, "at_scgpt_config": (0.330, 0.351)},
    "label_composition": {"ari": 0.269, "nmi": 0.289},
    "interfaces": {"n_tests": 129, "share_abs_z_gt_2_tile": 0.178, "share_abs_z_gt_2_calibrated": 0.054, "enriched": 2,
                   "depleted": 5, "z_star": 2.00, "effect_median_pct": 0.13, "effect_max_pct": 1.48,
                   "fpr_pct": (6.3, 4.8, 6.2), "niche_free": {"n_pairs": 86, "enriched": 2, "depleted": 7}},
    "interfaces_other_models": {   # same run, author labels (AUDIT §4 / the manuscript's summary table): the stage is model-agnostic
        "geneformer": {"n_tests": 132, "enriched": 3, "depleted": 6, "z_star": 2.00, "share_abs_z_gt_2_tile": 0.189, "share_abs_z_gt_2_calibrated": 0.068,
                       "niche_free": {"n_pairs": 86, "enriched": 2, "depleted": 0}},
        "c2sscale": {"n_tests": 125, "enriched": 1, "depleted": 4, "z_star": 2.054, "share_abs_z_gt_2_tile": 0.208, "share_abs_z_gt_2_calibrated": 0.040,
                     "niche_free": {"n_pairs": 86, "enriched": 2, "depleted": 2}},
    },
    # partner-marker tercile over the three models' niche-free calls (Supplementary note; primary row of the a3 summary)
    "tercile": {"n_rows": 39, "n_pairs": 13, "median_g": 0.071, "ci": (-0.770, 0.581), "verdict": "CI includes 0"},
    # marker-excluded QC (p6): spillover-robust pairs and the tercile gradient of the reduced score
    "marker_excluded": {"n_pairs": 13, "robust": ("macrophage|treg", "t_cd4_memory|dendritic", "t_cd4_memory|t_cd8_memory"),
                        "reduced_median_g": -1.067, "reduced_ci": (-2.056, 0.079)},
    # ligand-receptor screen (Table S12 / Results; per-pair values from the manuscript's screen table and its matched-radius re-run)
    "lr_screen": {"n_pairs": 397, "n_pairs_ge3": 97, "hits_ge1": 7, "hits_ge3": 33, "robust": 3, "depleted": 0,
                  "robust_pairs": ("CEACAM6_CEACAM6", "ESAM_ESAM", "PECAM1_PECAM1"),
                  "robust_fold_ge1": {"CEACAM6_CEACAM6": 1.360, "ESAM_ESAM": 1.334, "PECAM1_PECAM1": 1.299},
                  "radius_px": 100.0, "radius_um": 18.0, "seed": 7,
                  # the same screen at the matched radius (652.8 px = 117.5 um): hits at >= 1 / >= 3, robust
                  "matched": {"hits_ge1": 2, "hits_ge3": 52, "robust": 1, "robust_pairs": ("CEACAM6_CEACAM6",), "depleted": 0}},
    # matched-neighbour radii (Table S20): mean number of other cells within the radius, both sections
    "matched_radius": {"cosmx_px": 652.8, "cosmx_um": 117.5, "cosmx_cells_at_100px": 7.20, "cosmx_cells_at_matched": 286.70,
                       "xenium_um": 15.4, "xenium_cells_at_100um": 286.67, "xenium_cells_at_matched": 7.24,
                       "xenium_config": "xenium_scgpt_layer3"},
    "niche_concepts": {"shared_top25": (2, 8, 8), "jaccard": (0.042, 0.190, 0.190)},
}

# Spatial numbers of the other five layers (P3-C). CosMx: the author labels, the manuscript's three-niche
# configuration per model (SPS stage-B summary, `strict_v2_summary_counts.csv`). Xenium: the marker-derived labels
# (per-cell CSV, 14 labels incl. low_quality), the five-niche partition per model (readthrough P1: `p1_summary_k5.csv`,
# `p1_crosstalk_counts.csv`, `p1_concept_jaccard_summary.csv`, `C3_lr_screen_xenium.csv`), with the three-niche
# configuration of the stage-B summary as the second fixed configuration. Niche configurations are written in native
# units (CosMx px, Xenium µm) as `r<radius>/<space>/<method>/k<k>`.
SPATIAL_OTHER = {
    "cosmx_geneformer_layer2": {
        "labels": "author_cell_type",
        "niche": {"config": "r120/m/kmeans/k3", "ari": 0.342, "nmi": 0.292, "seed_ari": 0.998, "sizes": (27956, 24312, 28968), "rank": 1},
        "expression_baseline": {"winner": "r120/m/gmm/k3", "ari": 0.301, "nmi": 0.317, "at_model_config": (0.268, 0.269)},
        "label_composition": {"ari": 0.269, "nmi": 0.289},
        "interfaces": SPATIAL_COSMX_SCGPT["interfaces_other_models"]["geneformer"],
        "niche_concepts_vs": {"c2sscale": {"shared_top25": (6, 10, 11), "jaccard": (0.136, 0.250, 0.282)}},
    },
    "cosmx_c2s_layer14": {
        "labels": "author_cell_type",
        "niche": {"config": "r100/m/kmeans/k3", "ari": 0.377, "nmi": 0.323, "seed_ari": 0.998, "sizes": (27453, 25826, 27957), "rank": 1},
        "expression_baseline": {"winner": "r120/m/gmm/k3", "ari": 0.301, "nmi": 0.317, "at_model_config": (0.287, 0.292)},
        "label_composition": {"ari": 0.269, "nmi": 0.289},
        "interfaces": SPATIAL_COSMX_SCGPT["interfaces_other_models"]["c2sscale"],
    },
    "xenium_scgpt_layer3": {
        "labels": "marker_label",
        "niche": {"config": "r100/xm/gmm/k5", "ari": 0.0809, "nmi": 0.1443, "ari_excl_lowq": 0.0891, "ari_tacco": 0.0264,
                  "seed_ari": 0.9995, "purity": 0.531, "null": 0.484, "score": 3.7412, "rank": 17,
                  "sizes": (101058, 89078, 68372, 87980, 118732), "cores": 3},
        "niche_k3": {"config": "r100/m/gmm/k3", "ari": 0.0551, "nmi": 0.120, "seed_ari": 0.9999, "sizes": (126151, 155333, 183736)},
        "expression_baseline": {"winner": "r100/m/gmm/k5", "ari": 0.0754, "nmi": 0.1425, "at_model_config": (0.0900, 0.1817)},
        "label_composition": {"ari": 0.0738, "nmi": 0.1564},
        "interfaces": {"n_tests": 376, "enriched": 15, "depleted": 27, "z_star": 2.097, "share_abs_z_gt_2_tile": 0.372, "share_abs_z_gt_2_calibrated": 0.122,
                       "fpr_pct": (8.2, 5.9, None), "neighbour_corr_Padj": 0.141, "niche_free": {"n_pairs": 78, "enriched": 5, "depleted": 11}},
        "niche_concepts_vs": {"geneformer": {"matched_mean": 0.308, "unmatched_mean": 0.073, "jaccard": (0.282, 0.351, 0.163, 0.429, 0.316), "shared": (11, 13, 7, 15, 12)},
                              "c2sscale": {"matched_mean": 0.262, "unmatched_mean": 0.095, "jaccard": (0.316, 0.282, 0.042, 0.389, 0.282), "shared": (12, 11, 2, 14, 11)}},
    },
    "xenium_geneformer_layer2": {
        "labels": "marker_label",
        "niche": {"config": "r120/m/kmeans/k5", "ari": 0.0773, "nmi": 0.1410, "ari_excl_lowq": 0.0860, "ari_tacco": 0.0314,
                  "seed_ari": 0.9987, "purity": 0.876, "null": 0.681, "score": 3.7391, "rank": 18,
                  "sizes": (60661, 104282, 110418, 100330, 89529), "cores": 3},
        "niche_k3": {"config": "r150/m/gmm/k3", "ari": 0.0591, "nmi": 0.123, "seed_ari": 0.9999, "sizes": (159669, 154676, 150875)},
        "expression_baseline": {"winner": "r100/m/gmm/k5", "ari": 0.0754, "nmi": 0.1425, "at_model_config": (0.0518, 0.1052)},
        "label_composition": {"ari": 0.0738, "nmi": 0.1564},
        "interfaces": {"n_tests": 382, "enriched": 11, "depleted": 44, "z_star": 2.063, "share_abs_z_gt_2_tile": 0.335, "share_abs_z_gt_2_calibrated": 0.152,
                       "fpr_pct": (11.1, 5.7, None), "neighbour_corr_Padj": 0.180, "niche_free": {"n_pairs": 78, "enriched": 4, "depleted": 24}},
        "niche_concepts_vs": {"c2sscale": {"matched_mean": 0.287, "unmatched_mean": 0.099, "jaccard": (0.087, 0.316, 0.471, 0.282, 0.282), "shared": (4, 12, 16, 11, 11)}},
    },
    "xenium_c2s_layer14": {
        "labels": "marker_label",
        "niche": {"config": "r100/xm/kmeans/k5", "ari": 0.0684, "nmi": 0.1448, "ari_excl_lowq": 0.0766, "ari_tacco": 0.0343,
                  "seed_ari": 0.9957, "purity": 0.866, "null": 0.699, "score": 3.7356, "rank": 11,
                  "sizes": (140757, 80464, 115707, 52139, 76153), "cores": 3},
        "niche_k3": {"config": "r150/m/gmm/k3", "ari": 0.0602, "nmi": 0.124, "seed_ari": 0.9999, "sizes": (170974, 153311, 140935)},
        "expression_baseline": {"winner": "r100/m/gmm/k5", "ari": 0.0754, "nmi": 0.1425, "at_model_config": (0.0338, 0.0946)},
        "label_composition": {"ari": 0.0738, "nmi": 0.1564},
        "interfaces": {"n_tests": 376, "enriched": 18, "depleted": 24, "z_star": 2.048, "share_abs_z_gt_2_tile": 0.311, "share_abs_z_gt_2_calibrated": 0.120,
                       "fpr_pct": (12.2, 5.9, None), "neighbour_corr_Padj": 0.182, "niche_free": {"n_pairs": 78, "enriched": 5, "depleted": 13}},
    },
}

# The Xenium ligand–receptor screen (marker labels, 100 µm, seed 11; `C3_lr_screen_xenium.csv`) and the placement of
# its robust hits on the five-niche partitions (`p1_lr_screen_niche_context.csv`, ratio of the share of close pairs over
# the niche's share of cells, niches in the cached order n0…n4).
LR_SCREEN_XENIUM = {
    "n_pairs": 25, "n_pairs_ge3": 24, "hits_ge1": 3, "hits_ge3": 8, "robust": 3, "depleted": 3,
    "robust_pairs": ("CCL19_CCR7", "CDH1_CDH1", "CLEC2C_KLRB1"),
    "robust_fold_ge1": {"CCL19_CCR7": 2.577, "CDH1_CDH1": 1.674, "CLEC2C_KLRB1": 1.468},
    "radius_um": 100.0, "seed": 11,
    "placement_k5": {
        "xenium_scgpt_layer3": {"CCL19_CCR7": (3.20, 0.07, 1.90, 0.00, 0.05), "CLEC2C_KLRB1": (2.83, 0.39, 1.66, 0.07, 0.21), "CDH1_CDH1": (0.97, 2.94, 0.69, 0.25, 0.31)},
        "xenium_geneformer_layer2": {"CCL19_CCR7": (0.78, 0.03, 0.00, 0.26, 4.34), "CLEC2C_KLRB1": (0.79, 0.25, 0.08, 0.48, 3.73), "CDH1_CDH1": (0.08, 2.42, 0.20, 0.56, 1.44)},
        "xenium_c2s_layer14": {"CCL19_CCR7": (3.10, 0.31, 0.01, 0.05, 0.02), "CLEC2C_KLRB1": (2.71, 0.55, 0.08, 0.25, 0.22), "CDH1_CDH1": (0.95, 0.94, 0.21, 0.15, 2.94)},
    },
}

#: Xenium reference inputs of the interface test (marker labels: exact label = group, no group map).
XENIUM_INTERFACE_MODELS = {"scgpt": ("layer_2.norm2", "r100p0_xm_gmm_k5"), "geneformer": ("layer_1", "r120p0_m_kmeans_k5"),
                           "c2sscale": ("layer_13", "r100p0_xm_kmeans_k5")}
