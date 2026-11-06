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
    "lr_screen": {"n_pairs": 397, "hits_ge1": 7, "hits_ge3": 33, "robust": 3},
    "matched_radius": {"cosmx_px": 652.8, "cosmx_um": 117.5, "cosmx_cells": 287, "xenium_um": 15.4, "xenium_cells": 7.2},
    "niche_concepts": {"shared_top25": (2, 8, 8), "jaccard": (0.042, 0.190, 0.190)},
}
