"""``report.json``: every metric next to its baseline, collected from the stage outputs of a run.

:func:`collect` gathers the stage outputs of a run directory into one JSON-serialisable mapping
(the per-stage pickers :func:`niche_metrics`, :func:`interface_metrics` and :func:`lr_metrics`
reduce the stage summaries to the numbers shown beside their baselines). The "how to read this"
texts are :data:`HOW_TO_READ` (the general notes) and :data:`SECTION_NOTES` (one per section of
the page: what each number measures and what it does not). The HTML page is
:mod:`interpretfm.report.html`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from interpretfm.layers import CRITERIA
from interpretfm.manifest import RunManifest, StageManifest
from interpretfm.runner import list_stages

HOW_TO_READ: List[Dict[str, str]] = [
    {"title": "Gene detection vs concept generalization (the two splits)",
     "text": "Concept F1 is reported under two held-out units. In the row split, held-out rows belong to genes the "
             "feature selection has seen in other cells, so a high value shows that features detect known genes of a "
             "concept. In the whole-gene split, the test genes were never seen during selection, so it measures "
             "whether a feature generalizes to other members of the concept. The manuscript's values are high for the "
             "first and near baseline for the second: features are concept-labelled gene detectors. Read both columns, "
             "always next to the raw-dimension and untrained-SAE baselines computed on the same rows."},
    {"title": "Cell-level comparison with single genes",
     "text": "GSI excess and MHR score how population-specific cell-level profiles are. The single-gene row uses the "
             "log-normalized expression of the panel genes on the same cells, shuffles and half split; raw dimensions "
             "are each split into a positive and a negative part. A feature set is only informative beyond marker genes "
             "when it is at least as specific as the genes themselves; MHR has a chance level of k / number of genes."},
    {"title": "Features carry the measured panel's information",
     "text": "The concept universe is built from the genes on the panel, and every feature is labelled by the concept "
             "whose panel genes it detects. Concept labels therefore describe what is observable on this panel, not "
             "genome-wide pathway activity, and a feature's cell-level value is a summary of the cell's measured genes "
             "as seen by the model. Vocabulary coverage (dataset section) states how many panel genes the model knows."},
    {"title": "Layers",
     "text": "Each criterion is reported separately per layer with its own baselines, and the notes name the best layer "
             "per criterion. No composite score is computed; a representative layer is a configuration choice."},
    {"title": "Spatial sections",
     "text": "The spatial sections are skipped for a dataset without coordinates (scRNA-seq mode) and shown as not run. "
             "Niches are read against expression-only and label-composition partitions, the interface test against its "
             "uncalibrated tile null and its false-positive shares, and the ligand–receptor screen is model-free: it uses "
             "the counts, the coordinates and the labels only, and is the reference the feature-based sections are read against."},
]

#: What each section's numbers measure and what they do not (rendered under the section header).
SECTION_NOTES: Dict[str, str] = {
    "dataset": "Input checks only: QC-flagged cells are kept and counted, vocabulary coverage is the share of panel genes the model "
               "knows. Nothing here is an evaluation result.",
    "layers": "One table per criterion with the baselines in grey: TIS beside its shuffled and PCA controls, concept F1 under the row "
              "split (gene detection) and the whole-gene split (concept generalization) beside raw dimensions and the untrained SAE, "
              "the gene-level readout beside its expression and random controls, GSI excess and MHR beside raw dimensions and single "
              "genes. The columns are not combined into a score; each layer is compared with its own baselines, criterion by criterion.",
    "sae": "Training diagnostics of the selected λ per layer: FVU is the unexplained variance, L0 the mean active fraction per token, "
           "dead the share of features that never fire in the QC rows. They describe the fit, not the usefulness of the features.",
    "cells": "GSI excess and MHR score how population-specific the cell-level profiles are on the same cells, shuffles and split for "
             "SAE features, raw dimensions (split into a positive and a negative part) and single genes; chance for MHR is k over the "
             "number of genes. Values above the single-gene row mean the features are at least as specific as the measured genes; "
             "they do not show that the features carry information beyond the panel.",
    "niches": "ARI and NMI measure agreement of the niche partition with the dataset labels; purity is the fraction of a cell's "
              "neighbours sharing its niche, read against the tile-shuffle null (its z). The expression-only partition at its own "
              "winner and at the model's configuration, and the label-composition partition, are the baselines: a niche partition "
              "is only informative beyond expression when it is read next to them. Seed stability is the median seed-to-seed ARI at "
              "the finer k; the sweep winner is a configuration choice and a near-tie is possible. The top concepts name what the "
              "features detect inside a niche, not a mechanism.",
    "interfaces": "The share of pairs with |z| > 2 is shown under the legacy tile null and under the calibrated within-block null; "
                  "z* is the call threshold set so that smooth, contact-free fields give about 5 % of |z| > z*. The calibrated null "
                  "controls the per-pair false-positive rate under smooth contact-free fields (the false-positive shares show how "
                  "well); it is not a false-discovery rate across pairs, and its power is not evaluated. Calls depend on the niche "
                  "partition; the niche-free column is the partition-free variant. Effect sizes are in percent of the score.",
    "lr_screen": "Model-free: counts, coordinates and labels only. A hit is a ligand–receptor pair whose number of close "
                 "(ligand-positive, receptor-positive) cell pairs exceeds the within-type permutation null by fold ≥ 1.2 at q < 0.05; "
                 "robust = hit at both positivity thresholds. The fold depends on the radius (a larger radius dilutes near-contact "
                 "co-localization), so hit sets are radius-specific; the mean number of other cells within the radius is given to "
                 "compare neighbourhood sizes across platforms. The niche placement ratio is the share of a hit's close pairs in a "
                 "niche over the niche's share of cells (1 = as expected from the niche's size); it describes where the hit lives, "
                 "not whether the niche causes it. Co-localization is not evidence of signalling.",
    "stages": "Status and wall time per stage from the manifests; a stage marked skipped does not apply to this configuration.",
}


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN -> None, numpy scalars -> Python."""
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def collect(cfg, run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Everything the report shows, from the stage outputs in ``run_dir``."""
    import pandas as pd

    run_dir = Path(run_dir)
    dirs = {s.name: run_dir / s.dir_name for s in list_stages()}
    rm = RunManifest.read(run_dir)
    stages = []
    for s in list_stages():
        m = StageManifest.read(dirs[s.name])
        stages.append({"stage": s.name, "dir": s.dir_name, "status": m.status if m else "not run",
                       "duration_s": m.duration_s if m else None, "error": (m.error or "").splitlines()[0] if m and m.error else None})
    by_name = {s["stage"]: s for s in stages}
    spatial_status = by_name.get("niches", {}).get("status", "not run")
    spatial_note = {"done": "niches: run", "skipped": "skipped: the dataset has no spatial coordinates",
                    "failed": "niches: failed (see 10_niches/manifest.json)", "not run": "not run"}.get(spatial_status, spatial_status)
    if spatial_status == "not run" and not cfg.dataset.is_spatial:
        spatial_note = "not run (scRNA-seq mode: no coordinates)"

    qc = _read_json(dirs["ingest"] / "dataset_qc.json") or {}
    table = pd.read_csv(dirs["layers"] / "layer_table.csv") if (dirs["layers"] / "layer_table.csv").exists() else pd.DataFrame()
    rec = _read_json(dirs["layers"] / "recommendations.json") or {}
    sel = _read_json(dirs["cells"] / "selection.json") or {}
    spec = _read_json(dirs["cells"] / "specificity.json") or {}
    sae = _read_json(dirs["sae"] / "selected.json") or {}
    niches = _read_json(dirs["niches"] / "selected.json") or {}
    interfaces = _read_json(dirs["interfaces"] / "summary.json") or {}
    if_status = by_name.get("interfaces", {}).get("status", "not run")
    spatial_note += {"done": "; interfaces: run", "skipped": "", "failed": "; interfaces: failed (see 11_interfaces/manifest.json)",
                     "not run": "; interfaces: not run"}.get(if_status, f"; interfaces: {if_status}")
    lr = _read_json(dirs["lr_screen"] / "summary.json") or {}
    if lr and (dirs["lr_screen"] / "screen.csv").exists():
        p0 = str((lr.get("params") or {}).get("thresholds", [1])[0])
        cols = ["interaction_name", f"fold_thr{p0}", f"z_thr{p0}", f"q_thr{p0}", f"hit_thr{p0}", "robust_hit"]
        sc = pd.read_csv(dirs["lr_screen"] / "screen.csv")
        lr["screen_rows"] = sc[[c for c in cols if c in sc.columns]].to_dict(orient="records")
    lr_status = by_name.get("lr_screen", {}).get("status", "not run")
    spatial_note += {"done": "; lr_screen: run", "skipped": "", "failed": "; lr_screen: failed (see 12_lr_screen/manifest.json)",
                     "not run": "; lr_screen: not run"}.get(lr_status, f"; lr_screen: {lr_status}")

    metrics = []
    for r in table.to_dict(orient="records"):
        for c in CRITERIA:
            metrics.append({"layer": r["layer"], "criterion": c["name"], "column": c["column"], "value": r.get(c["column"]),
                            "baselines": {b: r.get(b) for b in c["baselines"]}, "note": c["note"]})
    report = {
        "run": {"name": cfg.run.name, "run_dir": str(run_dir), "package_version": rm.package_version if rm else None,
                "created": rm.created if rm else None, "updated": rm.updated if rm else None,
                "config_hash": rm.config_hash if rm else None, "config_file": rm.config_file if rm else None},
        "model": {"name": cfg.model.name, "layers": list(cfg.model.layers), "representative_layer": cfg.model.representative_layer,
                  "checkpoint": str(cfg.model.checkpoint) if cfg.model.checkpoint else None,
                  "cell_pooling": {"kind": cfg.model.cell_pooling.kind, "token": cfg.model.cell_pooling.token}},
        "dataset": {"h5ad": str(cfg.dataset.h5ad), "n_cells": qc.get("n_cells"), "n_genes": qc.get("n_genes"),
                    "qc": qc.get("qc"), "vocab_coverage": qc.get("vocab_coverage"), "labels": qc.get("labels"),
                    "spatial": qc.get("spatial"), "warnings": qc.get("warnings", []),
                    "mode": "spatial" if cfg.dataset.is_spatial else "scRNA-seq"},
        "stages": stages,
        "spatial": {"status": spatial_status, "note": spatial_note},
        "layer_table": table.to_dict(orient="records"),
        "recommendations": rec,
        "metrics": metrics,
        "sae": {layer: {"lambda": v.get("lambda"), "rule": v.get("rule"),
                        "qc": (v.get("qc") or {}).get(f"{v['lambda']:g}") if v.get("lambda") is not None else None}
                for layer, v in (sae.get("layers") or {}).items()} if sae else {},
        "cells": {"selection": sel.get("layers", {}), "specificity": {layer: {k: v for k, v in e.items() if k != "populations"}
                                                                      for layer, e in (spec.get("layers") or {}).items()}},
        "niches": niche_metrics(niches),
        "interfaces": interface_metrics(interfaces),
        "lr_screen": lr_metrics(lr),
        "how_to_read": HOW_TO_READ,
        "section_notes": SECTION_NOTES,
    }
    return _clean(report)


def niche_metrics(niches: Dict[str, Any]) -> Dict[str, Any]:
    """The niche numbers next to their baselines, per layer, from ``10_niches/selected.json``."""
    out: Dict[str, Any] = {}
    for layer, e in (niches.get("layers") or {}).items():
        if e.get("skipped"):
            out[layer] = {"skipped": e["skipped"]}
            continue
        sel, val, base = e.get("selected") or {}, e.get("validation") or {}, e.get("baselines") or {}
        ex, lc = base.get("expression_svd50") or {}, base.get("label_composition") or {}
        win, at = ex.get("winner") or {}, ex.get("at_model_config") or {}
        stab = e.get("stability") or {}

        def agree(d):
            return {"ari": d.get("ari"), "nmi": d.get("nmi")}

        out[layer] = {
            "config": sel.get("config_native"), "n_cells": e.get("n_cells"), "n_features": e.get("n_features"),
            "model": {**agree(val), "seed_ari": sel.get("seed_ari"), "purity": val.get("purity"),
                      "purity_null_mean": val.get("purity_null_mean"), "purity_z": val.get("purity_z"), "sizes": val.get("sizes"),
                      "score": sel.get("score")},
            "baselines": {
                "expression_winner": {"config": (win.get("selected") or {}).get("config_native"), **agree(win),
                                      "seed_ari": (win.get("selected") or {}).get("seed_ari"), "purity": win.get("purity"),
                                      "purity_null_mean": win.get("purity_null_mean"), "purity_z": win.get("purity_z"),
                                      "skipped": win.get("skipped")} if ex else None,
                "expression_at_model_config": {**agree(at), "purity": at.get("purity"), "purity_null_mean": at.get("purity_null_mean"),
                                               "purity_z": at.get("purity_z")} if at else None,
                "label_composition": {"config": (lc.get("published") or {}).get("config"), **agree(lc.get("published") or {})} if lc.get("published") else
                                     ({"skipped": lc.get("skipped")} if lc else None),
                "label_composition_at_model_config": {"config": (lc.get("at_model_config") or {}).get("config"),
                                                      **agree(lc.get("at_model_config") or {})} if lc.get("at_model_config") else None,
                "tile_null": {"purity_null_mean": val.get("purity_null_mean"), "purity_null_std": val.get("purity_null_std"),
                              "tile_size": val.get("tile_size"), "n_null": val.get("n_null")},
            },
            "seed_stability": {"k": stab.get("k"), "model": (stab.get("model") or {}).get("median_seed_ari"),
                               "expression": (stab.get("expression_svd50") or {}).get("median_seed_ari")},
            "top_concepts": (e.get("concepts") or {}).get("top"),
        }
    return out


def interface_metrics(interfaces: Dict[str, Any]) -> Dict[str, Any]:
    """The interface numbers next to their baselines, per layer, from ``11_interfaces/summary.json``:
    the share of ``|z| > 2`` under the legacy tile null beside the calibrated null, ``z*``, the
    smooth-field and i.i.d.-noise false-positive shares, the calls (niche and niche-free) and the
    effect sizes."""
    out: Dict[str, Any] = {}
    params = interfaces.get("params") or {}
    for layer, e in (interfaces.get("layers") or {}).items():
        if e.get("skipped"):
            out[layer] = {"skipped": e["skipped"]}
            continue
        cal = e.get("calibration") or {}
        sn, sf = e.get("n_supported_niche") or {}, e.get("n_supported_free") or {}
        eff = e.get("effect_pct_v2") or {}
        out[layer] = {
            "n_cells": e.get("n_cells"), "n_features": e.get("n_features"), "n_edges": e.get("n_edges"),
            "niche_column": e.get("niche_column"), "n_niches": e.get("n_niches"),
            "units": {"edge_radius_native": params.get("edge_radius"), "block_native": params.get("block"),
                      "legacy_tile_native": params.get("legacy_tile"), **(params.get("um") or {})},
            "n_pairs": {"niche": e.get("n_pairs_niche"), "free": e.get("n_pairs_free")},
            "share_abs_z_gt2": {"legacy_tile": e.get("frac_absz_gt2_v1"), "calibrated": e.get("frac_absz_gt2_v2")},
            "z_star": {"niche": cal.get("zstar_niche"), "free": cal.get("zstar_free")},
            "fpr": {"smooth_field_legacy": (cal.get("v1") or {}).get("fpr_abs_z_gt2"), "smooth_field_calibrated": (cal.get("v2n") or {}).get("fpr_abs_z_gt2"),
                    "smooth_field_free": (cal.get("v2f") or {}).get("fpr_abs_z_gt2"), "iid_noise": (cal.get("iid_noise_v2n") or {}).get("fpr_abs_z_gt2")},
            "calls": {"niche": {"enriched": sn.get("enriched", 0), "depleted": sn.get("depleted", 0), "not_supported": sn.get("not supported", 0)},
                      "free": {"enriched": sf.get("enriched", 0), "depleted": sf.get("depleted", 0), "not_supported": sf.get("not supported", 0)}},
            "effect_pct": {"median_abs": eff.get("median_abs"), "max_abs": eff.get("max_abs")},
            "neighbour_corr": {"P": cal.get("neighbour_corr_P"), "P_adj": cal.get("neighbour_corr_Padj")},
            "covariates": e.get("covariates"),
            "fired_on": {k: v for k, v in (e.get("fired_on") or {}).items() if k in ("n_rows", "n_features", "median_n_fired", "skipped", "enabled")},
        }
    return out


def lr_metrics(lr: Dict[str, Any]) -> Dict[str, Any]:
    """The ligand–receptor screen numbers from ``12_lr_screen/summary.json``: the radius in both units
    with the mean number of other cells within it, pairs tested and hits per threshold, robust hits, the
    depleted pairs, the hits' statistics at both thresholds and their niche placement per layer."""
    if not lr:
        return {}
    summ = lr.get("summary") or {}
    params = lr.get("params") or {}
    thr = [str(t) for t in params.get("thresholds") or []]
    return {
        "radius_um": lr.get("radius_um"), "radius_native": lr.get("radius_native"), "um_per_unit": lr.get("um_per_unit"),
        "matched": lr.get("matched"), "mean_other_cells": lr.get("mean_other_cells"), "n_cells": lr.get("n_cells"), "n_edges": lr.get("n_edges"),
        "labels": lr.get("labels"), "database": {k: v for k, v in (lr.get("database") or {}).items() if k in ("n_rows", "n_complexes", "packaged")},
        "n_pairs_on_panel": lr.get("n_pairs_on_panel"), "thresholds": thr, "primary": thr[0] if thr else None,
        "n_perm": params.get("n_perm"), "seed": params.get("seed"), "min_positive": params.get("min_positive"),
        "q_max": params.get("q_max"), "fold_min": params.get("fold_min"),
        "tested": {t: (summ.get("thresholds") or {}).get(t, {}).get("n_tested") for t in thr},
        "hits": {t: (summ.get("thresholds") or {}).get(t, {}).get("n_hits") for t in thr},
        "hit_names": {t: (summ.get("thresholds") or {}).get(t, {}).get("hits") for t in thr},
        "n_robust": summ.get("n_robust"), "robust": summ.get("robust"), "n_depleted": summ.get("n_depleted"), "depleted": summ.get("depleted"),
        "hits_table": lr.get("hits_table") or [], "screen_rows": lr.get("screen_rows") or [], "niche_column": lr.get("niche_column"),
        "placement": {layer: ({"skipped": e["skipped"]} if e.get("skipped") else {"n_niches": e.get("n_niches"), "n_cells_with_niche": e.get("n_cells_with_niche"),
                                                                                   "ratios": e.get("ratios")})
                      for layer, e in (lr.get("layers") or {}).items()},
    }


__all__ = ["HOW_TO_READ", "SECTION_NOTES", "collect", "interface_metrics", "lr_metrics", "niche_metrics"]
