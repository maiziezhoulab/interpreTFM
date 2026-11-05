"""Per-layer summary: one row per layer with every criterion and its baselines, reported
separately (no composite score), and per-criterion "best layer" notes.

The table joins the outputs of ``tis``, ``sae``, ``concept_eval``, ``gene_readout`` and
``cells``. The manuscript's representative layers were fixed in a preliminary screen and
kept; this module only reports which layer is best on each criterion so that a reader can
weigh them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

#: Criteria reported separately: column, higher-is-better, the baseline columns shown next to it, a short label.
CRITERIA: List[Dict[str, Any]] = [
    {"name": "TIS gain", "column": "tis_gain", "baselines": ["tis_shuffled_mean", "tis_pca_mean"],
     "note": "TIS of the model minus TIS of shuffled activations"},
    {"name": "Concept F1 (row split)", "column": "concept_f1_row_sae", "baselines": ["concept_f1_row_raw", "concept_f1_row_untrained"],
     "note": "held-out rows of known genes: gene detection"},
    {"name": "Concept F1 (whole-gene split)", "column": "concept_f1_gene_sae",
     "baselines": ["concept_f1_gene_raw", "concept_f1_gene_untrained"], "note": "held-out genes: concept generalization"},
    {"name": "Gene-level readout AUROC", "column": "readout_auroc_hidden",
     "baselines": ["readout_auroc_sae", "readout_auroc_expression_control", "readout_auroc_random"],
     "note": "linear readout of concept membership on unseen genes (hidden state)"},
    {"name": "GSI excess", "column": "gsi_excess_sae", "baselines": ["gsi_excess_raw", "gsi_excess_genes"],
     "note": "population specificity of cell-level features"},
    {"name": "MHR", "column": "mhr_sae", "baselines": ["mhr_raw", "mhr_genes", "mhr_chance"],
     "note": "marker-gene hit rate of leading features"},
    {"name": "Retained features", "column": "n_features_retained", "baselines": ["min_features"],
     "note": "concept-labelled features kept by the retention rule"},
]

EXPRESSION_CONTROLS = ("coexpression", "coexpression_pc50", "celltype", "trivial")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _get(d: Optional[Dict[str, Any]], *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def layer_row(run_dir: Union[str, Path], layer: str, dirs: Dict[str, Path], readout_scaling: str = "standardize",
              ) -> Dict[str, Any]:
    """One table row for ``layer`` from the stage directories (``dirs``: stage name -> directory)."""
    row: Dict[str, Any] = {"layer": layer}
    tis = _get(_read_json(dirs["tis"] / "tis.json"), "layers", layer, default={})
    row.update({"hidden_dim": _get(tis, "hidden_dim"), "tis_n_cells": _get(tis, "n_cells"),
                "tis_mean": _get(tis, "model", "mean"), "tis_shuffled_mean": _get(tis, "shuffled", "mean"),
                "tis_gain": _get(tis, "gain"), "tis_pca_mean": _get(tis, "pca", "mean")})
    sae = _get(_read_json(dirs["sae"] / "selected.json"), "layers", layer, default={})
    lam = _get(sae, "lambda")
    qc = _get(sae, "qc", f"{lam:g}", default={}) if lam is not None else {}
    row.update({"sae_lambda": lam, "sae_n_latents": _get(qc, "n_latents"), "sae_fvu": _get(qc, "fvu"),
                "sae_l0_frac": _get(qc, "l0_frac"), "sae_dead_frac": _get(qc, "dead_frac")})
    ev = _get(_read_json(dirs["concept_eval"] / "concept_f1.json"), "layers", layer, "evaluations", default={})
    for unit in ("row", "gene"):
        for rep in ("sae", "raw", "untrained"):
            row[f"concept_f1_{unit}_{rep}"] = _get(ev, f"{unit}_{rep}", "concept_f1")
        row[f"n_concepts_{unit}"] = _get(ev, f"{unit}_sae", "n_concepts")
    reps = _get(_read_json(dirs["gene_readout"] / "gene_readout.json"), "representations", default={})
    key = f"mean_probe_test_auroc_{readout_scaling}_mean"
    row["readout_auroc_hidden"] = _get(reps, f"hidden|{layer}", key)
    row["readout_auroc_hidden_sd"] = _get(reps, f"hidden|{layer}", f"mean_probe_test_auroc_{readout_scaling}_std")
    row["readout_auroc_sae"] = _get(reps, f"sae|{layer}", key)
    row["readout_auroc_untrained"] = _get(reps, f"untrained|{layer}", key)
    randoms = [v.get(key) for k, v in reps.items() if k.startswith("random_") and v.get(key) is not None]
    row["readout_auroc_random"] = float(np.mean(randoms)) if randoms else None
    controls = {c: _get(reps, f"{c}|-", key) for c in EXPRESSION_CONTROLS}
    controls = {c: v for c, v in controls.items() if v is not None}
    best = max(controls, key=controls.get) if controls else None
    row["readout_auroc_expression_control"] = controls.get(best) if best else None
    row["readout_expression_control"] = best
    sel = _get(_read_json(dirs["cells"] / "selection.json"), "layers", layer, default={})
    row.update({"n_features_retained": _get(sel, "n_features"), "n_concepts_kept": _get(sel, "n_concepts_kept"),
                "enough_features": _get(sel, "enough_features"),
                "min_features": _get(_read_json(dirs["cells"] / "selection.json"), "rule", "min_features"),
                "retained_test_f1_mean": _get(sel, "test_f1_mean")})
    spec = _get(_read_json(dirs["cells"] / "specificity.json"), "layers", layer, default={})
    for rep in ("sae", "raw", "genes"):
        row[f"gsi_excess_{rep}"] = _get(spec, "representations", rep, "gsi", "excess_mean")
        row[f"mhr_{rep}"] = _get(spec, "representations", rep, "mhr", "mhr")
        row[f"mhr_null_{rep}"] = _get(spec, "representations", rep, "mhr", "null_mean")
    row["gsi_excess_sae_retained"] = _get(spec, "representations", "sae", "gsi", "excess_mean_retained")
    row["mhr_chance"] = _get(spec, "representations", "sae", "mhr", "chance")
    row["mhr_k"] = _get(spec, "representations", "sae", "mhr", "k")
    row["specificity_n_cells"] = _get(spec, "n_cells")
    row["specificity_n_populations"] = _get(spec, "n_populations")
    return row


def layer_table(run_dir: Union[str, Path], layers: Sequence[str], dirs: Dict[str, Path], readout_scaling: str = "standardize"):
    """``DataFrame`` with one row per layer (see :func:`layer_row`)."""
    import pandas as pd

    return pd.DataFrame([layer_row(run_dir, layer, dirs, readout_scaling) for layer in layers])


def recommendations(table, representative_layer: Optional[str] = None) -> Dict[str, Any]:
    """Per criterion: the best layer, its value and the baselines at that layer. No composite."""
    out: Dict[str, Any] = {"note": "Criteria are reported separately; no composite score is computed. "
                                   "The representative layer is a configuration choice (model.representative_layer).",
                           "representative_layer": representative_layer, "criteria": []}
    for c in CRITERIA:
        col = c["column"]
        entry: Dict[str, Any] = {"criterion": c["name"], "column": col, "note": c["note"], "best_layer": None,
                                 "value": None, "baselines": {}, "per_layer": {}}
        if col in table.columns:
            vals = table[["layer", col]].dropna()
            entry["per_layer"] = {str(r.layer): float(getattr(r, col)) for r in vals.itertuples(index=False)}
            if len(vals):
                i = vals[col].astype(float).idxmax()
                entry["best_layer"] = str(table.loc[i, "layer"])
                entry["value"] = float(table.loc[i, col])
                entry["baselines"] = {b: (None if b not in table.columns or table.loc[i, b] is None or
                                          (isinstance(table.loc[i, b], float) and np.isnan(table.loc[i, b]))
                                          else float(table.loc[i, b])) for b in c["baselines"]}
                if representative_layer is not None and representative_layer in entry["per_layer"]:
                    entry["representative_value"] = entry["per_layer"][representative_layer]
                    entry["representative_is_best"] = entry["best_layer"] == representative_layer
        out["criteria"].append(entry)
    best = [e["best_layer"] for e in out["criteria"] if e["best_layer"] is not None]
    out["best_layer_counts"] = {layer: best.count(layer) for layer in sorted(set(best))}
    out["agreement"] = ("all criteria agree on one layer" if len(set(best)) == 1 and best else
                        "criteria disagree; read them separately" if best else "no criterion available")
    return out


def notes_text(rec: Dict[str, Any]) -> List[str]:
    """Human-readable one-liners, one per criterion."""
    lines = []
    for e in rec["criteria"]:
        if e["best_layer"] is None:
            lines.append(f"{e['criterion']}: not available")
            continue
        base = ", ".join(f"{k} {v:.3g}" for k, v in e["baselines"].items() if v is not None)
        line = f"{e['criterion']}: best at {e['best_layer']} ({e['value']:.3g}" + (f"; baselines: {base}" if base else "") + ")"
        if "representative_value" in e and not e.get("representative_is_best"):
            line += f"; representative layer {rec['representative_layer']}: {e['representative_value']:.3g}"
        lines.append(line)
    lines.append(rec["agreement"] + ".")
    return lines


__all__ = ["CRITERIA", "EXPRESSION_CONTROLS", "layer_row", "layer_table", "notes_text", "recommendations"]
