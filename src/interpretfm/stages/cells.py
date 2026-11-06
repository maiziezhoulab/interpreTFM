"""Stage ``cells``: feature retention, projection of features to cells, and population
specificity (GSI excess, MHR@k) of SAE features against raw dimensions and single genes.

Per layer with a selected SAE (:mod:`interpretfm.cells`):

1. **Retention** from the ``concept_eval`` row-split tables of the SAE: per concept with
   5–100 panel genes the feature with the best validation F1 at τ = 0.15, kept when ≥ 0.40,
   de-duplicated; a layer needs ``min_features`` (10) retained features to be projected.
2. **Projection**: the retained features are averaged over every cell's gene-token rows
   (special tokens excluded, first copy of duplicated cells) into ``<layer>/cells.h5ad``
   (cells x features; ``obs`` carries the labels, QC flag and ``has_row``; ``obsm['spatial_um']``
   the coordinates in µm).
3. **Specificity** on a sample of cells read from evenly spaced shards (cells at shard edges
   excluded): GSI excess and MHR@k for all SAE features, the raw dimensions split into
   positive and negative parts, and the single genes (log-normalized expression), on the same
   cells, label shuffles and half split. Needs ``dataset.labels``.

Parameters (``stages.cells``): ``retain`` (``{tau: 0.15, min_valid_f1: 0.40, concept_size: [5, 100],
min_features: 10}``), ``projection`` (``{dedup_cells: true, batch_size: 8192}``), ``specificity``
(``{min_pop: 30, n_shuffles: 5, mhr_k: 100, n_leading: 3, n_assoc: 50, n_null: 50, n_cells: 40000,
representations: [sae, raw, genes], exclude_qc_flagged: false, seed: run.seed}``), ``device``
(``auto``), ``store_root`` and ``checkpoints`` (``{<layer>: path}``).

Outputs: ``selection.json`` and ``specificity.json`` (per layer), and per layer
``<layer>/{features.csv, kept_concepts.csv, per_concept_best.csv, selection.json, cells.h5ad,
projection.json, specificity.json, gsi_features.csv, mhr_by_population.csv}``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from interpretfm.cells import (
    RetentionRule,
    accumulate_cell_profiles,
    log_normalized_expression,
    pick_shards_for_cells,
    select_features,
    specificity,
    specificity_params,
)
from interpretfm.concepts import Universe
from interpretfm.params import get_list, get_mapping, get_param
from interpretfm.runner import StageContext, register_stage
from interpretfm.sae import Encoder, load_checkpoint
from interpretfm.stages.concept_eval import selected_checkpoints
from interpretfm.stores import list_shards, resolve_store

OUTPUTS = ["selection.json", "specificity.json"]
REPRESENTATIONS = ("sae", "raw", "genes")


def retention_rule(params: Dict[str, Any]) -> RetentionRule:
    r = get_mapping(params, "retain", "cells")
    size = get_list(r, "concept_size", [5, 100], "cells.retain")
    if len(size) != 2:
        raise ValueError("stages.cells.retain.concept_size: expected [min, max]")
    rule = RetentionRule(tau=float(get_param(r, "tau", 0.15, "cells.retain", float)),
                         min_valid_f1=float(get_param(r, "min_valid_f1", 0.40, "cells.retain", float)),
                         size_range=(int(size[0]), int(size[1])),
                         min_features=get_param(r, "min_features", 10, "cells.retain", int),
                         unit=get_param(r, "unit", "row", "cells.retain", str, choices=("row", "gene")))
    if rule.min_features < 1:
        raise ValueError("stages.cells.retain.min_features must be >= 1")
    return rule


def _device(params: Dict[str, Any]):
    import torch

    d = get_param(params, "device", "auto", "cells", str)
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


@register_stage("cells", inputs=["ingest", "extract", "sae", "concepts", "concept_eval"], outputs=OUTPUTS, uses_labels=True)
def run_cells(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Feature retention, projection to cells, and GSI excess / MHR@k against raw dimensions and single genes."""
    import anndata as ad
    import pandas as pd

    cfg, p = ctx.config, ctx.params
    rule = retention_rule(p)
    proj = get_mapping(p, "projection", "cells")
    dedup = get_param(proj, "dedup_cells", True, "cells.projection", bool)
    batch_size = get_param(proj, "batch_size", 8192, "cells.projection", int)
    spec_cfg = get_mapping(p, "specificity", "cells")
    spec = specificity_params(spec_cfg, default_seed=cfg.run.seed)
    n_cells_target = get_param(spec_cfg, "n_cells", 40000, "cells.specificity", int)
    reps = get_list(spec_cfg, "representations", list(REPRESENTATIONS), "cells.specificity", choices=REPRESENTATIONS)
    exclude_flagged = get_param(spec_cfg, "exclude_qc_flagged", False, "cells.specificity", bool)
    device = _device(p)
    store = resolve_store(ctx)

    universe = Universe.read(ctx.input_dir("concepts") / "universe")
    sizes = universe.concept_sizes
    term_names = {}
    if "term_id" in universe.terms.columns and "name" in universe.terms.columns:
        term_names = dict(zip(universe.terms["term_id"].astype(str), universe.terms["name"].astype(str)))

    cells = pd.read_csv(ctx.input_dir("ingest") / "cells.csv", dtype={"cell": str}, keep_default_na=False)
    adata = ad.read_h5ad(cfg.dataset.h5ad)
    if cfg.dataset.counts_layer:
        adata.X = adata.layers[cfg.dataset.counts_layer]
    obs_names = [str(o) for o in adata.obs_names]
    has_labels = cfg.dataset.labels is not None
    labels = cells["label"].astype(str).values if has_labels else None
    qc_pass = cells["qc_pass"].astype(str).str.lower().isin(("true", "1")).values
    coords = cells[["x_um", "y_um"]].values.astype(np.float64) if "x_um" in cells.columns else None

    selection: Dict[str, Any] = {}
    spec_out: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        ckpts = selected_checkpoints(ctx, layer)
        if not ckpts.get("sae"):
            ctx.log.warning("cells: %s: no selected SAE; layer skipped", layer)
            selection[layer] = {"layer": layer, "skipped": "no selected SAE checkpoint"}
            spec_out[layer] = {"layer": layer, "skipped": "no selected SAE checkpoint"}
            continue
        encoder = Encoder(load_checkpoint(ckpts["sae"]), device)
        ldir = ctx.path(layer, "selection.json").parent

        # 1. retention
        fdir = ctx.input_dir("concept_eval") / layer / f"f1_{rule.unit}_sae"
        if not (fdir / "valid_concept_f1.csv").exists():
            raise FileNotFoundError(f"cells: {fdir}/valid_concept_f1.csv missing; run concept_eval with units [{rule.unit}]")
        valid = pd.read_csv(fdir / "valid_concept_f1.csv")
        test = pd.read_csv(fdir / "test_concept_f1.csv")
        best, kept, feats, info = select_features(valid, test, sizes, rule, term_names)
        best.to_csv(ldir / "per_concept_best.csv", index=False)
        kept.to_csv(ldir / "kept_concepts.csv", index=False)
        feats.to_csv(ldir / "features.csv", index=False)
        sel = {"layer": layer, "checkpoint": ckpts["sae"], "f1_dir": str(fdir), "rule": rule.to_dict(), **info}
        ctx.log.info("cells: %s: %d concepts kept (valid F1 >= %.2f), %d features%s", layer, info["n_concepts_kept"],
                     rule.min_valid_f1, info["n_features"], "" if info["enough_features"] else
                     f" (< {rule.min_features}: not projected)")

        # 2. projection
        if info["enough_features"]:
            fidx = feats["feature"].astype(int).tolist()
            prof = accumulate_cell_profiles(store, layer, encoder, obs_names, cfg.model.special_tokens, shards=None,
                                            feature_idx=fidx, raw_halves=False, dedup_cells=dedup, batch_size=batch_size,
                                            device=device)
            X = np.zeros((adata.n_obs, len(fidx)), dtype=np.float32)
            X[prof.cell_index] = prof.sae
            has_row = np.zeros(adata.n_obs, dtype=bool)
            has_row[prof.cell_index] = True
            obs = pd.DataFrame(index=adata.obs_names)
            if has_labels:
                obs[cfg.dataset.labels.column] = pd.Categorical(labels)
            obs["qc_pass"] = qc_pass
            obs["has_row"] = has_row
            obs["n_gene_tokens"] = 0
            obs.loc[obs.index[prof.cell_index], "n_gene_tokens"] = prof.counts
            var = feats.set_index("var_name")
            out = ad.AnnData(X=X, obs=obs, var=var)
            if coords is not None:
                out.obsm["spatial_um"] = coords
            out.uns["interpretfm"] = {"layer": layer, "model": cfg.model.name, "checkpoint": str(ckpts["sae"]),
                                      "rule": json.dumps(rule.to_dict()), "coordinates": "micrometres"}
            out.write_h5ad(ldir / "cells.h5ad")
            frac_active = (X[has_row] > 0).mean(0) if has_row.any() else np.zeros(len(fidx))
            proj_summary = {"h5ad": str(ldir / "cells.h5ad"), "n_obs": int(adata.n_obs), "n_features": len(fidx),
                            "n_has_row": int(has_row.sum()), "n_qc_flagged_with_row": int((has_row & ~qc_pass).sum()),
                            "feature_frac_cells_active_median": float(np.median(frac_active)) if len(fidx) else None,
                            "n_features_active_in_lt_1pct_cells": int((frac_active < 0.01).sum()), **prof.stats}
            ctx.write_json(f"{layer}/projection.json", proj_summary)
            sel["projection"] = {k: proj_summary[k] for k in ("n_has_row", "n_qc_flagged_with_row",
                                                               "feature_frac_cells_active_median")}
            ctx.log.info("cells: %s: projected %d features to %d cells", layer, len(fidx), proj_summary["n_has_row"])
        ctx.write_json(f"{layer}/selection.json", sel)
        selection[layer] = sel

        # 3. specificity
        if not has_labels:
            spec_out[layer] = {"layer": layer, "skipped": "no dataset.labels"}
            ctx.log.warning("cells: %s: population specificity skipped (no labels)", layer)
            continue
        shards = pick_shards_for_cells(list_shards(store, layer), n_cells_target)
        prof = accumulate_cell_profiles(store, layer, encoder, obs_names, cfg.model.special_tokens, shards=shards,
                                        feature_idx=None, raw_halves="raw" in reps, dedup_cells=dedup,
                                        drop_edge_cells=len(shards) < prof_total_shards(store, layer), batch_size=batch_size,
                                        device=device)
        idx = prof.cell_index
        use = np.ones(len(idx), dtype=bool)
        if exclude_flagged:
            use &= qc_pass[idx]
        idx = idx[use]
        rep_mats: Dict[str, np.ndarray] = {}
        if "sae" in reps:
            rep_mats["sae"] = prof.sae[use]
        if "raw" in reps:
            rep_mats["raw"] = prof.raw[use]
        expr = log_normalized_expression(adata.X[idx])
        if "genes" in reps:
            rep_mats["genes"] = expr
        res = specificity(rep_mats, expr, labels[idx], spec)
        res.update({"layer": layer, "checkpoint": ckpts["sae"], "shards_read": len(shards), "n_shards": prof.stats["n_shards"],
                    "n_cells_sampled": int(len(prof.cell_index)), "n_qc_flagged_used": int((~qc_pass[idx]).sum()),
                    "exclude_qc_flagged": exclude_flagged, "n_qc_flagged_excluded": int((~use).sum())})
        per_feature = res.pop("per_feature")
        per_pop = res.pop("per_population")
        if per_pop is not None and len(per_pop):
            per_pop.to_csv(ldir / "mhr_by_population.csv", index=False)
        if "sae" in per_feature:
            retained = set(feats["feature"].astype(int).tolist())
            gf = pd.DataFrame({"feature": np.arange(encoder.n_latents), **{k: v for k, v in per_feature["sae"].items()}})
            gf["retained"] = gf["feature"].isin(retained)
            gf.to_csv(ldir / "gsi_features.csv", index=False)
            r = gf[gf["retained"] & gf["excess"].notna()]
            res["representations"]["sae"]["gsi"]["excess_mean_retained"] = float(r["excess"].mean()) if len(r) else None
        ctx.write_json(f"{layer}/specificity.json", res)
        spec_out[layer] = res
        for name, e in res["representations"].items():
            if "gsi" in e:
                ctx.log.info("cells: %s %s: GSI excess %.3f, MHR@%d %.3f (null %.3f, chance %.3f)", layer, name,
                             e["gsi"]["excess_mean"], spec.mhr_k, e["mhr"]["mhr"], e["mhr"]["null_mean"] or float("nan"),
                             e["mhr"]["chance"])

    ctx.write_json("selection.json", {"rule": rule.to_dict(), "layers": selection})
    ctx.write_json("specificity.json", {"params": {**spec.__dict__, "n_cells": n_cells_target, "representations": reps,
                                                   "exclude_qc_flagged": exclude_flagged}, "layers": spec_out})
    return {"layers": {k: {"n_features": v.get("n_features"), "enough_features": v.get("enough_features")}
                       for k, v in selection.items()}}


def prof_total_shards(store, layer: str) -> int:
    return len(list_shards(store, layer))
