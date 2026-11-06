"""Stage ``niches``: spatial niches from the projected features, with the expression-only and
label-composition baselines, seed stability and niche concept labels (spatial datasets only).

Per projected layer (``08_cells/<layer>/cells.h5ad``, or ``cells_h5ad`` overrides), on the
cells that have a feature profile (:mod:`interpretfm.niches`, :mod:`interpretfm.niche_concepts`):

1. **Sweep** over radii (µm, converted with ``dataset.spatial.um_per_unit``), spaces (``m``,
   ``xm``), methods (k-means, GMM), numbers of clusters and seeds; configurations whose
   smallest cluster is below ``min_cluster`` are dropped; score
   ``S = score.ari · ARI_seed + score.purity_z · clip(z) + score.sep · separability``.
2. **Selection** of the highest score (``k_report`` restricts the number of clusters) and
   **validation** of its labels: sizes, neighbourhood purity, tile-shuffle null and z,
   separability, seed ARI, and ARI/NMI against ``dataset.labels`` when present. Niches are
   numbered by decreasing size.
3. **Baselines**: ``expression_svd50`` (truncated SVD of log-normalized expression through
   the same sweep and score, and at the model's selected configuration),
   ``label_composition`` (label fractions over the radius neighbours, PCA 15, k-means with 20
   initialisations, at ``label_composition.radius_um``/``k`` and at the model's configuration;
   needs labels) and the optional ``expression_variants`` (four preprocessing variants at the
   model's configuration, ``expression_variants.seeds``).
4. **Seed stability**: median seed-to-seed ARI over the configurations with
   ``seed_stability.k`` clusters, model versus expression.
5. **Concept labels**: inside-versus-outside tests of every feature per niche, the top
   ``concepts.top_k`` by effect (``concepts.rank_by``).

Parameters (``stages.niches``): ``radii_um`` (required), ``k`` ([3, 4, 5]), ``methods``,
``spaces``, ``seeds`` ([0, 1, 2]), ``min_cluster`` (150), ``n_pcs`` (50), ``kernel``,
``tile_mult`` (4), ``n_null`` (10), ``null_seed`` (0), ``score`` ({ari: 1.5, purity_z: 0.1,
sep: 1.0, purity_z_clip: 10}), ``k_report`` (null), ``baselines`` ([expression_svd50,
label_composition]), ``expression`` ({n_components: 50, seed: 0}), ``label_composition``
({radius_um: <first radius>, k: <first k>, n_pcs: 15, n_init: 20, seed: 0}),
``expression_variants`` ({enabled: false, seeds: [0, 1, 2]}), ``seed_stability`` ({k: [4, 5]}),
``concepts`` ({top_k: 25, active_threshold: 0.0, min_cells: 20, rank_by: effect}),
``labels`` (a label set for this stage with the keys of ``dataset.labels`` — ``column``, ``file``,
``file_column`` — used instead of ``dataset.labels`` for the ARI/NMI and the label-composition
baseline; the manuscript's Xenium niches are compared with marker-derived labels from a per-cell
CSV), ``fixed_configs`` ([{radius_um, space, method, k}]: configurations validated besides the winner, for
example a published one), ``cells_h5ad`` ({<layer>: path}: projected features to use instead of the
``cells`` stage's).

Outputs: ``selected.json`` (every layer: selected configuration, validation, baselines,
stability) and per layer ``<layer>/{sweep.csv, selected.json, labels.csv, validation.json,
baselines/expression_sweep.csv, baselines/baselines.json, baselines/labels.csv,
stability.json, concepts.csv, top_concepts.csv}``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from interpretfm.ingest import dataset_labels, stage_labels
from interpretfm.niche_concepts import niche_feature_tests, top_concepts
from interpretfm.niches import (
    EXPRESSION_VARIANTS,
    METHODS,
    SPACES,
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
from interpretfm.params import get_list, get_mapping, get_param, param_path
from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["selected.json"]
BASELINES = ("expression_svd50", "label_composition")


def niche_grid(params: Dict[str, Any]) -> NicheGrid:
    """The :class:`interpretfm.niches.NicheGrid` of ``stages.niches``."""
    radii = get_list(params, "radii_um", [], "niches")
    if not radii:
        raise ValueError("stages.niches.radii_um is required (neighbourhood radii in µm, e.g. [18, 21.6, 27])")
    score = get_mapping(params, "score", "niches")
    return NicheGrid(
        radii_um=[float(r) for r in radii],
        k=[int(v) for v in get_list(params, "k", [3, 4, 5], "niches")],
        methods=get_list(params, "methods", list(METHODS), "niches", choices=METHODS),
        spaces=get_list(params, "spaces", list(SPACES), "niches", choices=SPACES),
        seeds=[int(s) for s in get_list(params, "seeds", [0, 1, 2], "niches")],
        min_cluster=get_param(params, "min_cluster", 150, "niches", int),
        n_pcs=get_param(params, "n_pcs", 50, "niches", int),
        kernel=get_param(params, "kernel", "uniform", "niches", str, choices=("uniform", "gaussian")),
        sigma_frac=float(get_param(params, "sigma_frac", 0.25, "niches", float)),
        tile_mult=float(get_param(params, "tile_mult", 4.0, "niches", float)),
        n_null=get_param(params, "n_null", 10, "niches", int),
        null_seed=get_param(params, "null_seed", 0, "niches", int),
        w_ari=float(get_param(score, "ari", 1.5, "niches.score", float)),
        w_purity_z=float(get_param(score, "purity_z", 0.1, "niches.score", float)),
        w_sep=float(get_param(score, "sep", 1.0, "niches.score", float)),
        purity_z_clip=float(get_param(score, "purity_z_clip", 10.0, "niches.score", float)),
    )


def _selected_native(sel: Dict[str, Any]) -> Dict[str, Any]:
    return {"radius_um": float(sel["radius_um"]), "radius_native": float(sel["radius"]), "space": str(sel["space"]),
            "method": str(sel["method"]), "k": int(sel["n_clusters"]), "config": str(sel["config"]),
            "config_native": config_label(sel["radius_um"], sel["space"], sel["method"], sel["n_clusters"], sel["radius"]),
            "score": float(sel["score"]), "rank": int(sel["rank"]), "n_configs": int(sel["n_configs"]),
            "seed_ari": float(sel["ari_mean"]), "seed_ari_std": float(sel["ari_std"]), "pca_components": int(sel["pca_components"])}


def _labels_at(X, coords, cfg_sel: Dict[str, Any], grid: NicheGrid, seed: int, ref=None):
    """Labels (numbered by size) and validation at one configuration."""
    lab, E, neigh = cluster_niches(X, coords, cfg_sel["radius_native"], cfg_sel["space"], cfg_sel["method"], cfg_sel["k"],
                                   seed=seed, n_pcs=grid.n_pcs, kernel=grid.kernel, sigma_frac=grid.sigma_frac)
    lab = relabel_by_size(lab)
    val = niche_validation(lab, neigh, coords, E, tile_size=grid.tile_mult * cfg_sel["radius_native"], n_null=grid.n_null,
                           null_seed=grid.null_seed, ref_labels=ref, purity_z_clip=grid.purity_z_clip)
    return lab, val


def _feature_table(feats) -> pd.DataFrame:
    """One row per feature column: ``feature`` (column index), ``concept`` (from ``var['concept']``
    or the ``<concept>|feat<i>`` naming) and ``var_name``, plus the other ``var`` columns."""
    var = feats.var.copy().reset_index(drop=True)
    names = [str(v) for v in feats.var_names]
    var["var_name"] = names
    if "concept" not in var.columns:
        var["concept"] = [v.split("|")[-2] if v.count("|") >= 1 else v for v in names]
    if "feature" not in var.columns:
        var["feature"] = np.arange(len(var))
    first = ["feature", "concept", "var_name"]
    return var[first + [c for c in var.columns if c not in first]]


def _fixed_configs(params: Dict[str, Any], grid: NicheGrid, upu: float) -> list:
    """``fixed_configs``: configurations validated besides the winner (e.g. a published one)."""
    out = []
    for i, c in enumerate(get_list(params, "fixed_configs", [], "niches")):
        if not isinstance(c, dict):
            raise ValueError(f"stages.niches.fixed_configs[{i}]: expected a mapping with radius_um, space, method, k")
        where = f"niches.fixed_configs[{i}]"
        r_um = float(get_param(c, "radius_um", None, where, float) or 0)
        if r_um <= 0:
            raise ValueError(f"stages.{where}.radius_um: required, > 0")
        space = get_param(c, "space", None, where, str, choices=SPACES)
        method = get_param(c, "method", None, where, str, choices=METHODS)
        k = get_param(c, "k", None, where, int)
        if space is None or method is None or k is None:
            raise ValueError(f"stages.{where}: space, method and k are required")
        r_nat = radius_to_native(r_um, upu)
        out.append({"radius_um": r_um, "radius_native": r_nat, "space": space, "method": method, "k": int(k),
                    "config": config_label(r_um, space, method, k), "config_native": config_label(r_um, space, method, k, r_nat),
                    "pca_components": None})
    return out


def _sweep_row(table: pd.DataFrame, c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The sweep row of a configuration (``None`` when it was dropped by min_cluster)."""
    hit = table[(table["radius"].astype(float) == float(c["radius_native"])) & (table["space"] == c["space"])
                & (table["method"] == c["method"]) & (table["n_clusters"].astype(int) == int(c["k"]))]
    if not len(hit):
        return None
    row = hit.iloc[0]
    d = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    d["rank"] = int(np.flatnonzero(table.index == row.name)[0]) + 1
    d["n_configs"] = int(len(table))
    return d


def _row_dict(sel: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("config", "radius_um", "radius", "space", "method", "n_clusters", "score", "ari_mean", "ari_std", "purity",
            "purity_null_mean", "purity_null_std", "purity_z", "separability", "fragmentation", "min_cluster_size", "rank", "n_configs")
    return {k: sel[k] for k in keep if k in sel}


@register_stage("niches", inputs=["ingest", "cells"], outputs=OUTPUTS, applies=lambda cfg: cfg.dataset.is_spatial, uses_labels=True,
                input_overrides={"cells": "cells_h5ad"})
def run_niches(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Niche sweep, validation, expression-only and label-composition baselines, seed stability, concept labels."""
    import anndata as ad

    cfg, p = ctx.config, ctx.params
    sp = cfg.dataset.spatial
    grid = niche_grid(p)
    upu = float(sp.um_per_unit)
    k_report = get_param(p, "k_report", None, "niches", int)
    baselines = get_list(p, "baselines", list(BASELINES), "niches", choices=BASELINES)
    expr_cfg = get_mapping(p, "expression", "niches")
    n_svd = get_param(expr_cfg, "n_components", 50, "niches.expression", int)
    expr_seed = get_param(expr_cfg, "seed", 0, "niches.expression", int)
    lc_cfg = get_mapping(p, "label_composition", "niches")
    lc_radius_um = float(get_param(lc_cfg, "radius_um", grid.radii_um[0], "niches.label_composition", float))
    lc_k = get_param(lc_cfg, "k", int(grid.k[0]), "niches.label_composition", int)
    lc_pcs = get_param(lc_cfg, "n_pcs", 15, "niches.label_composition", int)
    lc_init = get_param(lc_cfg, "n_init", 20, "niches.label_composition", int)
    lc_seed = get_param(lc_cfg, "seed", 0, "niches.label_composition", int)
    ev_cfg = get_mapping(p, "expression_variants", "niches")
    ev_on = get_param(ev_cfg, "enabled", False, "niches.expression_variants", bool)
    ev_seeds = [int(s) for s in get_list(ev_cfg, "seeds", [0, 1, 2], "niches.expression_variants")]
    ev_variants = get_list(ev_cfg, "variants", list(EXPRESSION_VARIANTS), "niches.expression_variants", choices=EXPRESSION_VARIANTS)
    st_cfg = get_mapping(p, "seed_stability", "niches")
    st_k = [int(v) for v in get_list(st_cfg, "k", [4, 5], "niches.seed_stability")]
    c_cfg = get_mapping(p, "concepts", "niches")
    top_k = get_param(c_cfg, "top_k", 25, "niches.concepts", int)
    active_thr = float(get_param(c_cfg, "active_threshold", 0.0, "niches.concepts", float))
    min_cells = get_param(c_cfg, "min_cells", 20, "niches.concepts", int)
    rank_by = get_param(c_cfg, "rank_by", "effect", "niches.concepts", str, choices=("effect", "score"))
    overrides = get_mapping(p, "cells_h5ad", "niches")
    fixed = _fixed_configs(p, grid, upu)
    seed = int(cfg.run.seed)

    adata = ad.read_h5ad(cfg.dataset.h5ad)
    counts = adata.layers[cfg.dataset.counts_layer] if cfg.dataset.counts_layer else adata.X
    if sp.obsm not in adata.obsm:
        raise KeyError(f"dataset.spatial.obsm: '{sp.obsm}' not in adata.obsm")
    coords_all = np.asarray(adata.obsm[sp.obsm], dtype=np.float64)[:, :2]
    obs_names = pd.Index([str(o) for o in adata.obs_names])
    label_set = stage_labels(cfg, p, "niches")
    labels_all = dataset_labels(label_set, adata)
    has_labels = labels_all is not None
    label_name = label_set.column if has_labels else None
    cells_dir = ctx.input_dir("cells") if not overrides else None

    grid_info = grid.to_dict(upu)
    ctx.log.info("niches: radii %s µm = %s native units (%g µm per unit); %d configurations per sweep", grid.radii_um,
                 grid_info["radii_native"], upu, len(grid.radii_um) * len(grid.k) * len(grid.methods) * len(grid.spaces))
    summary: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        if layer in overrides:
            h5 = param_path(cfg, overrides[layer], "niches", f"cells_h5ad.{layer}")
        else:
            h5 = (cells_dir or ctx.input_dir("cells")) / layer / "cells.h5ad"
        if not h5.exists():
            ctx.log.warning("niches: %s: no projected features (%s); layer skipped", layer, h5)
            summary[layer] = {"layer": layer, "skipped": f"no projected features at {h5}"}
            continue
        ldir = ctx.path(layer, "selected.json").parent
        bdir = ctx.path(layer, "baselines", "baselines.json").parent
        feats = ad.read_h5ad(h5)
        fobs = pd.Index([str(o) for o in feats.obs_names])
        idx = obs_names.get_indexer(fobs)
        if (idx < 0).any():
            raise ValueError(f"niches: {layer}: {int((idx < 0).sum())} cells of {h5} are not in the dataset")
        use = feats.obs["has_row"].astype(bool).to_numpy() if "has_row" in feats.obs else np.ones(feats.n_obs, dtype=bool)
        idx = idx[use]
        X = feats.X[use]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = np.ascontiguousarray(X, dtype=np.float32)
        coords = coords_all[idx]
        ref = labels_all[idx] if has_labels else None
        cells = fobs[use]
        n_cells, n_feat = X.shape
        ctx.log.info("niches: %s: %d cells with features x %d features from %s", layer, n_cells, n_feat, h5)

        # 1-2. sweep, selection, validation
        table = niche_sweep(X, coords, grid, upu)
        table.insert(0, "layer", layer)
        table.to_csv(ldir / "sweep.csv", index=False)
        if not len(table):
            raise ValueError(f"niches: {layer}: no configuration passed min_cluster = {grid.min_cluster}")
        sel = select_config(table, k_report)
        chosen = _selected_native(sel)
        lab, val = _labels_at(X, coords, chosen, grid, seed, ref)
        ctx.log.info("niches: %s: selected %s (score %.3f, rank %d/%d): seed ARI %.3f, purity %.3f (null %.3f, z %.1f)%s", layer,
                     chosen["config_native"], chosen["score"], chosen["rank"], chosen["n_configs"], chosen["seed_ari"],
                     val["purity"], val["purity_null_mean"], val["purity_z"],
                     f", ARI {val['ari']:.3f} / NMI {val['nmi']:.3f} vs {label_name}" if has_labels else "")
        lab_table = pd.DataFrame({"cell": cells, "niche": lab})
        at_fixed: Dict[str, Any] = {}
        for c in fixed:
            row = _sweep_row(table, c)
            c = dict(c, pca_components=grid.pca_components(n_feat))
            flab, fval = _labels_at(X, coords, c, grid, seed, ref)
            lab_table[f"niche_at_{c['config']}"] = flab
            at_fixed[c["config"]] = {"config": c, "sweep_row": _row_dict(row) if row else None, "in_sweep": row is not None,
                                     "rank": row["rank"] if row else None, "score": row["score"] if row else None,
                                     "seed_ari": row["ari_mean"] if row else None, "is_winner": c["config"] == chosen["config"],
                                     "ari_vs_selected": label_agreement(lab, flab)["ari"], **fval}
            ctx.log.info("niches: %s: fixed configuration %s: rank %s, seed ARI %s, purity %.3f (null %.3f, z %.1f)%s", layer,
                         c["config_native"], at_fixed[c["config"]]["rank"], None if row is None else round(row["ari_mean"], 4),
                         fval["purity"], fval["purity_null_mean"], fval["purity_z"],
                         f", ARI {fval['ari']:.3f} / NMI {fval['nmi']:.3f}" if has_labels else "")
        lab_table.to_csv(ldir / "labels.csv", index=False)
        ctx.write_json(f"{layer}/validation.json", {"layer": layer, "selected": chosen, "sweep_row": _row_dict(sel), **val,
                                                    "reference_labels": label_name,
                                                    "n_cells_without_features": int((~use).sum()), "fixed_configs": at_fixed})

        # 3. baselines
        base: Dict[str, Any] = {}
        base_labels = pd.DataFrame({"cell": cells})
        expr_table = None
        if "expression_svd50" in baselines:
            F = expression_svd(counts[idx], n_svd, expr_seed)
            et = niche_sweep(F, coords, grid, upu)
            et.insert(0, "layer", layer)
            et.to_csv(bdir / "expression_sweep.csv", index=False)
            expr_table = et
            e: Dict[str, Any] = {"features": f"truncated SVD ({F.shape[1]} components, seed {expr_seed}) of log-normalized counts"}
            if len(et):
                esel = _selected_native(select_config(et, k_report))
                elab, eval_ = _labels_at(F, coords, esel, grid, seed, ref)
                base_labels["expression_winner"] = elab
                e["winner"] = {"selected": esel, "sweep_row": _row_dict(select_config(et, k_report)), **eval_}
                ctx.log.info("niches: %s: expression-only winner %s%s", layer, esel["config_native"],
                             f": ARI {eval_['ari']:.3f} / NMI {eval_['nmi']:.3f}" if has_labels else "")
            else:
                e["winner"] = {"skipped": f"no configuration passed min_cluster = {grid.min_cluster}"}
            mlab, mval = _labels_at(F, coords, chosen, grid, seed, ref)
            base_labels["expression_at_model_config"] = mlab
            e["at_model_config"] = {"selected": chosen, **mval}
            ctx.log.info("niches: %s: expression-only at %s%s", layer, chosen["config_native"],
                         f": ARI {mval['ari']:.3f} / NMI {mval['nmi']:.3f}" if has_labels else "")
            e["at_fixed_configs"] = {}
            for c in fixed:
                c = dict(c, pca_components=grid.pca_components(F.shape[1]))
                flab, fval = _labels_at(F, coords, c, grid, seed, ref)
                base_labels[f"expression_at_{c['config']}"] = flab
                e["at_fixed_configs"][c["config"]] = {"selected": c, **fval}
            base["expression_svd50"] = e
        if "label_composition" in baselines:
            if not has_labels:
                base["label_composition"] = {"skipped": "no dataset.labels"}
            else:
                lc_r = radius_to_native(lc_radius_um, upu)
                out_lc: Dict[str, Any] = {"params": {"n_pcs": lc_pcs, "n_init": lc_init, "seed": lc_seed, "labels": label_name}}
                for tag, r_um, r_nat, kk in (("published", lc_radius_um, lc_r, lc_k),
                                             ("at_model_config", chosen["radius_um"], chosen["radius_native"], chosen["k"])):
                    llab, comp = label_composition(ref, coords, r_nat, kk, n_pcs=lc_pcs, n_init=lc_init, seed=lc_seed)
                    llab = relabel_by_size(llab)
                    base_labels[f"label_composition_{tag}"] = llab
                    out_lc[tag] = {"radius_um": float(r_um), "radius_native": float(r_nat), "k": int(kk),
                                   "config": config_label(r_um, "composition", "kmeans", kk, r_nat),
                                   "sizes": {str(a): int(b) for a, b in zip(*np.unique(llab, return_counts=True))},
                                   "n_label_categories": int(comp.shape[1]), **label_agreement(ref, llab)}
                    ctx.log.info("niches: %s: label composition (%s, r %g µm, k %d): ARI %.3f / NMI %.3f", layer, tag, r_um, kk,
                                 out_lc[tag]["ari"], out_lc[tag]["nmi"])
                base["label_composition"] = out_lc
        if ev_on:
            ev: Dict[str, Any] = {"config": chosen["config_native"], "seeds": ev_seeds, "variants": {}}
            for variant in ev_variants:
                F = expression_variant(counts[idx], variant, n_svd, expr_seed)
                rows = []
                for s_ in ev_seeds:
                    vlab, _, _ = cluster_niches(F, coords, chosen["radius_native"], chosen["space"], chosen["method"], chosen["k"],
                                                seed=s_, n_pcs=grid.n_pcs, kernel=grid.kernel, sigma_frac=grid.sigma_frac)
                    vlab = relabel_by_size(vlab)
                    base_labels[f"expression_{variant}_seed{s_}"] = vlab
                    row = {"seed": int(s_), "ari_vs_model_niches": label_agreement(lab, vlab)["ari"]}
                    if has_labels:
                        row.update(label_agreement(ref, vlab))
                    rows.append(row)
                ev["variants"][variant] = {"seeds": rows, "ari_mean": float(np.mean([r["ari"] for r in rows])) if has_labels else None,
                                           "nmi_mean": float(np.mean([r["nmi"] for r in rows])) if has_labels else None}
            base["expression_variants"] = ev
        ctx.write_json(f"{layer}/baselines/baselines.json", base)
        base_labels.to_csv(bdir / "labels.csv", index=False)

        # 4. seed stability
        stab = {"k": st_k, "model": seed_stability(table, st_k)}
        if expr_table is not None and len(expr_table):
            stab["expression_svd50"] = seed_stability(expr_table, st_k)
            if stab["model"]["median_seed_ari"] is not None and stab["expression_svd50"]["median_seed_ari"] is not None:
                stab["model_above_expression"] = bool(stab["model"]["median_seed_ari"] > stab["expression_svd50"]["median_seed_ari"])
        ctx.write_json(f"{layer}/stability.json", stab)

        # 5. concept labels
        var = _feature_table(feats)
        tests = niche_feature_tests(X, lab, var, active_threshold=active_thr, min_cells=min_cells)
        tests.to_csv(ldir / "concepts.csv", index=False)
        top = top_concepts(tests, top_k, rank_by=rank_by)
        top.to_csv(ldir / "top_concepts.csv", index=False)
        concepts_info = {"top_k": top_k, "rank_by": rank_by, "active_threshold": active_thr, "min_cells": min_cells,
                         "n_tests": int(len(tests)), "n_significant_q05": int((tests["qval"] <= 0.05).sum()) if len(tests) else 0,
                         "top": {str(n): g.sort_values("rank")["concept"].astype(str).tolist() for n, g in top.groupby("niche")},
                         "at_fixed_configs": {}}
        for c in fixed:
            col = f"niche_at_{c['config']}"
            ftests = niche_feature_tests(X, lab_table[col].to_numpy(), var, active_threshold=active_thr, min_cells=min_cells)
            ftop = top_concepts(ftests, top_k, rank_by=rank_by)
            ftop.to_csv(ldir / f"top_concepts_at_{c['config'].replace('/', '_')}.csv", index=False)
            concepts_info["at_fixed_configs"][c["config"]] = {
                "file": f"top_concepts_at_{c['config'].replace('/', '_')}.csv",
                "top": {str(n): g.sort_values("rank")["concept"].astype(str).tolist() for n, g in ftop.groupby("niche")}}

        layer_summary = {"layer": layer, "cells_h5ad": str(h5), "n_cells": int(n_cells), "n_features": int(n_feat),
                         "n_cells_without_features": int((~use).sum()), "selected": chosen,
                         "validation": {k: v for k, v in val.items()}, "fixed_configs": at_fixed, "baselines": base, "stability": stab,
                         "concepts": concepts_info, "grid": grid_info}
        ctx.write_json(f"{layer}/selected.json", layer_summary)
        summary[layer] = layer_summary

    ctx.write_json("selected.json", {"grid": grid_info, "k_report": k_report, "baselines": baselines,
                                     "label_composition": {"radius_um": lc_radius_um, "k": lc_k, "n_pcs": lc_pcs, "n_init": lc_init, "seed": lc_seed},
                                     "reference_labels": label_name, "layers": summary})
    return {"radii_um": grid_info["radii_um"], "radii_native": grid_info["radii_native"],
            "layers": {k: (v["selected"]["config_native"] if "selected" in v else v.get("skipped")) for k, v in summary.items()}}


def read_niche_summary(run_dir: Path) -> Dict[str, Any]:
    """``10_niches/selected.json`` of a run directory (empty mapping when absent)."""
    from interpretfm.runner import stage_dir

    f = stage_dir(run_dir, "niches") / "selected.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)

