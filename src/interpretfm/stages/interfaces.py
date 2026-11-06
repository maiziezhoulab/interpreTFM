"""Stage ``interfaces``: the calibrated cell–cell interface test on the projected features
(spatial datasets with labels), with the legacy uncalibrated test beside it, plus the inputs of
the partner-marker tercile and marker-excluded checks.

Per projected layer (``08_cells/<layer>/cells.h5ad`` or ``cells_h5ad`` overrides) on the cells
that have a feature profile (:mod:`interpretfm.interfaces`):

1. **Labels**: the exact labels of ``dataset.labels`` (normalised: lower case, ``_`` for
   spaces, slashes and hyphens; ``labels.normalize: false`` keeps them verbatim) and their
   groups from ``dataset.labels.groups`` / ``group_map`` (labels without a group are their own
   group). Pairs are tested between groups; strata use the exact labels.
2. **Covariates** from the dataset: log total counts, log detected genes and log area
   (``covariates.area``: ``{column: <obs column>}`` or ``{product: [<a>, <b>]}``; without it
   the score is adjusted on counts and genes only, which the summary records).
3. **Scores**: the program score ``P`` (clip quantile ``clip_quantile``) and ``P_adj``
   (residualised within the exact label).
4. **Niche partition**: ``niche.column`` of ``10_niches/<layer>/labels.csv`` (``niche`` = the
   sweep winner, or ``niche_at_<config>`` of a fixed configuration; ``niche.config`` names
   the configuration instead), or ``niche_labels: {<layer>: csv}`` overrides.
5. **Tests**: the calibrated niche-stratified test, the niche-free test and the legacy tile
   test (:func:`interpretfm.interfaces.interface_test`); lengths in µm are converted with
   ``dataset.spatial.um_per_unit`` and recorded in both units.
6. **Partner markers** of the exact labels (top ``markers.top_n`` by one-vs-rest Wilcoxon on
   log1p CP10k) and their union per group; **fired-on genes** of every projected feature from
   the activation store (``fired_on``: first ``max_shards`` shards, threshold, rate and mass
   rules). These feed the two cross-model recipes documented in ``docs/pipeline/stage_interfaces.md``.

Parameters (``stages.interfaces``): ``edge_radius_um`` (required), ``block_um`` (default = the
edge radius), ``legacy_tile_um`` (10/3 × block), ``length_scales_um`` ([0.5, 1.25, 10/3] × block),
``n_perm`` (100), ``calib_perm`` (50), ``calib_reps`` (2), ``legacy_perm`` (20), ``min_edges``
(50), ``clip_quantile`` (0.995), ``exclude_groups`` ([low_quality, unresolved]),
``signal_fractions``, ``seeds`` (the reference seeds, see :data:`interpretfm.interfaces.DEFAULT_SEEDS`),
``labels`` ({normalize: true}, plus an optional label set for this stage with the keys of
``dataset.labels`` — ``column``, ``file``, ``file_column``, ``groups``, ``group_map`` — used
instead of ``dataset.labels``, as for the manuscript's Xenium marker-derived labels), ``covariates`` ({area: ...}), ``niche`` ({column: niche} or
{config: <label>}), ``niche_labels``, ``cells_h5ad``, ``markers`` ({enabled: true, top_n: 10}),
``fired_on`` ({enabled: true, max_shards: null, threshold: 0.15, rate: 0.2, mass: 0.8,
batch_size: 8192}), ``store_root``, ``token_vocab``, ``checkpoints`` ({<layer>: path}).

Outputs: ``summary.json``; ``markers.csv`` and ``markers_by_group.csv``; per layer
``<layer>/{niche_pairs.csv, free_pairs.csv, scores.csv, summary.json, fired_on.csv, fired_on.json}``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from interpretfm.genes import GeneTable, resolver_for
from interpretfm.ingest import counts_matrix, dataset_labels, label_group_map, stage_labels
from interpretfm.interfaces import (
    DEFAULT_EXCLUDE,
    DEFAULT_SEEDS,
    DEFAULT_SIGNAL_FRACTIONS,
    InterfaceParams,
    cell_area,
    covariates,
    fired_on_genes,
    group_labels,
    interface_test,
    markers_by_group,
    partner_markers,
    summary_row,
)
from interpretfm.params import get_list, get_mapping, get_param, param_path
from interpretfm.runner import StageContext, StageError, register_stage, stage_dir

OUTPUTS = ["summary.json"]
LAYER_FILES = ("niche_pairs.csv", "free_pairs.csv", "scores.csv", "summary.json")


def interface_params(params: Dict[str, Any], um_per_unit: float) -> InterfaceParams:
    """The :class:`interpretfm.interfaces.InterfaceParams` of ``stages.interfaces`` (native units)."""
    st = "interfaces"
    radius = get_param(params, "edge_radius_um", None, st, float)
    if radius is None or radius <= 0:
        raise ValueError("stages.interfaces.edge_radius_um is required (edge radius in µm, > 0; the manuscript used 120 native units)")
    block = float(get_param(params, "block_um", radius, st, float))
    tile = float(get_param(params, "legacy_tile_um", block * 10.0 / 3.0, st, float))
    scales = [float(v) for v in get_list(params, "length_scales_um", [0.5 * block, 1.25 * block, block * 10.0 / 3.0], st)]
    seeds = dict(DEFAULT_SEEDS)
    given = get_mapping(params, "seeds", st)
    unknown = sorted(set(given) - set(seeds))
    if unknown:
        raise ValueError(f"stages.interfaces.seeds: unknown seed name(s) {unknown}; expected a subset of {sorted(seeds)}")
    seeds.update({k: int(v) for k, v in given.items()})
    return InterfaceParams.from_um(
        float(um_per_unit), edge_radius_um=float(radius), block_um=block, legacy_tile_um=tile, length_scales_um=scales,
        n_perm=get_param(params, "n_perm", 100, st, int), calib_perm=get_param(params, "calib_perm", 50, st, int),
        calib_reps=get_param(params, "calib_reps", 2, st, int), legacy_perm=get_param(params, "legacy_perm", 20, st, int),
        min_edges=get_param(params, "min_edges", 50, st, int), clip_quantile=float(get_param(params, "clip_quantile", 0.995, st, float)),
        exclude=[str(x) for x in get_list(params, "exclude_groups", list(DEFAULT_EXCLUDE), st)],
        signal_fractions=[float(v) for v in get_list(params, "signal_fractions", list(DEFAULT_SIGNAL_FRACTIONS), st)], seeds=seeds)


def _niche_column(params: Dict[str, Any]) -> str:
    n = get_mapping(params, "niche", "interfaces")
    col = get_param(n, "column", None, "interfaces.niche", str)
    cfg_label = get_param(n, "config", None, "interfaces.niche", str)
    if col and cfg_label:
        raise ValueError("stages.interfaces.niche: give either column or config, not both")
    return col or (f"niche_at_{cfg_label}" if cfg_label else "niche")


def _niche_table(ctx: StageContext, layer: str, overrides: Dict[str, Any]) -> pd.DataFrame:
    if layer in overrides:
        p = param_path(ctx.config, overrides[layer], "interfaces", f"niche_labels.{layer}")
    else:
        p = ctx.input_dir("niches") / layer / "labels.csv"
    if not p.exists():
        raise FileNotFoundError(f"interfaces: {layer}: no niche labels at {p}")
    t = pd.read_csv(p, dtype={"cell": str})
    if "cell" not in t.columns:
        raise ValueError(f"interfaces: {p}: expected a 'cell' column")
    return t.set_index("cell")


def _fired_on(ctx: StageContext, layer: str, feats, genes: GeneTable, fo: Dict[str, Any]):
    """Fired-on genes of the projected features from the store (None with a note when unavailable)."""
    from interpretfm.sae import Encoder, load_checkpoint
    from interpretfm.stores import resolve_store

    cfg = ctx.config
    try:
        store = resolve_store(ctx)
    except (StageError, FileNotFoundError, KeyError) as e:
        return None, {"skipped": f"no activation store: {e}"}
    override = get_mapping(ctx.params, "checkpoints", "interfaces").get(layer)
    ckpt = str(param_path(cfg, override, "interfaces", f"checkpoints.{layer}")) if override else None
    if ckpt is None:
        sel_file = ctx.input_dir("cells") / "selection.json"
        with open(sel_file) as fh:
            sel = json.load(fh).get("layers", {}).get(layer, {})
        ckpt = sel.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        return None, {"skipped": f"no SAE checkpoint for {layer} (stages.interfaces.checkpoints or 08_cells/selection.json)"}
    encoder = Encoder(load_checkpoint(ckpt), "cpu")
    if "feature" not in feats.var.columns:
        return None, {"skipped": "cells.h5ad has no var['feature'] column (latent indices)"}
    fidx = feats.var["feature"].astype(int).to_numpy()
    resolver = resolver_for(cfg, ctx.params, store, genes, stage="interfaces")
    table, info = fired_on_genes(store, layer, encoder, resolver, genes.keys, feature_idx=fidx,
                                 max_shards=get_param(fo, "max_shards", None, "interfaces.fired_on", int),
                                 threshold=float(get_param(fo, "threshold", 0.15, "interfaces.fired_on", float)),
                                 rate=float(get_param(fo, "rate", 0.2, "interfaces.fired_on", float)),
                                 mass=float(get_param(fo, "mass", 0.8, "interfaces.fired_on", float)),
                                 batch_size=get_param(fo, "batch_size", 8192, "interfaces.fired_on", int))
    sym = dict(zip(genes.keys, genes.symbols))
    table.insert(1, "var_name", [str(v) for v in feats.var_names])
    if "concept" in feats.var.columns:
        table.insert(2, "concept", feats.var["concept"].astype(str).to_numpy())
    table["fired_on_symbol"] = [";".join(sym.get(k) or k for k in x.split(";")) if x else "" for x in table["fired_on"]]
    info.update({"checkpoint": ckpt, "store": str(store), "layer": layer})
    return table, info


@register_stage("interfaces", inputs=["ingest", "extract", "cells", "niches"], outputs=OUTPUTS, applies=lambda cfg: cfg.dataset.is_spatial,
                uses_labels=True, input_overrides={"cells": "cells_h5ad", "extract": "cells_h5ad"})
def run_interfaces(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Calibrated interface test with the legacy tile test, niche-free variant, partner markers and fired-on genes."""
    import anndata as ad

    cfg, p = ctx.config, ctx.params
    ds, sp = cfg.dataset, cfg.dataset.spatial
    upu = float(sp.um_per_unit)
    params = interface_params(p, upu)
    lab_cfg = get_mapping(p, "labels", "interfaces")
    normalize = get_param(lab_cfg, "normalize", True, "interfaces.labels", bool)
    cov_cfg = get_mapping(p, "covariates", "interfaces")
    area_spec = get_mapping(cov_cfg, "area", "interfaces.covariates")
    niche_col = _niche_column(p)
    niche_overrides = get_mapping(p, "niche_labels", "interfaces")
    overrides = get_mapping(p, "cells_h5ad", "interfaces")
    mk = get_mapping(p, "markers", "interfaces")
    mk_on = get_param(mk, "enabled", True, "interfaces.markers", bool)
    top_n = get_param(mk, "top_n", 10, "interfaces.markers", int)
    fo = get_mapping(p, "fired_on", "interfaces")
    fo_on = get_param(fo, "enabled", True, "interfaces.fired_on", bool)

    label_set = stage_labels(cfg, p, "interfaces")
    if label_set is None:
        raise ValueError("stages.interfaces needs dataset.labels or stages.interfaces.labels (the cell groups whose interfaces are tested)")
    adata = ad.read_h5ad(ds.h5ad)
    counts = counts_matrix(adata, ds.counts_layer)
    if sp.obsm not in adata.obsm:
        raise KeyError(f"dataset.spatial.obsm: '{sp.obsm}' not in adata.obsm")
    coords_all = np.asarray(adata.obsm[sp.obsm], dtype=np.float64)[:, :2]
    obs_names = pd.Index([str(o) for o in adata.obs_names])
    labels_all = dataset_labels(label_set, adata)
    gmap = label_group_map(label_set)
    exact_all, group_all = group_labels(labels_all, gmap, normalize=normalize)
    area = cell_area(adata.obs, area_spec)
    cov_all = covariates(counts, area)
    cov_all.index = obs_names
    gk = cfg.stage_params("concepts").get("gene_keys") or {}
    genes = GeneTable.from_adata(adata, ds.gene_id, symbol_column=gk.get("symbol_column"), ensembl_column=gk.get("ensembl_column"))
    ctx.log.info("interfaces: edge radius %g µm = %g native units, block %g, legacy tile %g, length scales %s (native); %d groups from %d labels; "
                 "covariates %s", params.edge_radius * upu, params.edge_radius, params.block, params.legacy_tile, list(params.length_scales),
                 len(set(group_all)), len(set(exact_all)), [c for c in ("log_counts", "log_genes", "log_area") if c in cov_all.columns])
    if area is None:
        ctx.log.warning("interfaces: no covariates.area configured: P is adjusted on log counts and log genes only")

    markers_info: Dict[str, Any] = {"enabled": mk_on, "top_n": top_n}
    if mk_on:
        M = partner_markers(counts, genes.keys, exact_all, top_n=top_n)
        sym = dict(zip(genes.keys, genes.symbols))
        M["symbol"] = [sym.get(g) or g for g in M["gene"]]
        M.to_csv(ctx.path("markers.csv"), index=False)
        by_group = markers_by_group(M, {e: g for e, g in zip(exact_all, group_all)})
        G = pd.DataFrame({"group": sorted(by_group), "markers": [";".join(sorted(by_group[g])) for g in sorted(by_group)],
                          "markers_symbol": [";".join(sorted(sym.get(x) or x for x in by_group[g])) for g in sorted(by_group)],
                          "n_markers": [len(by_group[g]) for g in sorted(by_group)]})
        G.to_csv(ctx.path("markers_by_group.csv"), index=False)
        markers_info.update({"n_labels": int(M["label"].nunique()), "n_groups": len(by_group), "file": "markers.csv", "by_group": "markers_by_group.csv"})
        ctx.log.info("interfaces: partner markers: top %d per label for %d labels, %d groups", top_n, M["label"].nunique(), len(by_group))

    summary: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        if layer in overrides:
            h5 = param_path(cfg, overrides[layer], "interfaces", f"cells_h5ad.{layer}")
        else:
            h5 = ctx.input_dir("cells") / layer / "cells.h5ad"
        if not h5.exists():
            ctx.log.warning("interfaces: %s: no projected features (%s); layer skipped", layer, h5)
            summary[layer] = {"layer": layer, "skipped": f"no projected features at {h5}"}
            continue
        ldir = ctx.path(layer, "summary.json").parent
        feats = ad.read_h5ad(h5)
        fobs = pd.Index([str(o) for o in feats.obs_names])
        use = feats.obs["has_row"].astype(bool).to_numpy() if "has_row" in feats.obs else np.ones(feats.n_obs, dtype=bool)
        cells = fobs[use]
        idx = obs_names.get_indexer(cells)
        if (idx < 0).any():
            raise ValueError(f"interfaces: {layer}: {int((idx < 0).sum())} cells of {h5} are not in the dataset")
        nt = _niche_table(ctx, layer, niche_overrides)
        if niche_col not in nt.columns:
            raise KeyError(f"interfaces: {layer}: niche column '{niche_col}' not in the niche labels (columns: {list(nt.columns)}); "
                           "set stages.interfaces.niche.column / .config")
        niche = nt[niche_col].reindex(cells)
        if niche.isna().any():
            raise ValueError(f"interfaces: {layer}: {int(niche.isna().sum())} cells with features have no niche label in column '{niche_col}'")
        niche = niche.astype(str).to_numpy()
        X = feats.X[use]
        ctx.log.info("interfaces: %s: %d cells x %d features, niche partition '%s' (%d niches)", layer, int(use.sum()), feats.n_vars, niche_col,
                     len(set(niche)))
        res = interface_test(X, coords_all[idx], exact_all[idx], group_all[idx], niche, cov_all.iloc[idx].reset_index(drop=True), params,
                             model=cfg.model.name)
        res["niche"].to_csv(ldir / "niche_pairs.csv", index=False)
        res["free"].to_csv(ldir / "free_pairs.csv", index=False)
        sc = res["scores"]
        sc.insert(0, "cell", np.asarray(cells))
        sc.to_csv(ldir / "scores.csv", index=False)
        info = dict(res["summary"])
        info.update({"layer": layer, "cells_h5ad": str(h5), "niche_column": niche_col, "n_niches": len(set(niche)),
                     "n_cells_without_features": int((~use).sum()), "row": summary_row(res["summary"])})
        cal = info["calibration"]
        ctx.log.info("interfaces: %s: %d niche pairs (|z|>2: tile %.1f %%, calibrated %.1f %%; %s; z* %.3f), %d niche-free pairs (%s; z* %.3f); "
                     "FPR smooth v1 %.1f %% / v2 %.1f %%, noise %.1f %%; neighbour corr P %.3f, P_adj %.3f", layer, info["n_pairs_niche"],
                     100 * info["frac_absz_gt2_v1"], 100 * info["frac_absz_gt2_v2"], info["n_supported_niche"], cal["zstar_niche"],
                     info["n_pairs_free"], info["n_supported_free"], cal["zstar_free"], 100 * cal["v1"]["fpr_abs_z_gt2"],
                     100 * cal["v2n"]["fpr_abs_z_gt2"], 100 * cal["iid_noise_v2n"]["fpr_abs_z_gt2"], cal["neighbour_corr_P"], cal["neighbour_corr_Padj"])
        if fo_on:
            table, finfo = _fired_on(ctx, layer, feats, genes, fo)
            if table is not None:
                table.to_csv(ldir / "fired_on.csv", index=False)
                finfo.update({"file": "fired_on.csv", "n_empty": int((table["n_fired"] == 0).sum()), "median_n_fired": float(table["n_fired"].median())})
                ctx.log.info("interfaces: %s: fired-on genes of %d features from %d gene-token rows (%d shards); median %g genes per feature",
                             layer, finfo["n_features"], finfo["n_rows"], len(finfo["shards_used"]), finfo["median_n_fired"])
            else:
                ctx.log.warning("interfaces: %s: fired-on genes skipped: %s", layer, finfo["skipped"])
            info["fired_on"] = finfo
            ctx.write_json(f"{layer}/fired_on.json", finfo)
        else:
            info["fired_on"] = {"enabled": False}
        ctx.write_json(f"{layer}/summary.json", info)
        summary[layer] = info

    ctx.write_json("summary.json", {"params": params.to_dict(), "niche_column": niche_col, "labels": {"column": label_set.column, "file": None if label_set.file is None else str(label_set.file), "normalize": normalize,
                                     "n_exact": int(len(set(exact_all))), "n_groups": int(len(set(group_all))), "group_map": gmap,
                                     "exclude": list(params.exclude)},
                                    "covariates": [c for c in ("log_counts", "log_genes", "log_area") if c in cov_all.columns],
                                    "markers": markers_info, "layers": summary})
    d = params.to_dict()
    return {"edge_radius_um": d["um"]["edge_radius"], "edge_radius_native": params.edge_radius, "block_um": d["um"]["block"],
            "block_native": params.block, "niche_column": niche_col,
            "layers": {k: (v["row"] if "row" in v else v.get("skipped")) for k, v in summary.items()}}


def read_interface_summary(run_dir: Path) -> Dict[str, Any]:
    """``11_interfaces/summary.json`` of a run directory (empty mapping when absent)."""
    f = stage_dir(run_dir, "interfaces") / "summary.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)


def read_interface_run(run_dir: Path, layer: str) -> Dict[str, Any]:
    """The per-layer outputs of a finished ``interfaces`` stage, for the cross-model recipes:
    ``free`` / ``niche`` pair tables, ``scores`` (indexed by cell), ``fired_on`` (``None`` when
    not computed), ``summary`` and the run's ``markers_by_group`` table."""
    d = stage_dir(run_dir, "interfaces")
    ldir = d / layer
    if not (ldir / "summary.json").exists():
        raise FileNotFoundError(f"no interfaces output for layer {layer!r} in {d}")
    out = {"free": pd.read_csv(ldir / "free_pairs.csv"), "niche": pd.read_csv(ldir / "niche_pairs.csv"),
           "scores": pd.read_csv(ldir / "scores.csv", dtype={"cell": str}).set_index("cell"),
           "fired_on": pd.read_csv(ldir / "fired_on.csv") if (ldir / "fired_on.csv").exists() else None,
           "markers_by_group": pd.read_csv(d / "markers_by_group.csv") if (d / "markers_by_group.csv").exists() else None}
    with open(ldir / "summary.json") as fh:
        out["summary"] = json.load(fh)
    return out
