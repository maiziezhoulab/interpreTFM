"""Stage ``lr_screen``: the model-free ligand–receptor co-localization screen, with the niche
placement of its hits.

The screen (:mod:`interpretfm.lr_screen`) reads the dataset only: raw counts, coordinates and the
labels of ``dataset.labels`` (the within-type null permutes positivity within these labels). No
foreground-model output enters it; it is computed once per run and is the model-free reference
for the feature-based spatial stages. The niche placement then uses the niche partition of each
projected layer (``10_niches/<layer>/labels.csv``, column picked as in ``interfaces``).

1. **Pairs**: the database pairs (packaged CellChatDB human export, or ``database`` /
   ``complexes`` overrides) whose ligand and receptor genes, complexes resolved to subunits, are
   all on the panel (gene symbols of ``dataset.gene_id`` / ``concepts.gene_keys``), de-duplicated
   by gene sets.
2. **Radius**: ``radius_um`` converted with ``dataset.spatial.um_per_unit``; or
   ``match_neighbors_to: <count>``, the radius at which the section's mean number of other cells
   within the radius equals ``count`` (:func:`interpretfm.neighbours.match_radius`, grid
   ``match_grid_step`` native units, rounded to ``match_round_decimals`` decimals). The manifest and
   ``summary.json`` record the radius in both units and the mean other-cell count at the radius.
3. **Screen**: positivity thresholds ``thresholds`` (first = primary), ``min_positive`` cells per
   side, ``n_perm`` within-type permutations with ``seed``, fold / z / BH q per threshold, hits
   (``q_max``, ``fold_min``), robust hits and the depleted flag.
4. **Placement**: for every hit at the primary threshold or robust hit (plus ``placement_pairs``),
   per layer: the share of close pairs attributed to the ligand-positive cell in each niche over the
   niche's share of cells, and (model-free) the cell-type pairs carrying the close pairs.

Parameters (``stages.lr_screen``): ``radius_um`` or ``match_neighbors_to`` (one of the two),
``match_grid_step`` (0.1), ``match_round_decimals`` (1), ``thresholds`` ([1, 3]), ``n_perm`` (100),
``seed`` (0), ``min_positive`` (100), ``q_max`` (0.05), ``fold_min`` (1.2), ``database`` (CSV),
``complexes`` (CSV, or ``false`` for none), ``niche`` ({column: ...} or {config: ...}),
``labels`` (a label set for this stage with the keys of ``dataset.labels`` — ``column``, ``file``,
``file_column``, ``groups``, ``group_map`` — used instead of ``dataset.labels``; the manuscript's
Xenium screen permutes within marker-derived labels read from a per-cell CSV),
``niche_labels`` ({<layer>: csv} overrides), ``placement_pairs`` ([interaction names]),
``placement`` (true).

Outputs: ``summary.json``, ``pairs.csv`` (the panel pairs), ``screen.csv`` (one row per pair with
the statistics at every threshold, the hit flags and the top cell-type pairs of the placed hits),
``<layer>/placement.csv`` (long: interaction, niche, share of pairs, share of cells, ratio).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from interpretfm.genes import GeneTable
from interpretfm.ingest import counts_matrix, dataset_labels, stage_labels
from interpretfm.lr_screen import (
    DEFAULT_COMPLEXES,
    DEFAULT_INTERACTIONS,
    ScreenParams,
    celltype_pairs,
    load_cellchatdb,
    neighbour_graph,
    niche_placement,
    panel_pairs,
    screen_pairs,
    screen_summary,
)
from interpretfm.neighbours import match_radius, mean_other_cells
from interpretfm.params import get_list, get_mapping, get_param, param_path
from interpretfm.runner import StageContext, register_stage, stage_dir
from interpretfm.units import native_to_um, um_to_native

OUTPUTS = ["summary.json", "screen.csv"]
STAGE = "lr_screen"


def screen_params(params: Dict[str, Any], radius_native: float) -> ScreenParams:
    """The :class:`interpretfm.lr_screen.ScreenParams` of ``stages.lr_screen`` at a radius in native units."""
    thresholds = [int(t) for t in get_list(params, "thresholds", [1, 3], STAGE)]
    return ScreenParams(radius=float(radius_native), thresholds=thresholds, n_perm=int(get_param(params, "n_perm", 100, STAGE, int)),
                        seed=int(get_param(params, "seed", 0, STAGE, int)), min_positive=int(get_param(params, "min_positive", 100, STAGE, int)),
                        q_max=float(get_param(params, "q_max", 0.05, STAGE, float)), fold_min=float(get_param(params, "fold_min", 1.2, STAGE, float)))


def resolve_radius(params: Dict[str, Any], um_per_unit: float, coords: np.ndarray) -> Dict[str, Any]:
    """The radius in native units from ``radius_um`` or ``match_neighbors_to`` (with the matching record)."""
    radius_um = get_param(params, "radius_um", None, STAGE, float)
    target = get_param(params, "match_neighbors_to", None, STAGE, float)
    if (radius_um is None) == (target is None):
        raise ValueError("stages.lr_screen: give exactly one of radius_um (µm) or match_neighbors_to (mean number of other cells within the radius)")
    if radius_um is not None:
        if radius_um <= 0:
            raise ValueError(f"stages.lr_screen.radius_um: must be > 0, got {radius_um}")
        r = round(um_to_native(float(radius_um), um_per_unit), 6)
        return {"radius_native": r, "radius_um": float(radius_um), "matched": None}
    m = match_radius(coords, float(target), step=float(get_param(params, "match_grid_step", 0.1, STAGE, float)),
                     round_decimals=int(get_param(params, "match_round_decimals", 1, STAGE, int)),
                     r_max=get_param(params, "match_r_max", None, STAGE, float))
    m["radius_um"] = native_to_um(m["radius"], um_per_unit)
    return {"radius_native": float(m["radius"]), "radius_um": float(m["radius_um"]), "matched": m}


def _niche_column(params: Dict[str, Any]) -> str:
    n = get_mapping(params, "niche", STAGE)
    col = get_param(n, "column", None, STAGE + ".niche", str)
    cfg_label = get_param(n, "config", None, STAGE + ".niche", str)
    if col and cfg_label:
        raise ValueError("stages.lr_screen.niche: give either column or config, not both")
    return col or (f"niche_at_{cfg_label}" if cfg_label else "niche")


def _niche_labels(ctx: StageContext, layer: str, overrides: Dict[str, Any], column: str, obs_names: pd.Index):
    """Niche label per dataset cell (object array, ``None`` without a label), or ``(None, note)``."""
    if layer in overrides:
        p = param_path(ctx.config, overrides[layer], STAGE, f"niche_labels.{layer}")
    else:
        p = ctx.input_dir("niches") / layer / "labels.csv"
    if not p.exists():
        return None, f"no niche labels at {p}"
    t = pd.read_csv(p, dtype={"cell": str})
    if "cell" not in t.columns:
        raise ValueError(f"lr_screen: {p}: expected a 'cell' column")
    if column not in t.columns:
        raise KeyError(f"lr_screen: {layer}: niche column '{column}' not in {p} (columns: {list(t.columns)}); set stages.lr_screen.niche.column / .config")
    s = t.set_index("cell")[column].reindex(obs_names)
    lab = np.array([None if pd.isna(v) else str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) for v in s], dtype=object)
    return lab, None


@register_stage("lr_screen", inputs=["ingest", "niches"], outputs=OUTPUTS, applies=lambda cfg: cfg.dataset.is_spatial, uses_labels=True)
def run_lr_screen(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Model-free ligand–receptor co-localization screen with the niche placement of its hits."""
    import anndata as ad
    import scipy.sparse as sp

    cfg, p = ctx.config, ctx.params
    ds, spat = cfg.dataset, cfg.dataset.spatial
    upu = float(spat.um_per_unit)
    label_set = stage_labels(cfg, p, STAGE)
    if label_set is None:
        raise ValueError("stages.lr_screen needs dataset.labels or stages.lr_screen.labels (the cell types of the within-type null)")
    db_path = get_param(p, "database", None, STAGE, str)
    cx = p.get("complexes", None)
    if cx is not None and not isinstance(cx, (str, bool)):
        raise ValueError(f"stages.lr_screen.complexes: expected a CSV path or false, got {cx!r}")
    db_file = param_path(cfg, db_path, STAGE, "database") if db_path else DEFAULT_INTERACTIONS
    cx_file = False if cx is False else (param_path(cfg, cx, STAGE, "complexes") if isinstance(cx, str) else DEFAULT_COMPLEXES)
    niche_col = _niche_column(p)
    niche_overrides = get_mapping(p, "niche_labels", STAGE)
    placement_on = get_param(p, "placement", True, STAGE, bool)
    extra_pairs = [str(x) for x in get_list(p, "placement_pairs", [], STAGE)]

    adata = ad.read_h5ad(ds.h5ad)
    X = sp.csr_matrix(counts_matrix(adata, ds.counts_layer)).astype(np.float32)
    if spat.obsm not in adata.obsm:
        raise KeyError(f"dataset.spatial.obsm: '{spat.obsm}' not in adata.obsm")
    coords = np.asarray(adata.obsm[spat.obsm], dtype=np.float64)[:, :2]
    obs_names = pd.Index([str(o) for o in adata.obs_names])
    labels = dataset_labels(label_set, adata)
    gk = cfg.stage_params("concepts").get("gene_keys") or {}
    genes = GeneTable.from_adata(adata, ds.gene_id, symbol_column=gk.get("symbol_column"), ensembl_column=gk.get("ensembl_column"))
    symbols = [s if s is not None else k for s, k in zip(genes.symbols, genes.keys)]

    db, cmap = load_cellchatdb(db_file, cx_file)
    pairs = panel_pairs(db, cmap, symbols)
    pairs.to_csv(ctx.path("pairs.csv"), index=False)
    rad = resolve_radius(p, upu, coords)
    params = screen_params(p, rad["radius_native"])
    mean_count = rad["matched"]["mean_other_cells"] if rad["matched"] else mean_other_cells(coords, params.radius)
    ctx.log.info("lr_screen: %d cells, %d labels; %d database rows -> %d pairs on the panel (%d complexes known); radius %g native units = %g µm "
                 "(%s), mean other cells within the radius %.3f; thresholds %s, %d permutations, seed %d", X.shape[0], len(set(labels.tolist())),
                 len(db), len(pairs), len(cmap), params.radius, rad["radius_um"], "matched to %g" % rad["matched"]["target"] if rad["matched"] else "radius_um",
                 mean_count, params.thresholds, params.n_perm, params.seed)
    A = neighbour_graph(coords, params.radius)
    n_edges = int(A.nnz // 2)
    res = screen_pairs(X, symbols, labels, coords, pairs, params, A=A,
                       progress=lambda k, n, r: ctx.log.info("lr_screen: pair %d / %d (%s)", k, n, r.get("interaction_name")))
    summ = screen_summary(res, params)
    p0 = params.primary
    ctx.log.info("lr_screen: tested %s; hits %s; robust %d (%s); depleted %d", {t: v["n_tested"] for t, v in summ["thresholds"].items()},
                 {t: v["n_hits"] for t, v in summ["thresholds"].items()}, summ["n_robust"], ", ".join(summ["robust"]), summ["n_depleted"])

    # placement of the hits
    placed = res.index[(res[f"hit_thr{p0}"].fillna(False).astype(bool)) | (res["robust_hit"].fillna(False).astype(bool))
                       | res["interaction_name"].isin(extra_pairs)].tolist()
    unknown = sorted(set(extra_pairs) - set(res["interaction_name"]))
    if unknown:
        ctx.log.warning("lr_screen: placement_pairs not among the panel pairs: %s", unknown)
    res["top_celltype_pairs"] = ""
    for i in placed:
        res.at[i, "top_celltype_pairs"] = celltype_pairs(X, symbols, A, labels, res.at[i, "ligand_genes"].split(";"), res.at[i, "receptor_genes"].split(";"), thr=p0)
    layers: Dict[str, Any] = {}
    if placement_on:
        for layer in cfg.model.layers:
            lab, note = _niche_labels(ctx, layer, niche_overrides, niche_col, obs_names)
            if lab is None:
                ctx.log.warning("lr_screen: %s: placement skipped: %s", layer, note)
                layers[layer] = {"layer": layer, "skipped": note}
                continue
            P = niche_placement(res, X, symbols, A, lab, placed, thr=p0)
            ldir = ctx.path(layer, "placement.csv").parent
            P.to_csv(ldir / "placement.csv", index=False)
            n_lab = int(sum(v is not None for v in lab))
            ratios = {name: {str(r.niche): float(r.ratio) for r in g.itertuples()} for name, g in P.groupby("interaction_name", sort=False)}
            layers[layer] = {"layer": layer, "niche_column": niche_col, "n_niches": int(len(set(v for v in lab if v is not None))),
                             "n_cells_with_niche": n_lab, "n_placed": len(placed), "file": f"{layer}/placement.csv", "ratios": ratios}
            ctx.log.info("lr_screen: %s: placement of %d pairs over %d niches ('%s', %d of %d cells labelled)", layer, len(placed),
                         layers[layer]["n_niches"], niche_col, n_lab, len(lab))
    res.to_csv(ctx.path("screen.csv"), index=False)

    hits_table = res.loc[placed, ["interaction_name", "pathway_name", "ligand_genes", "receptor_genes"] + [c for thr in params.thresholds for c in
                        (f"n_lig_pos_thr{thr}", f"n_rec_pos_thr{thr}", f"obs_thr{thr}", f"fold_thr{thr}", f"z_thr{thr}", f"q_thr{thr}", f"hit_thr{thr}")]
                        + ["robust_hit", f"depleted_thr{p0}", "top_celltype_pairs"]]
    out = {"params": params.to_dict(), "radius_um": rad["radius_um"], "radius_native": params.radius, "um_per_unit": upu, "matched": rad["matched"],
           "mean_other_cells": float(mean_count), "n_cells": int(X.shape[0]), "n_edges": n_edges, "labels": {"column": label_set.column, "file": None if label_set.file is None else str(label_set.file), "n": int(len(set(labels.tolist())))},
           "database": {"interactions": str(db_file), "complexes": str(cx_file) if cx_file is not False else None, "n_rows": int(len(db)), "n_complexes": len(cmap),
                        "packaged": db_path is None},
           "n_pairs_on_panel": int(len(pairs)), "summary": summ, "niche_column": niche_col, "placed": [str(x) for x in res.loc[placed, "interaction_name"]],
           "hits_table": json.loads(hits_table.to_json(orient="records")), "layers": layers}
    ctx.write_json("summary.json", out)
    return {"radius_um": rad["radius_um"], "radius_native": params.radius, "mean_other_cells": float(mean_count),
            "matched": {k: rad["matched"][k] for k in ("target", "radius", "mean_other_cells")} if rad["matched"] else None,
            "n_pairs": int(len(pairs)), "n_tested": {t: v["n_tested"] for t, v in summ["thresholds"].items()},
            "n_hits": {t: v["n_hits"] for t, v in summ["thresholds"].items()}, "n_robust": summ["n_robust"], "n_depleted": summ["n_depleted"],
            "robust": summ["robust"]}


def read_lr_summary(run_dir: Path) -> Dict[str, Any]:
    """``12_lr_screen/summary.json`` of a run directory (empty mapping when absent)."""
    f = stage_dir(run_dir, "lr_screen") / "summary.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)
