"""The HTML page of the report: figures (matplotlib, embedded as PNG) and a Jinja2 template.

:func:`make_figures` draws one figure per comparison from ``report.json`` (layer criteria with
their baselines, niches against their baselines, the interface shares and false-positive
rates, the ligand–receptor screen); :func:`render_html` turns the report and the figures into
one self-contained page with ``templates/report.html.j2``; :func:`write_report` writes
``report.json`` (always) and ``report.html``.

Every section of the page carries a short note (:data:`interpretfm.report.metrics.SECTION_NOTES`)
on what its numbers measure and what they do not. Sections whose stage did not run render as
"not run". Wording is neutral: numbers next to their baselines, no verdicts.
"""
from __future__ import annotations

import base64
import html as _html
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from interpretfm.layers import CRITERIA
from interpretfm.report.metrics import collect

#: Fixed colours per entity (a categorical palette validated for adjacent pairs); controls are grey.
COLORS = {"model": "#2a78d6", "sae": "#2a78d6", "raw": "#eb6834", "untrained": "#1baf7a", "genes": "#eda100",
          "control": "#8a8985", "hidden": "#2a78d6", "light": "#c3c2b7", "sae2": "#7fb0e8"}

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE = "report.html.j2"

CAPTIONS = {"tis": "TIS per layer with the shuffled and expression-PCA controls.",
            "concept_f1_row": "Concept F1 on held-out rows (gene detection).",
            "concept_f1_gene": "Concept F1 on held-out genes (concept generalization).",
            "readout": "Linear readout of concept membership on unseen genes; dashed line: chance.",
            "gsi": "GSI excess (population specificity) of SAE features, raw dimensions and single genes.",
            "mhr": "Marker-gene hit rate; dashed line: chance (k / genes).",
            "features": "Retained concept-labelled features; dashed line: the minimum per layer.",
            "niches_agreement": "Niche agreement with the labels (ARI, NMI): the model's partition beside the expression-only partition "
                                "(its own winner and the model's configuration) and the label-composition partition.",
            "niches_stability": "Seed stability (median seed-to-seed ARI at the finer k): model beside expression-only.",
            "interfaces_share": "Share of pairs with |z| > 2 under the legacy tile null and the calibrated null, with the "
                                "false-positive shares of smooth contact-free fields (legacy and calibrated) and of i.i.d. noise.",
            "lr_screen": "Ligand–receptor screen at the primary threshold: fold against z for every tested pair; filled points are hits "
                         "(q < q_max and fold ≥ fold_min), ringed points robust hits; dashed lines at fold 1 and fold_min.",
            "lr_placement": "Niche placement of the placed pairs (share of close pairs in the niche over the niche's share of cells; "
                            "dashed line: 1, as expected from the niche's size), per layer."}


# --- figures ------------------------------------------------------------------------------------


def _bar_groups(ax, labels: Sequence[str], series: List[Dict[str, Any]], ylabel: str, hline: Optional[float] = None,
                hline_label: str = "chance") -> None:
    import numpy as np

    n = len(series)
    x = np.arange(len(labels))
    width = 0.8 / max(n, 1)
    for i, s in enumerate(series):
        vals = [np.nan if v is None else float(v) for v in s["values"]]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, label=s["label"], color=s["color"], linewidth=0)
    if hline is not None:
        ax.axhline(hline, color=COLORS["control"], linewidth=1, linestyle="--", label=hline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _layer_figures(report: Dict[str, Any], plt) -> Dict[str, str]:
    rows = report.get("layer_table") or []
    if not rows:
        return {}
    layers = [str(r["layer"]) for r in rows]

    def col(name):
        return [r.get(name) for r in rows]

    figs: Dict[str, str] = {}
    specs = [
        ("tis", "TIS", [{"label": "model", "values": col("tis_mean"), "color": COLORS["model"]},
                        {"label": "shuffled", "values": col("tis_shuffled_mean"), "color": COLORS["control"]},
                        {"label": "expression PCA", "values": col("tis_pca_mean"), "color": COLORS["light"]}], None),
        ("concept_f1_row", "Concept F1, row split", [{"label": "SAE", "values": col("concept_f1_row_sae"), "color": COLORS["sae"]},
                                                     {"label": "raw dims", "values": col("concept_f1_row_raw"), "color": COLORS["raw"]},
                                                     {"label": "untrained SAE", "values": col("concept_f1_row_untrained"), "color": COLORS["untrained"]}], None),
        ("concept_f1_gene", "Concept F1, whole-gene split", [{"label": "SAE", "values": col("concept_f1_gene_sae"), "color": COLORS["sae"]},
                                                             {"label": "raw dims", "values": col("concept_f1_gene_raw"), "color": COLORS["raw"]},
                                                             {"label": "untrained SAE", "values": col("concept_f1_gene_untrained"), "color": COLORS["untrained"]}], None),
        ("readout", "Gene-level readout AUROC", [{"label": "hidden state", "values": col("readout_auroc_hidden"), "color": COLORS["hidden"]},
                                                  {"label": "SAE features", "values": col("readout_auroc_sae"), "color": COLORS["sae2"]},
                                                  {"label": "expression control", "values": col("readout_auroc_expression_control"), "color": COLORS["genes"]},
                                                  {"label": "random", "values": col("readout_auroc_random"), "color": COLORS["control"]}], 0.5),
        ("gsi", "GSI excess", [{"label": "SAE", "values": col("gsi_excess_sae"), "color": COLORS["sae"]},
                               {"label": "raw dims (±)", "values": col("gsi_excess_raw"), "color": COLORS["raw"]},
                               {"label": "single genes", "values": col("gsi_excess_genes"), "color": COLORS["genes"]}], None),
        ("mhr", "MHR", [{"label": "SAE", "values": col("mhr_sae"), "color": COLORS["sae"]},
                        {"label": "raw dims (±)", "values": col("mhr_raw"), "color": COLORS["raw"]},
                        {"label": "single genes", "values": col("mhr_genes"), "color": COLORS["genes"]}],
         next((v for v in col("mhr_chance") if v is not None), None)),
        ("features", "Retained features", [{"label": "retained", "values": col("n_features_retained"), "color": COLORS["sae"]}],
         next((v for v in col("min_features") if v is not None), None)),
    ]
    for name, ylabel, series, hline in specs:
        if all(v is None for s in series for v in s["values"]):
            continue
        fig, ax = plt.subplots(figsize=(max(4.4, 0.9 * len(layers) + 3.4), 2.6), dpi=110)
        _bar_groups(ax, layers, series, ylabel, hline, "minimum" if name == "features" else "chance")
        fig.tight_layout()
        figs[name] = _png(fig)
    return figs


def _niche_figures(report: Dict[str, Any], plt) -> Dict[str, str]:
    niches = {k: v for k, v in (report.get("niches") or {}).items() if not v.get("skipped")}
    if not niches:
        return {}
    layers = list(niches)
    figs: Dict[str, str] = {}

    def pick(e, *path):
        d = e
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d

    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 1.6 * len(layers) + 5.6), 2.7), dpi=110)
    for ax, metric in zip(axes, ("ari", "nmi")):
        series = [{"label": "model", "values": [pick(e, "model", metric) for e in niches.values()], "color": COLORS["model"]},
                  {"label": "expression (own winner)", "values": [pick(e, "baselines", "expression_winner", metric) for e in niches.values()], "color": COLORS["raw"]},
                  {"label": "expression (model config.)", "values": [pick(e, "baselines", "expression_at_model_config", metric) for e in niches.values()], "color": "#f4a37f"},
                  {"label": "label composition", "values": [pick(e, "baselines", "label_composition", metric) for e in niches.values()], "color": COLORS["genes"]}]
        _bar_groups(ax, layers, series, metric.upper() + " vs labels")
        if ax is axes[0]:
            ax.get_legend().remove()
    fig.tight_layout()
    figs["niches_agreement"] = _png(fig)
    stab = [(pick(e, "seed_stability", "model"), pick(e, "seed_stability", "expression")) for e in niches.values()]
    if any(v is not None for pair in stab for v in pair):
        fig, ax = plt.subplots(figsize=(max(4.4, 0.9 * len(layers) + 3.4), 2.5), dpi=110)
        _bar_groups(ax, layers, [{"label": "model", "values": [s[0] for s in stab], "color": COLORS["model"]},
                                 {"label": "expression-only", "values": [s[1] for s in stab], "color": COLORS["raw"]}], "median seed ARI")
        fig.tight_layout()
        figs["niches_stability"] = _png(fig)
    return figs


def _interface_figures(report: Dict[str, Any], plt) -> Dict[str, str]:
    itf = {k: v for k, v in (report.get("interfaces") or {}).items() if not v.get("skipped")}
    if not itf:
        return {}
    layers = list(itf)
    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 1.6 * len(layers) + 5.6), 2.7), dpi=110)
    _bar_groups(axes[0], layers, [{"label": "legacy tile null", "values": [e["share_abs_z_gt2"].get("legacy_tile") for e in itf.values()], "color": COLORS["control"]},
                                  {"label": "calibrated null", "values": [e["share_abs_z_gt2"].get("calibrated") for e in itf.values()], "color": COLORS["model"]}],
                "share of pairs with |z| > 2")
    axes[0].get_legend().remove()
    _bar_groups(axes[1], layers, [{"label": "smooth field, legacy", "values": [e["fpr"].get("smooth_field_legacy") for e in itf.values()], "color": COLORS["control"]},
                                  {"label": "smooth field, calibrated", "values": [e["fpr"].get("smooth_field_calibrated") for e in itf.values()], "color": COLORS["model"]},
                                  {"label": "i.i.d. noise", "values": [e["fpr"].get("iid_noise") for e in itf.values()], "color": COLORS["light"]}],
                "false-positive share of |z| > 2", 0.05, "5 %")
    fig.tight_layout()
    return {"interfaces_share": _png(fig)}


def _lr_figures(report: Dict[str, Any], plt) -> Dict[str, str]:
    import numpy as np

    lr = report.get("lr_screen") or {}
    if not lr or not lr.get("hits_table") and not lr.get("tested"):
        return {}
    figs: Dict[str, str] = {}
    p0 = lr.get("primary")
    rows = lr.get("screen_rows") or []
    if rows and p0:
        fold = np.array([r.get(f"fold_thr{p0}") if r.get(f"fold_thr{p0}") is not None else np.nan for r in rows], float)
        z = np.array([r.get(f"z_thr{p0}") if r.get(f"z_thr{p0}") is not None else np.nan for r in rows], float)
        hit = np.array([bool(r.get(f"hit_thr{p0}")) for r in rows])
        robust = np.array([bool(r.get("robust_hit")) for r in rows])
        ok = np.isfinite(fold) & np.isfinite(z)
        if ok.any():
            fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=110)
            ax.scatter(fold[ok & ~hit], z[ok & ~hit], s=12, facecolors="none", edgecolors=COLORS["control"], linewidths=0.7, label="tested pair")
            ax.scatter(fold[ok & hit], z[ok & hit], s=18, color=COLORS["model"], label="hit")
            ax.scatter(fold[ok & robust], z[ok & robust], s=60, facecolors="none", edgecolors=COLORS["raw"], linewidths=1.2, label="robust hit")
            for v in (1.0, lr.get("fold_min") or 1.2):
                ax.axvline(v, color=COLORS["control"], linewidth=0.8, linestyle="--")
            ax.set_xlabel(f"fold (observed + 1) / (null mean + 1), ≥ {p0} transcript(s)")
            ax.set_ylabel("z")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            fig.tight_layout()
            figs["lr_screen"] = _png(fig)
    placement = {layer: e for layer, e in (lr.get("placement") or {}).items() if not e.get("skipped") and e.get("ratios")}
    if placement:
        fig, axes = plt.subplots(1, len(placement), figsize=(max(4.4, 3.4 * len(placement) + 1.6), 2.8), dpi=110, squeeze=False)
        for ax, (layer, e) in zip(axes[0], placement.items()):
            names = list(e["ratios"])
            niches = sorted({n for r in e["ratios"].values() for n in r}, key=lambda v: (len(v), v))
            x = np.arange(len(names))
            width = 0.8 / max(len(niches), 1)
            cmap = plt.get_cmap("tab10")
            for i, nn in enumerate(niches):
                vals = [e["ratios"][name].get(nn, np.nan) for name in names]
                ax.bar(x + (i - (len(niches) - 1) / 2) * width, vals, width * 0.9, label=f"niche {nn}", color=cmap(i % 10), linewidth=0)
            ax.axhline(1.0, color=COLORS["control"], linewidth=0.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("placement ratio")
            ax.set_title(layer, fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        figs["lr_placement"] = _png(fig)
    return figs


def make_figures(report: Dict[str, Any]) -> Dict[str, str]:
    """Every figure of the page as base64 PNG strings (``{name: data}``); sections without data draw nothing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: Dict[str, str] = {}
    figs.update(_layer_figures(report, plt))
    figs.update(_niche_figures(report, plt))
    figs.update(_interface_figures(report, plt))
    figs.update(_lr_figures(report, plt))
    return figs


# --- HTML ---------------------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "–"
        return f"{v:.{digits}g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], base_cols: Sequence[int] = ()) -> Dict[str, Any]:
    """A table for the template: headers and rows of ``{"value": text, "base": bool}`` cells."""
    return {"headers": [str(h) for h in headers],
            "rows": [[{"value": _fmt(v), "base": i in base_cols} for i, v in enumerate(r)] for r in rows]}


def _dataset_table(ds: Dict[str, Any]) -> Dict[str, Any]:
    qc = ds.get("qc") or {}
    cov = ds.get("vocab_coverage") or {}
    labels = ds.get("labels")
    rows = [["cells", ds.get("n_cells")], ["genes", ds.get("n_genes")], ["mode", ds.get("mode")],
            [f"cells flagged by QC (< {qc.get('min_transcripts')} transcripts or < {qc.get('min_genes')} genes)",
             f"{qc.get('n_flagged')} ({100 * (qc.get('fraction_flagged') or 0):.1f}%), kept and flagged" if qc else "–"],
            ["genes in the model vocabulary", (f"{cov.get('n_in_vocab')}/{cov.get('n_genes')} ({100 * cov['fraction']:.1f}%), from {cov.get('source')}"
                                               if cov.get("fraction") is not None else f"not computed: {cov.get('note', '')}")],
            ["labels", f"{labels['column']}: {labels['n_labels']} labels, {labels.get('n_labels_ge_30_cells')} with ≥ 30 cells" if labels else "none"],
            ["coordinates", f"{ds['spatial']['obsm']}, {ds['spatial']['um_per_unit']} µm per unit" if ds.get("spatial") else "none"]]
    return _table(["check", "value"], rows)


def _criterion_tables(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for c in CRITERIA:
        headers = ["layer", c["name"]] + [b for b in c["baselines"]]
        rows = [[row["layer"], row.get(c["column"])] + [row.get(b) for b in c["baselines"]] for row in r["layer_table"]]
        out.append({"title": c["name"], "note": c["note"], "table": _table(headers, rows, base_cols=list(range(2, len(headers))))})
    return out


def _sae_table(sae: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for layer, e in sae.items():
        q = e.get("qc") or {}
        rows.append([layer, e.get("lambda"), e.get("rule"), q.get("n_latents"), q.get("fvu"), q.get("l0_frac"), q.get("dead_frac")])
    return _table(["layer", "λ", "rule", "features", "FVU", "L0 fraction", "dead fraction"], rows)


def _cell_tables(cells: Dict[str, Any]) -> Dict[str, Any]:
    sel = cells.get("selection") or {}
    rows = [[layer, e.get("n_concepts_kept"), e.get("n_features"), e.get("enough_features"), e.get("valid_f1_mean"), e.get("test_f1_mean"),
             (e.get("projection") or {}).get("n_has_row"), e.get("skipped") or ""] for layer, e in sel.items()]
    selection = _table(["layer", "concepts kept", "features", "enough", "valid F1 (mean)", "test F1 (mean)", "cells projected", "note"], rows)
    rows = []
    for layer, e in (cells.get("specificity") or {}).items():
        if e.get("skipped"):
            rows.append([layer, "–", "–", "–", "–", "–", "–", "–", "–", e["skipped"]])
            continue
        reps = e.get("representations") or {}

        def g(rep, *k):
            d = reps.get(rep) or {}
            for kk in k:
                d = d.get(kk) if isinstance(d, dict) else None
                if d is None:
                    return None
            return d
        rows.append([layer, e.get("n_cells"), e.get("n_populations"), g("sae", "gsi", "excess_mean"), g("raw", "gsi", "excess_mean"),
                     g("genes", "gsi", "excess_mean"), g("sae", "mhr", "mhr"), g("raw", "mhr", "mhr"), g("genes", "mhr", "mhr"),
                     f"chance {_fmt(g('sae', 'mhr', 'chance'))}, random-feature null {_fmt(g('sae', 'mhr', 'null_mean'))}; "
                     f"{e.get('n_qc_flagged_used', 0)} QC-flagged cells used"])
    specificity = _table(["layer", "cells", "populations", "GSI excess SAE", "raw", "genes", "MHR SAE", "raw", "genes", "note"], rows, base_cols=[4, 5, 7, 8])
    return {"selection": selection, "specificity": specificity}


def _niche_table(niches: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for layer, e in niches.items():
        if e.get("skipped"):
            rows.append([layer, e["skipped"]] + ["–"] * 9)
            continue
        m, b = e["model"], e["baselines"]
        ew, ea, lc = b.get("expression_winner") or {}, b.get("expression_at_model_config") or {}, b.get("label_composition") or {}
        rows.append([layer, e.get("config"), m.get("ari"), m.get("nmi"), m.get("seed_ari"),
                     f"{_fmt(m.get('purity'))} (null {_fmt(m.get('purity_null_mean'))}, z {_fmt(m.get('purity_z'), 4)})",
                     f"{_fmt(ew.get('ari'))} / {_fmt(ew.get('nmi'))} at {ew.get('config') or '–'}" if ew and not ew.get("skipped") else "–",
                     f"{_fmt(ea.get('ari'))} / {_fmt(ea.get('nmi'))}" if ea else "–",
                     f"{_fmt(lc.get('ari'))} / {_fmt(lc.get('nmi'))} at {lc.get('config') or '–'}" if lc and not lc.get("skipped") else "–",
                     f"{_fmt((e.get('seed_stability') or {}).get('model'))} vs expression {_fmt((e.get('seed_stability') or {}).get('expression'))}",
                     "; ".join(f"niche {n}: {', '.join(v[:3])}" for n, v in (e.get("top_concepts") or {}).items())])
    return _table(["layer", "selected configuration", "ARI", "NMI", "seed ARI", "purity (model)",
                   "expression-only winner ARI / NMI", "expression at the model's configuration",
                   "label composition ARI / NMI", "seed stability (median ARI, k 4-5)", "top concepts"], rows, base_cols=[6, 7, 8])


def _interface_table(itf: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for layer, e in itf.items():
        if e.get("skipped"):
            rows.append([layer, e["skipped"]] + ["–"] * 8)
            continue
        sh, zs, fp, c, eff = e["share_abs_z_gt2"], e["z_star"], e["fpr"], e["calls"], e["effect_pct"]
        rows.append([layer, f"{e['n_pairs']['niche']} / {e['n_pairs']['free']}",
                     f"{_fmt(sh.get('calibrated'))} vs tile null {_fmt(sh.get('legacy_tile'))}",
                     f"{_fmt(zs.get('niche'))} / {_fmt(zs.get('free'))}",
                     f"smooth {_fmt(fp.get('smooth_field_calibrated'))} (tile null {_fmt(fp.get('smooth_field_legacy'))}), noise {_fmt(fp.get('iid_noise'))}",
                     f"{c['niche']['enriched']} / {c['niche']['depleted']}", f"{c['free']['enriched']} / {c['free']['depleted']}",
                     f"{_fmt(eff.get('median_abs'))} / {_fmt(eff.get('max_abs'))}",
                     f"P {_fmt(e['neighbour_corr'].get('P'))}, P_adj {_fmt(e['neighbour_corr'].get('P_adj'))}",
                     f"{e.get('niche_column')}; {', '.join(e.get('covariates') or [])}"])
    return _table(["layer", "pairs (niche / niche-free)", "share |z| > 2 (calibrated vs tile null)", "z* (niche / free)",
                   "false-positive share of |z| > 2", "enriched / depleted (niche)", "enriched / depleted (niche-free)",
                   "median / max |effect| %", "neighbour correlation", "niche partition; covariates"], rows, base_cols=[2, 4])


def _lr_tables(lr: Dict[str, Any]) -> Dict[str, Any]:
    thr = lr.get("thresholds") or []
    m = lr.get("matched")
    radius = f"{_fmt(lr.get('radius_um'), 4)} µm = {_fmt(lr.get('radius_native'), 5)} native units"
    if m:
        radius += f" (matched to a mean of {_fmt(m.get('target'), 4)} other cells within the radius)"
    summary_rows = [["radius", radius], ["mean number of other cells within the radius", lr.get("mean_other_cells")],
                    ["cells / edges within the radius", f"{lr.get('n_cells')} / {lr.get('n_edges')}"],
                    ["labels of the within-type null", f"{(lr.get('labels') or {}).get('column')} ({(lr.get('labels') or {}).get('n')} types)"],
                    ["database pairs / pairs with every gene on the panel", f"{(lr.get('database') or {}).get('n_rows')} / {lr.get('n_pairs_on_panel')}"],
                    ["permutations, seed, minimum positive cells", f"{lr.get('n_perm')}, {lr.get('seed')}, {lr.get('min_positive')}"],
                    ["hit rule", f"q < {lr.get('q_max')} and fold ≥ {lr.get('fold_min')}; robust = hit at every threshold; depleted = q < {lr.get('q_max')} "
                                 f"and fold ≤ 1/{lr.get('fold_min')}"]]
    for t in thr:
        summary_rows.append([f"tested / hits at ≥ {t} transcript(s)", f"{(lr.get('tested') or {}).get(t)} / {(lr.get('hits') or {}).get(t)}"])
    summary_rows.append(["robust hits", f"{lr.get('n_robust')}: {', '.join(lr.get('robust') or []) or '–'}"])
    summary_rows.append([f"depleted at ≥ {lr.get('primary')} transcript(s)", f"{lr.get('n_depleted')}: {', '.join(lr.get('depleted') or []) or '–'}"])
    headers = ["pair", "pathway"]
    for t in thr:
        headers += [f"positive cells L / R (≥ {t})", f"observed (≥ {t})", f"fold (≥ {t})", f"z (≥ {t})", f"q (≥ {t})", f"hit (≥ {t})"]
    headers += ["robust", "top cell-type pairs (ligand cell → receptor cell)"]
    rows = []
    for h in lr.get("hits_table") or []:
        row = [h.get("interaction_name"), h.get("pathway_name")]
        for t in thr:
            row += [f"{h.get(f'n_lig_pos_thr{t}')} / {h.get(f'n_rec_pos_thr{t}')}", h.get(f"obs_thr{t}"), h.get(f"fold_thr{t}"), h.get(f"z_thr{t}"),
                    h.get(f"q_thr{t}"), bool(h.get(f"hit_thr{t}")) if h.get(f"hit_thr{t}") is not None else None]
        row += [bool(h.get("robust_hit")), h.get("top_celltype_pairs") or ""]
        rows.append(row)
    placement = []
    for layer, e in (lr.get("placement") or {}).items():
        if e.get("skipped"):
            placement.append({"layer": layer, "skipped": e["skipped"]})
            continue
        niches = sorted({n for r in (e.get("ratios") or {}).values() for n in r}, key=lambda v: (len(v), v))
        prow = [[name] + [ratios.get(n) for n in niches] for name, ratios in (e.get("ratios") or {}).items()]
        placement.append({"layer": layer, "n_niches": e.get("n_niches"), "n_cells_with_niche": e.get("n_cells_with_niche"),
                          "table": _table(["pair"] + [f"niche {n}" for n in niches], prow)})
    return {"summary": _table(["quantity", "value"], summary_rows), "hits": _table(headers, rows), "placement": placement,
            "niche_column": lr.get("niche_column")}


def _status_text(stage_rows: List[Dict[str, Any]], name: str, mode_spatial: bool) -> str:
    st = next((s["status"] for s in stage_rows if s["stage"] == name), "not run")
    if st == "skipped":
        return "skipped: the dataset has no spatial coordinates"
    if st == "not run" and not mode_spatial:
        return "not run (scRNA-seq mode: no coordinates)"
    return st


def render_html(report: Dict[str, Any], figures: Dict[str, str]) -> str:
    """The self-contained HTML page from ``report.json`` content and the figures (base64 PNG)."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from interpretfm.layers import notes_text

    r = report
    spatial = r["dataset"].get("mode") == "spatial"
    ctx: Dict[str, Any] = {
        "run": r["run"], "model": r["model"], "dataset": r["dataset"], "how_to_read": r.get("how_to_read") or [],
        "notes": r.get("section_notes") or {}, "figures": figures, "captions": CAPTIONS,
        "dataset_table": _dataset_table(r["dataset"]), "warnings": r["dataset"].get("warnings") or [],
        "recommendation_lines": notes_text(r["recommendations"]) if (r.get("recommendations") or {}).get("criteria") else [],
        "criterion_tables": _criterion_tables(r) if r.get("layer_table") else [],
        "layer_figures": [n for n in ("tis", "concept_f1_row", "concept_f1_gene", "readout", "gsi", "mhr", "features") if n in figures],
        "sae_table": _sae_table(r["sae"]) if r.get("sae") else None,
        "cell_tables": _cell_tables(r["cells"]) if (r.get("cells") or {}).get("selection") else None,
        "spatial_note": r["spatial"]["note"],
        "niches": {"status": _status_text(r["stages"], "niches", spatial), "table": _niche_table(r["niches"]) if r.get("niches") else None,
                   "figures": [n for n in ("niches_agreement", "niches_stability") if n in figures]},
        "interfaces": {"status": _status_text(r["stages"], "interfaces", spatial), "table": _interface_table(r["interfaces"]) if r.get("interfaces") else None,
                       "figures": [n for n in ("interfaces_share",) if n in figures]},
        "lr_screen": {"status": _status_text(r["stages"], "lr_screen", spatial), "tables": _lr_tables(r["lr_screen"]) if r.get("lr_screen") else None,
                      "figures": [n for n in ("lr_screen", "lr_placement") if n in figures]},
        "stages_table": _table(["stage", "status", "seconds", "error"], [[s["stage"], s["status"], s["duration_s"], s.get("error") or ""] for s in r["stages"]]),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True)
    env.filters["e"] = _html.escape
    return env.get_template(TEMPLATE).render(**ctx)


def write_report(cfg, run_dir: Union[str, Path], out_dir: Union[str, Path], html_page: bool = True) -> Dict[str, Any]:
    """Collect, draw and write ``report.json`` (always) and ``report.html`` (when ``html_page``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = collect(cfg, run_dir)
    with open(out_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    if html_page:
        figures = make_figures(report)
        (out_dir / "report.html").write_text(render_html(report, figures), encoding="utf-8")
    return report


__all__ = ["CAPTIONS", "COLORS", "TEMPLATE", "TEMPLATE_DIR", "make_figures", "render_html", "write_report"]
