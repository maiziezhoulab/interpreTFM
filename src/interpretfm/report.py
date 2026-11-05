"""The run report: every metric next to its baseline, as ``report.json`` and one self-contained
HTML file (inline CSS, matplotlib figures embedded as PNG; no external resources).

:func:`collect` gathers the stage outputs of a run directory into one JSON-serialisable
mapping; :func:`make_figures` draws the per-layer comparisons; :func:`render_html` writes
the page. The "how to read this" text is in :data:`HOW_TO_READ`.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from interpretfm.layers import CRITERIA
from interpretfm.manifest import RunManifest, StageManifest
from interpretfm.runner import list_stages

#: Fixed colours per entity (a categorical palette validated for adjacent pairs); controls are grey.
COLORS = {"model": "#2a78d6", "sae": "#2a78d6", "raw": "#eb6834", "untrained": "#1baf7a", "genes": "#eda100",
          "control": "#8a8985", "hidden": "#2a78d6"}

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
]


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
    spatial_status = by_name.get("spatial", {}).get("status", "not run")
    spatial_note = {"done": "run", "skipped": "skipped: the dataset has no spatial coordinates",
                    "failed": "failed (see 10_spatial/manifest.json)", "not run": "not run"}.get(spatial_status, spatial_status)
    if spatial_status == "not run" and not cfg.dataset.is_spatial:
        spatial_note = "not run (scRNA-seq mode: no coordinates)"

    qc = _read_json(dirs["ingest"] / "dataset_qc.json") or {}
    table = pd.read_csv(dirs["layers"] / "layer_table.csv") if (dirs["layers"] / "layer_table.csv").exists() else pd.DataFrame()
    rec = _read_json(dirs["layers"] / "recommendations.json") or {}
    sel = _read_json(dirs["cells"] / "selection.json") or {}
    spec = _read_json(dirs["cells"] / "specificity.json") or {}
    sae = _read_json(dirs["sae"] / "selected.json") or {}

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
        "how_to_read": HOW_TO_READ,
    }
    return _clean(report)


# --- figures ------------------------------------------------------------------------------------


def _bar_groups(ax, layers: Sequence[str], series: List[Dict[str, Any]], ylabel: str, hline: Optional[float] = None,
                hline_label: str = "chance") -> None:
    import numpy as np

    n = len(series)
    x = np.arange(len(layers))
    width = 0.8 / max(n, 1)
    for i, s in enumerate(series):
        vals = [np.nan if v is None else float(v) for v in s["values"]]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, label=s["label"], color=s["color"], linewidth=0)
    if hline is not None:
        ax.axhline(hline, color=COLORS["control"], linewidth=1, linestyle="--", label=hline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30 if len(layers) > 3 else 0, ha="right" if len(layers) > 3 else "center")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def make_figures(report: Dict[str, Any]) -> Dict[str, str]:
    """Per-layer comparison figures as base64 PNG strings (``{name: data}``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["layer_table"]
    if not rows:
        return {}
    layers = [str(r["layer"]) for r in rows]

    def col(name):
        return [r.get(name) for r in rows]

    figs: Dict[str, str] = {}
    specs = [
        ("tis", "TIS", [{"label": "model", "values": col("tis_mean"), "color": COLORS["model"]},
                        {"label": "shuffled", "values": col("tis_shuffled_mean"), "color": COLORS["control"]},
                        {"label": "expression PCA", "values": col("tis_pca_mean"), "color": "#c3c2b7"}], None),
        ("concept_f1_row", "Concept F1, row split", [{"label": "SAE", "values": col("concept_f1_row_sae"), "color": COLORS["sae"]},
                                                     {"label": "raw dims", "values": col("concept_f1_row_raw"), "color": COLORS["raw"]},
                                                     {"label": "untrained SAE", "values": col("concept_f1_row_untrained"), "color": COLORS["untrained"]}], None),
        ("concept_f1_gene", "Concept F1, whole-gene split", [{"label": "SAE", "values": col("concept_f1_gene_sae"), "color": COLORS["sae"]},
                                                             {"label": "raw dims", "values": col("concept_f1_gene_raw"), "color": COLORS["raw"]},
                                                             {"label": "untrained SAE", "values": col("concept_f1_gene_untrained"), "color": COLORS["untrained"]}], None),
        ("readout", "Gene-level readout AUROC", [{"label": "hidden state", "values": col("readout_auroc_hidden"), "color": COLORS["hidden"]},
                                                  {"label": "SAE features", "values": col("readout_auroc_sae"), "color": "#7fb0e8"},
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
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        figs[name] = base64.b64encode(buf.getvalue()).decode("ascii")
    return figs


# --- HTML ---------------------------------------------------------------------------------------

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;
margin:0;padding:0 16px 48px;line-height:1.45}main{max-width:1100px;margin:0 auto}h1{font-size:1.6rem;margin:28px 0 4px}
h2{font-size:1.2rem;margin:32px 0 8px;border-bottom:1px solid #e6e5e2;padding-bottom:4px}h3{font-size:1rem;margin:18px 0 6px}
p,li{font-size:0.95rem}.muted{color:#52514e;font-size:0.85rem}table{border-collapse:collapse;font-size:0.82rem;margin:8px 0;
max-width:100%;display:block;overflow-x:auto}th,td{padding:4px 8px;border-bottom:1px solid #e6e5e2;text-align:right;
white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f3f2ef;font-weight:600}
.base{color:#52514e}.warn{background:#fff4e0;border-left:3px solid #eda100;padding:6px 10px;margin:6px 0;font-size:0.9rem}
.grid{display:flex;flex-wrap:wrap;gap:16px}.grid figure{margin:0;flex:1 1 320px;max-width:520px}figure img{width:100%;height:auto}
figcaption{font-size:0.82rem;color:#52514e}code{font-size:0.85em;background:#f3f2ef;padding:1px 4px;border-radius:3px}
.ok{color:#008300}.bad{color:#e34948}
"""


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
    return html.escape(str(v))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], base_cols: Sequence[int] = ()) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>"]
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            cls = ' class="base"' if i in base_cols else ""
            cells.append(f"<td{cls}>{_fmt(v)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_html(report: Dict[str, Any], figures: Dict[str, str]) -> str:
    """The self-contained HTML page."""
    r = report
    run, ds, model = r["run"], r["dataset"], r["model"]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>interpreTFM report: {html.escape(run['name'])}</title><style>{CSS}</style></head><body><main>"]
    parts.append(f"<h1>interpreTFM report: {html.escape(run['name'])}</h1>")
    parts.append(f"<p class='muted'>model {html.escape(model['name'])}, layers {html.escape(', '.join(model['layers']))}; "
                 f"representative layer {html.escape(str(model['representative_layer']))}; package {html.escape(str(run['package_version']))}; "
                 f"updated {html.escape(str(run['updated']))}; run directory <code>{html.escape(run['run_dir'])}</code></p>")

    # how to read
    parts.append("<h2>How to read this report</h2>")
    for item in r["how_to_read"]:
        parts.append(f"<h3>{html.escape(item['title'])}</h3><p>{html.escape(item['text'])}</p>")

    # dataset
    parts.append("<h2>Dataset and input checks</h2>")
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
    parts.append(_table(["check", "value"], rows))
    for w in ds.get("warnings") or []:
        parts.append(f"<div class='warn'>{html.escape(w)}</div>")

    # layer table with baselines
    parts.append("<h2>Layers: every criterion next to its baselines</h2>")
    rec = r.get("recommendations") or {}
    if rec.get("criteria"):
        parts.append("<ul>")
        from interpretfm.layers import notes_text
        for line in notes_text(rec):
            parts.append(f"<li>{html.escape(line)}</li>")
        parts.append("</ul>")
    if r["layer_table"]:
        for c in CRITERIA:
            headers = ["layer", c["name"]] + [b for b in c["baselines"]]
            rows = [[row["layer"], row.get(c["column"])] + [row.get(b) for b in c["baselines"]] for row in r["layer_table"]]
            parts.append(f"<h3>{html.escape(c['name'])} <span class='muted'>({html.escape(c['note'])})</span></h3>")
            parts.append(_table(headers, rows, base_cols=list(range(2, len(headers)))))
    else:
        parts.append("<p class='warn'>No layer table: run the <code>layers</code> stage.</p>")
    if figures:
        parts.append("<div class='grid'>")
        captions = {"tis": "TIS per layer with the shuffled and expression-PCA controls.",
                    "concept_f1_row": "Concept F1 on held-out rows (gene detection).",
                    "concept_f1_gene": "Concept F1 on held-out genes (concept generalization).",
                    "readout": "Linear readout of concept membership on unseen genes; dashed line: chance.",
                    "gsi": "GSI excess (population specificity) of SAE features, raw dimensions and single genes.",
                    "mhr": "Marker-gene hit rate; dashed line: chance (k / genes).",
                    "features": "Retained concept-labelled features; dashed line: the minimum per layer."}
        for name, data in figures.items():
            parts.append(f"<figure><img alt='{html.escape(captions.get(name, name))}' src='data:image/png;base64,{data}'>"
                         f"<figcaption>{html.escape(captions.get(name, name))}</figcaption></figure>")
        parts.append("</div>")

    # SAE and cells
    if r.get("sae"):
        parts.append("<h2>Sparse autoencoders</h2>")
        rows = []
        for layer, e in r["sae"].items():
            q = e.get("qc") or {}
            rows.append([layer, e.get("lambda"), e.get("rule"), q.get("n_latents"), q.get("fvu"), q.get("l0_frac"), q.get("dead_frac")])
        parts.append(_table(["layer", "λ", "rule", "features", "FVU", "L0 fraction", "dead fraction"], rows))
    sel = (r.get("cells") or {}).get("selection") or {}
    if sel:
        parts.append("<h2>Cell-level features</h2>")
        rows = []
        for layer, e in sel.items():
            rows.append([layer, e.get("n_concepts_kept"), e.get("n_features"), e.get("enough_features"), e.get("valid_f1_mean"),
                         e.get("test_f1_mean"), (e.get("projection") or {}).get("n_has_row"), e.get("skipped") or ""])
        parts.append(_table(["layer", "concepts kept", "features", "enough", "valid F1 (mean)", "test F1 (mean)", "cells projected", "note"], rows))
        spec = (r.get("cells") or {}).get("specificity") or {}
        rows = []
        for layer, e in spec.items():
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
        parts.append(_table(["layer", "cells", "populations", "GSI excess SAE", "raw", "genes", "MHR SAE", "raw", "genes", "note"],
                            rows, base_cols=[4, 5, 7, 8]))

    # spatial + stages
    parts.append("<h2>Spatial analyses</h2>")
    parts.append(f"<p>spatial: {html.escape(r['spatial']['note'])}</p>")
    parts.append("<h2>Stages</h2>")
    rows = [[s["stage"], s["status"], s["duration_s"], s.get("error") or ""] for s in r["stages"]]
    parts.append(_table(["stage", "status", "seconds", "error"], rows))
    parts.append("<p class='muted'>All numbers are in <code>report.json</code> next to this file.</p>")
    parts.append("</main></body></html>")
    return "".join(parts)


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


__all__ = ["COLORS", "HOW_TO_READ", "collect", "make_figures", "render_html", "write_report"]
