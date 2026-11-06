"""Stage ``report``: ``report.json`` with every metric next to its baseline and one self-contained
HTML page (inline CSS, embedded PNG figures, no external resources).

Reads the outputs of every finished stage in the run directory (:func:`interpretfm.report.collect`);
``layers`` must be done unless ``stages.report.spatial_only`` is true (a run of the spatial stages
on published projected features, without the model-level stages). The spatial stage is reported
as run, skipped or not run. ``output.html`` switches the HTML page; ``report.json`` is always written.

Outputs: ``report.json`` and, with ``output.html``, ``report.html``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.report import write_report
from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["report.json"]


@register_stage("report", inputs=["ingest", "layers"], outputs=OUTPUTS, uses_labels=True, input_overrides={"layers": "spatial_only"})
def run_report(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """HTML and JSON report with every metric next to its baseline."""
    cfg = ctx.config
    spatial_only = bool(ctx.params.get("spatial_only", False))
    for name in ("ingest",) if spatial_only else ("ingest", "layers"):  # a clear StageError when the run is incomplete
        ctx.input_dir(name)
    report = write_report(cfg, ctx.run_dir, ctx.stage_dir, html_page=cfg.output.html)
    ctx.log.info("report: %s%s", ctx.stage_dir / "report.json", f" and {ctx.stage_dir / 'report.html'}" if cfg.output.html else "")
    ctx.log.info("report: spatial: %s", report["spatial"]["note"])
    return {"html": cfg.output.html, "n_layers": len(report["layer_table"]), "spatial": report["spatial"]["status"]}
