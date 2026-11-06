"""The run report: every metric next to its baseline, as ``report.json`` and one self-contained
HTML file (inline CSS, matplotlib figures embedded as PNG; no external resources).

:func:`collect` (:mod:`interpretfm.report.metrics`) gathers the stage outputs of a run
directory into one JSON-serialisable mapping; :func:`make_figures` draws the comparisons and
:func:`render_html` fills the Jinja2 template ``templates/report.html.j2``
(:mod:`interpretfm.report.html`); :func:`write_report` writes both files. The "how to read
this" texts are :data:`HOW_TO_READ` and :data:`SECTION_NOTES`.
"""
from interpretfm.report.html import COLORS, make_figures, render_html, write_report
from interpretfm.report.metrics import HOW_TO_READ, SECTION_NOTES, collect, interface_metrics, lr_metrics, niche_metrics

__all__ = ["COLORS", "HOW_TO_READ", "SECTION_NOTES", "collect", "interface_metrics", "lr_metrics", "make_figures", "niche_metrics",
           "render_html", "write_report"]
