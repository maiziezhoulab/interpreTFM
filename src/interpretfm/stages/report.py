"""Stage ``report``: HTML and JSON report with every metric next to its baseline.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['metrics.json']


@register_stage("report", inputs=["layers"], outputs=OUTPUTS)
def run_report(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """HTML and JSON report with every metric next to its baseline."""
    raise NotImplementedError(
        "stage 'report' is not implemented in this development version; "
        "parameters are read from config.stages.report (ctx.params)"
    )
