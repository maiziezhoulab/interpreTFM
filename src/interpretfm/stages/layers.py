"""Stage ``layers``: Per-criterion layer summary and recommendations (no composite score).

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['layer_table.csv']


@register_stage("layers", inputs=["cells"], outputs=OUTPUTS)
def run_layers(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Per-criterion layer summary and recommendations (no composite score)."""
    raise NotImplementedError(
        "stage 'layers' is not implemented in this development version; "
        "parameters are read from config.stages.layers (ctx.params)"
    )
