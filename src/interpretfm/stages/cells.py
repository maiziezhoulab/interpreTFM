"""Stage ``cells``: Feature retention and projection to cells; GSI and MHR@100 against single genes.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['specificity.json']


@register_stage("cells", inputs=["concepts"], outputs=OUTPUTS)
def run_cells(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Feature retention and projection to cells; GSI and MHR@100 against single genes."""
    raise NotImplementedError(
        "stage 'cells' is not implemented in this development version; "
        "parameters are read from config.stages.cells (ctx.params)"
    )
