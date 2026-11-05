"""Stage ``sae``: SAE training per layer with the fixed lambda-selection rule, plus the untrained baseline.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['selected.json']


@register_stage("sae", inputs=["extract"], outputs=OUTPUTS)
def run_sae(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """SAE training per layer with the fixed lambda-selection rule, plus the untrained baseline."""
    raise NotImplementedError(
        "stage 'sae' is not implemented in this development version; "
        "parameters are read from config.stages.sae (ctx.params)"
    )
