"""Stage ``tis``: Transcriptomic interpretability score per layer, with shuffled and PCA controls.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['tis.json']


@register_stage("tis", inputs=["extract"], outputs=OUTPUTS)
def run_tis(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Transcriptomic interpretability score per layer, with shuffled and PCA controls."""
    raise NotImplementedError(
        "stage 'tis' is not implemented in this development version; "
        "parameters are read from config.stages.tis (ctx.params)"
    )
