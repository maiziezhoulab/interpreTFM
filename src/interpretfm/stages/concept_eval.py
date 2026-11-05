"""Stage ``concept_eval``: Held-out concept alignment: concept F1 of SAE features, raw dimensions and the untrained SAE under the row and whole-gene splits.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["concept_f1.json"]


@register_stage("concept_eval", inputs=["sae", "concepts"], outputs=OUTPUTS)
def run_concept_eval(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Held-out concept alignment: concept F1 of SAE features, raw dimensions and the untrained SAE under the row and whole-gene splits."""
    raise NotImplementedError(
        "stage 'concept_eval' is not implemented in this development version; "
        "parameters are read from config.stages.concept_eval (ctx.params)"
    )
