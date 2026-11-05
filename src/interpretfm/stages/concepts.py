"""Stage ``concepts``: Concept universe, held-out concept F1 (row and gene splits) and the gene-level readout.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['concept_f1.json']


@register_stage("concepts", inputs=["sae"], outputs=OUTPUTS)
def run_concepts(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Concept universe, held-out concept F1 (row and gene splits) and the gene-level readout."""
    raise NotImplementedError(
        "stage 'concepts' is not implemented in this development version; "
        "parameters are read from config.stages.concepts (ctx.params)"
    )
