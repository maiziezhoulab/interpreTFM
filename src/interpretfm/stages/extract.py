"""Stage ``extract``: Layer-resolved activation extraction into the sharded store (runs in the model environment).

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['store.json']


@register_stage("extract", inputs=["ingest"], outputs=OUTPUTS, env="model")
def run_extract(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Layer-resolved activation extraction into the sharded store (runs in the model environment)."""
    raise NotImplementedError(
        "stage 'extract' is not implemented in this development version; "
        "parameters are read from config.stages.extract (ctx.params)"
    )
