"""Stage ``spatial``: Niche sweep, calibrated interface test and ligand-receptor screen (spatial datasets only).

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ['niches/selected.json']


@register_stage("spatial", inputs=["cells"], outputs=OUTPUTS, applies=lambda cfg: cfg.dataset.is_spatial)
def run_spatial(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Niche sweep, calibrated interface test and ligand-receptor screen (spatial datasets only)."""
    raise NotImplementedError(
        "stage 'spatial' is not implemented in this development version; "
        "parameters are read from config.stages.spatial (ctx.params)"
    )
