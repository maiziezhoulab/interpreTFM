"""Stage ``gene_readout``: Gene-level linear readout of concept membership on unseen genes, with expression and random controls.

Stub: the signature, inputs and outputs are final; the implementation is pending.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["gene_readout.json"]


@register_stage("gene_readout", inputs=["extract", "sae", "concepts"], outputs=OUTPUTS)
def run_gene_readout(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Gene-level linear readout of concept membership on unseen genes, with expression and random controls."""
    raise NotImplementedError(
        "stage 'gene_readout' is not implemented in this development version; "
        "parameters are read from config.stages.gene_readout (ctx.params)"
    )
