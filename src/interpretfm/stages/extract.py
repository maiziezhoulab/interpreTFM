"""Stage ``extract``: Layer-resolved activation extraction into the sharded store (runs in the model environment).

Extraction with a model adapter is pending. What works today is the cached-store mode: when
``stages.extract.store_root`` points at an existing activation store (the 0.9 sharded format,
``<root>/activations/<layer>/shard_*/``), the stage checks that every configured layer is
present and writes ``store.json`` so that the downstream stages read that store.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.params import param_path
from interpretfm.runner import StageContext, register_stage
from interpretfm.stores import layer_root, list_shards, write_store_spec

OUTPUTS = ['store.json']


@register_stage("extract", inputs=["ingest"], outputs=OUTPUTS, env="model")
def run_extract(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """Layer-resolved activation extraction into the sharded store (runs in the model environment)."""
    store_root = ctx.params.get("store_root")
    if store_root:
        root = param_path(ctx.config, store_root, "extract", "store_root")
        n_shards = {}
        for layer in ctx.config.model.layers:
            if not layer_root(root, layer).is_dir():
                raise FileNotFoundError(f"stages.extract.store_root: layer {layer!r} not found under {root}/activations")
            n_shards[layer] = len(list_shards(root, layer))
        write_store_spec(ctx.stage_dir, root, ctx.config.model.layers, {"source": "cached", "n_shards": n_shards})
        ctx.log.info("extract: using the cached activation store %s (%s)", root, n_shards)
        return {"source": "cached", "store_root": str(root), "n_shards": n_shards}
    raise NotImplementedError(
        "stage 'extract' with a model adapter is not implemented in this development version; "
        "set stages.extract.store_root to use an existing activation store"
    )
