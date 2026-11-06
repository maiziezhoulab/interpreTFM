"""A synthetic model for tests and the tutorial: no weights, no GPU, seconds on a laptop.

``model.name: toy`` writes the same synthetic activations as
:func:`interpretfm.testing.write_toy_store`: one classification-token row per cell
(``cell_pooling.token``, default ``60695``) followed by one row per detected gene, with a
concept direction, a cell-type direction and noise in ``hidden_dim`` (option, default 16)
dimensions. It exercises the whole extraction path (subprocess boundary, cell subsets,
store and manifests) without any model stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from interpretfm.genes import GeneTable
from interpretfm.models.base import ExtractionSpec, ModelExtractor


class ToyExtractor(ModelExtractor):
    name = "toy"
    gene_kind = "symbol"
    needs_checkpoint = False
    uses_device = False

    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path, log) -> Dict[str, Any]:
        from interpretfm.testing import make_toy_concepts, write_toy_store

        self.check(spec, genes)
        hidden_dim = int(spec.options.get("hidden_dim", 16))
        concepts = make_toy_concepts(adata) if spec.options.get("concepts", True) else None
        cls_token = spec.pooling_token or "60695"
        write_toy_store(store_root, adata, layers=list(spec.layers), hidden_dim=hidden_dim, concepts=concepts, seed=spec.seed,
                        rows_per_shard=spec.rows_per_shard, cls_token=cls_token)
        log.info("toy: wrote synthetic activations for %d cells (%d dims)", adata.n_obs, hidden_dim)
        return {"layers_written": list(spec.layers), "n_cells_extracted": int(adata.n_obs), "hidden_dim": hidden_dim,
                "synthetic": True}
