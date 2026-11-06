"""C2S-Scale: registered so that the name is known; the adapter is wired in a later step.

The library adapter (:mod:`interp_pipeline.adapters.models.c2s_scale`) and the raw-dump
conversion exist; connecting them to the ``extract`` stage (cell sentences, last-token
pooling, fp16 store) is a separate package. Until then ``model.name: c2s_scale`` extracts
only from a cached store (``stages.extract.store_root``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from interpretfm.genes import GeneTable
from interpretfm.models.base import ExtractionError, ExtractionSpec, ModelExtractor


class C2SScaleExtractor(ModelExtractor):
    name = "c2s_scale"
    gene_kind = "symbol"

    def check(self, spec: ExtractionSpec, genes: GeneTable) -> Dict[str, Any]:
        raise ExtractionError("model.name c2s_scale: fresh extraction is not wired to the extract stage in this version; "
                              "use stages.extract.store_root with an existing C2S-Scale activation store")

    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path, log) -> Dict[str, Any]:
        return self.check(spec, genes)
