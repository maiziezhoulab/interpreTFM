"""Model adapters for the ``extract`` stage (imported only inside the model environment).

``ADAPTERS`` maps ``model.name`` to the class implementing
:class:`interpretfm.models.base.ModelExtractor`; :func:`get_extractor` imports it lazily so
that listing the names never loads a model stack. Available: ``scgpt`` (vendored scGPT,
gene symbols), ``geneformer`` (Hugging Face Geneformer, Ensembl ids), ``toy`` (a synthetic
model for tests and the tutorial, no weights), ``c2s_scale`` (Hugging Face Gemma-2 C2S-Scale,
cell sentences of gene symbols).
"""
from __future__ import annotations

import importlib
from typing import Dict, List

from interpretfm.models.base import ExtractionError, ExtractionSpec, ModelExtractor

ADAPTERS: Dict[str, str] = {
    "scgpt": "interpretfm.models.scgpt:ScGPTExtractor",
    "geneformer": "interpretfm.models.geneformer:GeneformerExtractor",
    "toy": "interpretfm.models.toy:ToyExtractor",
    "c2s_scale": "interpretfm.models.c2s_scale:C2SScaleExtractor",
}


def available_models() -> List[str]:
    return sorted(ADAPTERS)


def get_extractor(name: str) -> ModelExtractor:
    """Instantiate the adapter registered for ``model.name``.

    Raises :class:`ExtractionError` for an unknown name or when the adapter's imports fail
    (the message names the environment file to use).
    """
    key = str(name).lower().replace("-", "_")
    if key not in ADAPTERS:
        raise ExtractionError(f"model.name: no extraction adapter for {name!r}; available: {available_models()}")
    module_name, _, cls_name = ADAPTERS[key].partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ExtractionError(f"model.name {name!r}: the adapter's dependencies are not installed in this "
                              f"environment ({e}); run the extract stage in the model's environment "
                              f"(model.env, envs/{key}.yml)") from e
    cls = getattr(module, cls_name)
    return cls()


__all__ = ["ADAPTERS", "ExtractionError", "ExtractionSpec", "ModelExtractor", "available_models", "get_extractor"]
