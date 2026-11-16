"""Randomly re-initialized model weights (``stages.extract.weights``), for scGPT and Geneformer.

``trained`` keeps the loaded weights. ``random_all`` re-initializes every parameter with the
module's own initialization; ``random_table`` does the same and then copies the trained
gene/token-embedding table back, so that only the table is the pretrained one. The generator is
seeded once (``torch.manual_seed(seed)``) right before the re-initialization:

- scGPT: ``reset_parameters()`` (or ``_reset_parameters()``) of every submodule in
  ``modules()`` order (the PyTorch defaults), then the model's own ``init_weights()``;
  embedding table ``encoder.embedding.weight``;
- Geneformer (Hugging Face BERT): ``model.apply(model._init_weights)``; embedding table
  ``bert.embeddings.word_embeddings.weight``.

The draws come from the generator of the device the parameters are on, so the same seed gives
different weights on a GPU and on the CPU; the adapters re-initialize after the model has been
moved to the extraction device.
"""
from __future__ import annotations

from typing import Any, Dict

from interpretfm.models.base import ExtractionError

WEIGHTS = ("trained", "random_table", "random_all")
#: models with a re-initialization rule, and the name of their gene/token-embedding table
EMBEDDING_TABLE = {"scgpt": "encoder.embedding.weight", "geneformer": "bert.embeddings.word_embeddings.weight"}


def check_weights(model: str, weights: str) -> None:
    """Raise :class:`ExtractionError` for an unknown ``weights`` value or a model without a
    re-initialization rule."""
    if weights not in WEIGHTS:
        raise ExtractionError(f"stages.extract.weights: expected one of {list(WEIGHTS)}, got {weights!r}")
    if weights != "trained" and str(model).lower().replace("-", "_") not in EMBEDDING_TABLE:
        raise ExtractionError(f"stages.extract.weights: {weights!r} is available for {sorted(EMBEDDING_TABLE)} only; "
                              f"{model} extracts the trained weights (weights: trained)")


def reinitialize(module, model: str, weights: str, seed: int = 0) -> Dict[str, Any]:
    """Re-initialize ``module`` (a ``torch.nn.Module``) in place; returns a record of what changed.

    ``model`` is ``scgpt`` or ``geneformer``. The record holds ``weights``, ``seed``, the number of
    parameter tensors, how many of them changed and the names of those that did not.
    """
    check_weights(model, weights)
    if weights == "trained":
        return {"weights": "trained"}
    import torch

    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    torch.manual_seed(int(seed))
    if model == "scgpt":
        for m in module.modules():
            if m is module:
                continue
            f = getattr(m, "reset_parameters", None) or getattr(m, "_reset_parameters", None)
            if callable(f):
                f()
        if hasattr(module, "init_weights"):
            module.init_weights()
    else:
        module.apply(module._init_weights)
    params = dict(module.named_parameters())
    table = EMBEDDING_TABLE[model]
    if table not in params:
        raise ExtractionError(f"{model}: the model has no parameter {table!r} (the gene-embedding table)")
    if weights == "random_table":
        with torch.no_grad():
            params[table].copy_(before[table])
    changed = {n: bool(not torch.equal(before[n], p.detach())) for n, p in params.items()}
    return {"weights": weights, "seed": int(seed), "embedding_table": table, "n_params": len(changed),
            "n_changed": int(sum(changed.values())), "unchanged": [n for n, c in changed.items() if not c]}
