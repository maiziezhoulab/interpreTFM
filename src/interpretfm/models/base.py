"""The adapter interface behind the extraction boundary.

An adapter turns a dataset into token-level activations of one model and writes them to the
sharded activation store (:class:`interp_pipeline.io.activation_store.ActivationStore`). It is
imported only inside the ``extract`` stage, which runs in the model's environment, so the
heavy model stacks never load in the core environment.

Every adapter receives the same :class:`ExtractionSpec` (built from the configuration by
:func:`interpretfm.extract.spec_from_config`) and the panel's gene names as a
:class:`interpretfm.genes.GeneTable`, so that the vocabulary mapping (symbols for scGPT,
Ensembl ids for Geneformer) is decided by the ``dataset.gene_id`` section and not by column
names hard-coded in model code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from interpretfm.genes import GeneTable

DTYPES = ("float32", "float16")


@dataclass
class ExtractionSpec:
    """Everything an adapter needs to know, taken from ``model`` and ``stages.extract``."""

    model: str
    layers: List[str]
    checkpoint: Optional[Path] = None
    device: str = "cuda"  #: ``cuda``, ``cuda:N`` or ``cpu``
    batch_size: int = 8  #: cells per forward pass
    max_length: int = 512  #: maximum tokens per cell (models with a fixed context)
    dtype: str = "float32"  #: storage dtype of the activations (``float32`` or ``float16``)
    rows_per_shard: int = 200_000  #: token rows per store shard (approximate)
    pooling_kind: str = "token"  #: ``model.cell_pooling.kind``
    pooling_token: Optional[str] = None  #: ``model.cell_pooling.token``
    special_tokens: List[str] = field(default_factory=list)
    seed: int = 0
    weights: str = "trained"  #: ``trained``, ``random_table`` or ``random_all`` (:mod:`interpretfm.models.reinit`)
    weights_seed: int = 0  #: seed of the re-initialization
    options: Dict[str, Any] = field(default_factory=dict)  #: ``stages.extract.<model>``: model-specific keys

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoint"] = None if self.checkpoint is None else str(self.checkpoint)
        return d


class ExtractionError(RuntimeError):
    """Extraction cannot run: missing weights, no GPU, unknown layer, genes outside the vocabulary."""


class ModelExtractor(ABC):
    """One model's extraction: ``check`` (cheap, before loading anything) and ``extract``."""

    #: the ``model.name`` this adapter serves
    name: str = ""
    #: which gene identifier the model's vocabulary uses (``symbol`` or ``ensembl``)
    gene_kind: str = "symbol"
    #: whether weights (``model.checkpoint``) are required
    needs_checkpoint: bool = True
    #: whether ``spec.device`` matters (a synthetic model ignores it)
    uses_device: bool = True

    def check(self, spec: ExtractionSpec, genes: GeneTable) -> Dict[str, Any]:
        """Validate the specification before any model code loads; returns a summary (gene coverage, warnings)."""
        if self.needs_checkpoint:
            if spec.checkpoint is None:
                raise ExtractionError(f"model.checkpoint is required to extract {self.name} activations")
            if not Path(spec.checkpoint).exists():
                raise ExtractionError(f"model.checkpoint: path not found: {spec.checkpoint}")
        if spec.dtype not in DTYPES:
            raise ExtractionError(f"stages.extract.dtype: expected one of {list(DTYPES)}, got {spec.dtype!r}")
        if spec.batch_size <= 0 or spec.rows_per_shard <= 0:
            raise ExtractionError("stages.extract.batch_size and rows_per_shard must be > 0")
        from interpretfm.models.reinit import check_weights

        check_weights(self.name, spec.weights)
        return {"model": self.name, "n_genes": len(genes.keys)}

    def prepare(self, adata, spec: ExtractionSpec, log) -> None:
        """Called once with the **whole** dataset before any cell subset is applied (no-op by default).

        Adapters whose tokenization depends on dataset-level state record it here, so that a
        cell's tokens do not change with ``stages.extract.cell_subset`` (C2S-Scale: the shard
        and the position of every cell, which seed the tie-breaking of its cell sentence).
        """

    def gene_names(self, genes: GeneTable) -> List[Optional[str]]:
        """The panel's names in the model's vocabulary kind (``None`` where unknown)."""
        return list(genes.symbols if self.gene_kind == "symbol" else genes.ensembl)

    @abstractmethod
    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path,
                log) -> Dict[str, Any]:
        """Write the activations of ``spec.layers`` for every cell of ``adata`` to ``store_root``.

        ``adata.X`` holds raw counts (the stage copies ``dataset.counts_layer`` into ``X``).
        ``work_dir`` is a scratch directory inside the stage directory (tokenized datasets,
        prepared copies). Returns a JSON-serialisable summary that goes into the per-layer
        manifests (e.g. the vocabulary coverage, the model version, timing).
        """


def layer_index(layer: str, n_layers: Optional[int] = None, model: str = "") -> int:
    """The zero-based block index of a stored layer name (``layer_3`` or ``layer_3.norm2``)."""
    head = layer.split(".")[0]
    if not head.startswith("layer_") or not head[len("layer_"):].isdigit():
        raise ExtractionError(f"model.layers: {layer!r} is not a layer name of the form layer_<i>[.norm2]"
                              + (f" ({model})" if model else ""))
    i = int(head[len("layer_"):])
    if n_layers is not None and i >= n_layers:
        raise ExtractionError(f"model.layers: {layer!r} does not exist; {model or 'the model'} has layers 0..{n_layers - 1}")
    return i


def gene_coverage(names: Sequence[Optional[str]], vocab) -> Dict[str, Any]:
    """How many panel names the model vocabulary contains (``vocab``: a set-like of names)."""
    known = [n for n in names if n is not None and n in vocab]
    missing = [n for n in names if n is None or n not in vocab]
    return {"n_genes": len(names), "n_in_vocab": len(known), "n_missing": len(missing),
            "fraction": (len(known) / len(names)) if names else None,
            "missing_examples": [str(m) for m in missing[:20]]}
