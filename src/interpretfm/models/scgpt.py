"""scGPT adapter: the vendored whole-human model, gene symbols, ``<cls>`` + expressed genes.

Tokenization is the manuscript's: raw counts, zero genes dropped, ``log1p``, a ``<cls>``
token in front, at most ``max_length`` tokens per cell (panel order, truncated), no sampling.
The stored rows of a layer are the outputs of ``transformer_encoder.layers[i]`` (the post-norm
``norm2`` output, hence the layer names ``layer_<i>.norm2``) for the real tokens of every cell;
padding positions are not stored. The model runs in evaluation mode under ``torch.no_grad``
in float32; cells are processed in dataset order in batches of ``batch_size``. With
``stages.extract.weights`` other than ``trained`` the loaded model is re-initialized on the
extraction device before the forward passes (:func:`interpretfm.models.reinit.reinitialize`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from interpretfm.genes import GeneTable
from interpretfm.models.base import ExtractionError, ExtractionSpec, ModelExtractor, gene_coverage, layer_index


class ScGPTExtractor(ModelExtractor):
    name = "scgpt"
    gene_kind = "symbol"

    def check(self, spec: ExtractionSpec, genes: GeneTable) -> Dict[str, Any]:
        info = super().check(spec, genes)
        ckpt = Path(spec.checkpoint)
        for f in ("best_model.pt", "vocab.json"):
            if not (ckpt / f).is_file():
                raise ExtractionError(f"model.checkpoint: {ckpt} must contain {f} (the scGPT whole-human checkpoint folder)")
        with open(ckpt / "vocab.json") as fh:
            vocab = json.load(fh)
        names = self.gene_names(genes)
        cov = gene_coverage(names, vocab)
        info.update({"vocab": str(ckpt / "vocab.json"), "vocab_size": len(vocab), "coverage": cov,
                     "pad_token": str(vocab.get("<pad>")), "cls_token": str(vocab.get("<cls>"))})
        if cov["n_in_vocab"] == 0:
            raise ExtractionError("no panel gene is in the scGPT vocabulary; check dataset.gene_id (scGPT needs gene "
                                  "symbols: set gene_id.kind: symbol or give stages.extract.gene_keys.symbol_column)")
        for layer in spec.layers:
            layer_index(layer, model="scgpt")
        return info

    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path, log) -> Dict[str, Any]:
        import torch

        from interp_pipeline.adapters.model_base import ModelSpec
        from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
        from interp_pipeline.extraction.extractor import extract_activations
        from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
        from interp_pipeline.types.dataset import StandardDataset
        from interpretfm.models.reinit import reinitialize

        info = self.check(spec, genes)
        names = self.gene_names(genes)
        with open(Path(spec.checkpoint) / "vocab.json") as fh:
            vocab = json.load(fh)
        keep = np.array([n is not None and n in vocab for n in names], dtype=bool)
        if not keep.all():
            log.warning("scgpt: %d of %d panel genes are not in the vocabulary and are left out of the token sequences",
                        int((~keep).sum()), len(names))
        sub = adata[:, keep].copy() if not keep.all() else adata
        gene_names = np.array([n for n, k in zip(names, keep) if k], dtype=object)

        counts = np.asarray(sub.X.sum(axis=1)).ravel()
        nonempty = counts > 0
        if not nonempty.all():
            log.warning("scgpt: %d cells have no counts over the vocabulary genes and get no rows", int((~nonempty).sum()))
            sub = sub[nonempty].copy()
        t0 = time.time()
        adapter = ScGPTAdapter(gene_names=gene_names)
        handle = adapter.load(ModelSpec(name="scgpt", checkpoint=str(spec.checkpoint), device=spec.device, options={}))
        module = ScGPTAdapter.module(handle)
        weights = reinitialize(module, "scgpt", spec.weights, spec.weights_seed)
        module.eval()
        if spec.weights != "trained":
            log.info("scgpt: weights %s (seed %d): %d of %d parameter tensors re-initialized", spec.weights,
                     spec.weights_seed, weights["n_changed"], weights["n_params"])
        n_layers = int(handle.model.config.nlayers)
        for layer in spec.layers:
            layer_index(layer, n_layers, model="scgpt")
        log.info("scgpt: model loaded from %s on %s (%d layers, %.1f s)", spec.checkpoint, handle.device, n_layers,
                 time.time() - t0)
        store = ActivationStore(ActivationStoreSpec(root=str(store_root)))
        extraction_cfg = {"batch_size": spec.batch_size, "max_length": spec.max_length, "model_name": "scgpt",
                          "max_shards": None, "target_tokens_per_shard": spec.rows_per_shard, "save_dtype": spec.dtype}
        t1 = time.time()
        with torch.no_grad():
            _index, layers_done = extract_activations(StandardDataset(adata=sub, obs_key_map={}), handle, adapter,
                                                      list(spec.layers), store, extraction_cfg)
        info.update({"layers_written": layers_done, "n_cells_extracted": int(sub.n_obs), "weights": weights,
                     "n_cells_without_counts": int((~nonempty).sum()), "n_genes_used": int(keep.sum()),
                     "tokenization": {"normalize": "log1p of raw counts", "include_zero_genes": False, "add_cls": True,
                                      "max_length": spec.max_length, "padding_rows_stored": False},
                     "model_dtype": "float32", "seconds_load": round(t1 - t0, 1), "seconds_forward": round(time.time() - t1, 1)})
        return info
