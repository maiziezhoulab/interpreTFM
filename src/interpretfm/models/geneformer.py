"""Geneformer adapter: Hugging Face ``BertForMaskedLM`` weights, Ensembl ids, rank-value encoding.

The Geneformer tokenizer (``geneformer.tokenizer.TranscriptomeTokenizer`` of the installed
Geneformer package) ranks every cell's genes by their median-normalized expression; V2 models
add ``<cls>`` and ``<eos>``. Genes are matched to the token dictionary by Ensembl id; for a
dataset keyed by symbols the package's gene-name dictionary supplies the ids, and genes it
does not know are left out (reported in the manifest). A prepared copy of the dataset and the
tokenized Arrow dataset are written under the stage's ``work`` directory. The stored rows of
``layer_<i>`` are the outputs of transformer block ``i`` (``hidden_states[i + 1]``) for every
token of a cell, including ``<cls>`` and ``<eos>``; one store shard holds ``cells_per_shard``
consecutive cells. The model runs in evaluation mode under ``torch.no_grad`` in float32. With
``stages.extract.weights`` other than ``trained`` the loaded model is re-initialized on the
extraction device before the forward passes (:func:`interpretfm.models.reinit.reinitialize`).

Options (``stages.extract.geneformer``): ``model_version`` (``V2``, or ``V1`` for the
30M-cell models and their dictionaries), ``nproc`` (tokenizer processes, 1), ``cells_per_shard``
(default: from ``rows_per_shard`` and the median detected genes).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from interpretfm.genes import GeneTable
from interpretfm.models.base import ExtractionError, ExtractionSpec, ModelExtractor, gene_coverage, layer_index

VERSIONS = ("V1", "V2")


def _dictionary_files(model_version: str) -> Dict[str, Path]:
    import geneformer

    v = model_version.upper()
    if v == "V1":
        return {"token": Path(geneformer.TOKEN_DICTIONARY_FILE_30M), "gene_name": Path(geneformer.ENSEMBL_DICTIONARY_FILE_30M),
                "median": Path(geneformer.GENE_MEDIAN_FILE_30M)}
    return {"token": Path(geneformer.TOKEN_DICTIONARY_FILE), "gene_name": Path(geneformer.ENSEMBL_DICTIONARY_FILE),
            "median": Path(geneformer.GENE_MEDIAN_FILE)}


class GeneformerExtractor(ModelExtractor):
    name = "geneformer"
    gene_kind = "ensembl"

    def _version(self, spec: ExtractionSpec) -> str:
        v = str(spec.options.get("model_version", "V2")).upper()
        if v not in VERSIONS:
            raise ExtractionError(f"stages.extract.geneformer.model_version: expected one of {list(VERSIONS)}, got {v!r}")
        return v

    def ensembl_ids(self, genes: GeneTable, model_version: str) -> List[Optional[str]]:
        """Ensembl ids of the panel: the dataset's own, else the symbol looked up in Geneformer's dictionary."""
        ids = list(genes.ensembl)
        if all(i is not None for i in ids):
            return ids
        with open(_dictionary_files(model_version)["gene_name"], "rb") as fh:
            name_to_id = pickle.load(fh)
        lower = {str(k).upper(): v for k, v in name_to_id.items()}
        out = []
        for ens, sym in zip(ids, genes.symbols):
            if ens is None and sym is not None:
                ens = name_to_id.get(sym) or lower.get(str(sym).upper())
            out.append(None if ens is None else str(ens))
        return out

    def check(self, spec: ExtractionSpec, genes: GeneTable) -> Dict[str, Any]:
        info = super().check(spec, genes)
        version = self._version(spec)
        ckpt = Path(spec.checkpoint)
        if not (ckpt / "config.json").is_file():
            raise ExtractionError(f"model.checkpoint: {ckpt} must be a Hugging Face model folder with config.json "
                                  "(for example the top level of the ctheodoris/Geneformer repository, or one of its "
                                  "model sub-folders)")
        with open(ckpt / "config.json") as fh:
            mcfg = json.load(fh)
        n_layers = int(mcfg.get("num_hidden_layers", 0))
        for layer in spec.layers:
            layer_index(layer, n_layers, model="geneformer")
        files = _dictionary_files(version)
        with open(files["token"], "rb") as fh:
            token_dict = pickle.load(fh)
        if int(mcfg.get("vocab_size", 0)) != len(token_dict):
            raise ExtractionError(f"model.checkpoint: vocab_size {mcfg.get('vocab_size')} does not match the {version} token "
                                  f"dictionary ({len(token_dict)} entries); set stages.extract.geneformer.model_version "
                                  "to the version that matches the weights")
        ids = self.ensembl_ids(genes, version)
        cov = gene_coverage(ids, token_dict)
        if cov["n_in_vocab"] == 0:
            raise ExtractionError("no panel gene is in the Geneformer token dictionary; check dataset.gene_id "
                                  "(Ensembl ids, or symbols the package's gene-name dictionary knows)")
        info.update({"model_version": version, "n_layers": n_layers, "hidden_size": int(mcfg.get("hidden_size", 0)),
                     "token_dictionary": str(files["token"]), "coverage": cov,
                     "n_genes_without_ensembl_id": int(sum(i is None for i in ids))})
        return info

    def extract(self, adata, genes: GeneTable, spec: ExtractionSpec, store_root: Path, work_dir: Path, log) -> Dict[str, Any]:
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        from interp_pipeline.extraction.geneformer_extraction import extract_geneformer_to_store, tokenize_geneformer_dataset
        from interpretfm.models.reinit import reinitialize

        info = self.check(spec, genes)
        version = info["model_version"]
        ids = self.ensembl_ids(genes, version)
        keep = np.array([i is not None for i in ids], dtype=bool)
        if not keep.all():
            log.warning("geneformer: %d of %d panel genes have no Ensembl id and are left out", int((~keep).sum()), len(ids))
        X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
        X = sp.csr_matrix(X)[:, np.flatnonzero(keep)]
        var = pd.DataFrame(index=pd.Index([i for i, k in zip(ids, keep) if k], name="ensembl_id"))
        var["ensembl_id"] = var.index.astype(str)
        if not var.index.is_unique:
            dup = var.index[var.index.duplicated()].unique().tolist()
            log.warning("geneformer: %d duplicated Ensembl ids; the tokenizer collapses them (%s...)", len(dup), dup[:5])
            var.index = pd.Index([f"{i}__{n}" if n else i for n, i in zip(var.groupby(level=0).cumcount(), var.index)])
        n_counts = np.asarray(X.sum(axis=1)).ravel()
        n_detected = np.asarray((X > 0).sum(axis=1)).ravel()
        cells_per_shard = int(spec.options.get("cells_per_shard") or max(1, spec.rows_per_shard // (int(np.median(n_detected)) + 2)))
        obs = pd.DataFrame(index=pd.Index([str(o) for o in adata.obs_names]))
        obs["cell_id"] = obs.index.astype(str)
        obs["n_counts"] = n_counts.astype(np.int64)
        obs["shards"] = [f"shard_{i // cells_per_shard}" for i in range(adata.n_obs)]
        prepared_dir = work_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepared_dir / "cells.h5ad"
        ad.AnnData(X=X.astype(np.float32), obs=obs, var=var).write_h5ad(prepared)
        t0 = time.time()
        tokenized = tokenize_geneformer_dataset(str(prepared), str(work_dir / "tokenized"), "cells", model_version=version,
                                                nproc=int(spec.options.get("nproc", 1)))
        t1 = time.time()
        log.info("geneformer: tokenized %d cells (%s, %.1f s): %s", adata.n_obs, version, t1 - t0, tokenized)
        if spec.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise ExtractionError("stages.extract.device: cuda requested but no GPU is visible to this process")
        weights: Dict[str, Any] = {}

        def after_load(handle) -> None:
            weights.update(reinitialize(handle.model, "geneformer", spec.weights, spec.weights_seed))
            handle.model.eval()
            if spec.weights != "trained":
                log.info("geneformer: weights %s (seed %d): %d of %d parameter tensors re-initialized", spec.weights,
                         spec.weights_seed, weights["n_changed"], weights["n_params"])

        layers_done = extract_geneformer_to_store(model_dir=str(spec.checkpoint), tokenized_dataset_path=tokenized,
                                                  store_root=str(store_root), layers=list(spec.layers), model_version=version,
                                                  device=spec.device, forward_batch_size=spec.batch_size,
                                                  save_dtype="fp16" if spec.dtype == "float16" else "fp32",
                                                  after_load=after_load)
        info.update({"layers_written": list(layers_done), "n_cells_extracted": int(adata.n_obs), "n_genes_used": int(keep.sum()),
                     "weights": weights,
                     "cells_per_shard": cells_per_shard, "tokenized_dataset": str(tokenized), "prepared_h5ad": str(prepared),
                     "tokenization": {"rank_value": "median-normalized expression, descending", "special_tokens": version == "V2",
                                      "padding_rows_stored": False},
                     "model_dtype": "float32", "seconds_tokenize": round(t1 - t0, 1), "seconds_forward": round(time.time() - t1, 1)})
        return info
