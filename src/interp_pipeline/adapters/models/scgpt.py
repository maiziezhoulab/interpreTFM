from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from interp_pipeline.adapters.model_base import ModelAdapter, ModelSpec
from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

# Vendored scGPT model and tokenizer (see interp_pipeline/vendor/scgpt_local)
from interp_pipeline.vendor.scgpt_local.load_model_from_pretrain import (
    create_clean_model_from_pretrain,
)


@dataclass
class ScGPTHandle:
    model: torch.nn.Module
    tokenizer: Any  # interp_pipeline.vendor.scgpt_local.tokenizer.Tokenizer
    device: torch.device


def layer_index(layer_name: str) -> int:
    """Block index of ``layer_<i>`` or ``layer_<i>.norm2`` (the stored name of the manuscript's layers)."""
    head = layer_name.split(".")[0]
    if not head.startswith("layer_") or not head[len("layer_"):].isdigit():
        raise ValueError(f"Unsupported scGPT layer name {layer_name!r}; expected layer_<i> or layer_<i>.norm2")
    return int(head[len("layer_"):])


class ScGPTAdapter(ModelAdapter):
    """
    Adapter around the vendored scGPT implementation.

    Key points:
    - model(src, values, src_key_padding_mask) returns token-level hidden states [B,T,H]
    - activations are the outputs of transformer_encoder.layers[i] (the post-norm ``norm2``
      output), captured by running the encoder blocks one by one with the padding mask, as the
      extraction used for the manuscript did
    - only the real (non-padding) tokens of every cell are kept; ``gene_names`` (symbols in
      the model vocabulary, one per column of ``adata``) can be given explicitly instead of
      being read from ``var["feature_name"]`` / ``var["index"]``
    """

    def __init__(self, gene_names: Optional[Sequence[str]] = None):
        self.gene_names = None if gene_names is None else np.asarray(list(gene_names), dtype=object)

    def load(self, spec: ModelSpec) -> ScGPTHandle:
        device = torch.device(spec.device)
        model, tokenizer = create_clean_model_from_pretrain(spec.checkpoint, device=device)
        return ScGPTHandle(model=model, tokenizer=tokenizer, device=device)

    @staticmethod
    def module(model_handle: ScGPTHandle) -> torch.nn.Module:
        """The plain ``torch.nn.Module`` (the loader wraps the model for tracing)."""
        m = model_handle.model
        return getattr(m, "_model", m)

    def list_layers(self, model_handle: ScGPTHandle) -> List[str]:
        n = int(self.module(model_handle).config.nlayers)
        return [f"layer_{i}.norm2" for i in range(n)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        # every scGPT row is stored as a gene token; the <cls> row is identified by its token id
        return "gene"

    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: ScGPTHandle,
        batch_size: int,
        max_length: int,
    ) -> Iterable[Dict[str, Any]]:
        """
        Batches are built by running the tokenizer on chunks of adata.X (raw counts).

        Output keys:
          - cell_ids: list[str] length B
          - genes: Tensor[B,T]
          - expressions: Tensor[B,T]
          - src_key_padding_mask: Tensor[B,T] bool
        """
        adata = dataset.adata
        if self.gene_names is not None:
            gene_names = self.gene_names
            if len(gene_names) != adata.n_vars:
                raise ValueError(f"gene_names has {len(gene_names)} entries for {adata.n_vars} genes")
        else:
            gene_key = "feature_name"
            if gene_key not in adata.var.columns:
                if "index" in adata.var.columns:
                    gene_key = "index"
                else:
                    raise ValueError("AnnData .var must include a gene symbol column like 'feature_name' or 'index'")
            gene_names = adata.var[gene_key].to_numpy()

        X = adata.X
        n = adata.n_obs
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cell_ids = [str(x) for x in adata.obs_names[start:end]]
            chunk = X[start:end]
            if not isinstance(chunk, np.ndarray):
                chunk = chunk.toarray()
            chunk = np.asarray(chunk, dtype=np.float32)

            # Tokenize: returns (genes, expressions, src_key_padding_mask)
            genes, expressions, src_key_padding_mask = model_handle.tokenizer(
                chunk,
                gene_names,
                max_length=max_length,
                include_zero_genes=False,
                normalize=True,
                add_cls=True,
            )

            yield {
                "cell_ids": cell_ids,
                "genes": genes,
                "expressions": expressions,
                "src_key_padding_mask": src_key_padding_mask,
            }

    def forward_and_capture(
        self,
        model_handle: ScGPTHandle,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the embedding and the encoder blocks one by one and record the output of the
        requested blocks ([B,T,H], padding positions included; dropped in process_captured).
        """
        model = self.module(model_handle)
        device = model_handle.device

        genes = batch["genes"].to(device)                  # [B,T]
        expressions = batch["expressions"].to(device)      # [B,T]
        padmask = batch["src_key_padding_mask"].to(device) # [B,T] bool

        want = {layer_index(lname): lname for lname in layers}
        captured: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            x = model.encoder(genes) + model.value_encoder(expressions)  # [B,T,H]
            for i, layer in enumerate(model.transformer_encoder.layers):
                x = layer(x, src_key_padding_mask=padmask)
                if i in want:
                    captured[want[i]] = x.detach().to("cpu")
                if i >= max(want):
                    break
        return captured

    def process_captured(
        self,
        captured: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Flatten [B, T, H] -> [N, H] over the real tokens of every cell (padding dropped),
        with the token ids and the cell id of every row.
        """
        genes: torch.Tensor = batch["genes"].to("cpu")   # [B, T]
        cell_ids = batch["cell_ids"]
        padmask = batch["src_key_padding_mask"].to("cpu")  # [B, T] bool, True=padding
        keep = ~padmask

        T_list: List[int] = keep.sum(dim=1).tolist()
        tok_list = [str(x) for x in genes[keep].tolist()]

        ex_list: List[str] = []
        for cid, t in zip(cell_ids, T_list):
            ex_list.extend([str(cid)] * t)

        result: Dict[str, Dict[str, Any]] = {}
        for layer_name, acts_btH in captured.items():
            flat = acts_btH[keep]   # [N, H]
            result[layer_name] = {
                "acts": flat,
                "tok": tok_list,
                "ex": ex_list,
                "token_unit": self.infer_token_unit(layer_name),
                "T_list": T_list,
            }
        return result
