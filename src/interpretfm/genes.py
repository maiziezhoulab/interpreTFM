"""Gene keys and token-to-gene resolution, driven by the configuration.

Every gene-level table of the pipeline is keyed by the dataset's gene identifier
(``dataset.gene_id``: the ``var`` column, or the index, and whether it holds symbols or
Ensembl ids). :class:`GeneTable` collects, per gene, that *key* together with the other
names the ``var`` table knows (symbol and Ensembl columns named in the stage parameters,
plus the index), so that a token id, a g:Profiler result or an offline concept table can be
mapped back to the key whichever name it uses.

Token ids stored by a model's extraction are resolved by :class:`TokenResolver`: a token
is a gene name directly, or a vocabulary id translated through a ``token_vocab`` file
(JSON ``{name: id}`` or ``{id: name}``); special tokens (``model.special_tokens`` and the
usual ``<cls>``/``<pad>``/``<eos>`` markers) resolve to nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

SPECIAL_MARKERS = {"<pad>", "<PAD>", "PAD", "<cls>", "<CLS>", "CLS", "<eos>", "<EOS>", "EOS", "<unk>", "<mask>"}


def _guess_column(var, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in var.columns:
            return c
    return None


@dataclass
class GeneTable:
    """Per-gene names of a panel: the canonical ``key`` plus symbol and Ensembl id when known."""

    keys: List[str]
    kind: str  #: ``symbol`` or ``ensembl``: what ``keys`` are
    symbols: List[Optional[str]] = field(default_factory=list)
    ensembl: List[Optional[str]] = field(default_factory=list)
    alias_to_key: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_adata(cls, adata, gene_id, symbol_column: Optional[str] = None,
                   ensembl_column: Optional[str] = None) -> "GeneTable":
        """Build from ``adata.var`` and the ``dataset.gene_id`` section (``.column``, ``.kind``).

        ``symbol_column`` / ``ensembl_column`` name extra ``var`` columns; when not given,
        common names (``gene_symbol``, ``gene_name``, ``feature_name``; ``ensembl_id``,
        ``gene_ids``, ``gene_id``) are tried.
        """
        var = adata.var
        col = gene_id.column
        if col == "index":
            keys = [str(x) for x in var.index]
        else:
            if col not in var.columns:
                raise KeyError(f"dataset.gene_id.column {col!r} not in adata.var")
            keys = [str(x) for x in var[col]]
        kind = gene_id.kind
        sym_col = symbol_column or _guess_column(var, ("gene_symbol", "gene_name", "feature_name", "symbol"))
        ens_col = ensembl_column or _guess_column(var, ("ensembl_id", "ensembl", "gene_ids", "gene_id"))
        if symbol_column and symbol_column not in var.columns and symbol_column != "index":
            raise KeyError(f"gene_keys.symbol_column {symbol_column!r} not in adata.var")
        if ensembl_column and ensembl_column not in var.columns and ensembl_column != "index":
            raise KeyError(f"gene_keys.ensembl_column {ensembl_column!r} not in adata.var")

        def column(name: Optional[str]) -> List[Optional[str]]:
            if name is None:
                return [None] * len(keys)
            vals = var.index if name == "index" else var[name]
            return [None if (v is None or str(v) in ("", "nan", "None")) else str(v) for v in vals]

        symbols = column(sym_col) if kind != "symbol" else keys[:]
        ensembl = column(ens_col) if kind != "ensembl" else keys[:]
        if kind == "symbol" and sym_col and sym_col != col:
            symbols = [s or k for s, k in zip(column(sym_col), keys)]
        if kind == "ensembl" and ens_col and ens_col != col:
            ensembl = [e or k for e, k in zip(column(ens_col), keys)]
        alias: Dict[str, str] = {}
        index_names = [str(x) for x in var.index]
        for i, k in enumerate(keys):
            for a in (k, symbols[i], ensembl[i], index_names[i]):
                if a and a not in alias:
                    alias[a] = k
        return cls(keys=keys, kind=kind, symbols=symbols, ensembl=ensembl, alias_to_key=alias)

    def names_for_query(self, kind: Optional[str] = None) -> List[str]:
        """Gene names to send to an annotation service: symbols or Ensembl ids (default: the keys)."""
        kind = kind or self.kind
        if kind == self.kind:
            return list(self.keys)
        names = self.symbols if kind == "symbol" else self.ensembl
        return [n or k for n, k in zip(names, self.keys)]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({"key": self.keys, "symbol": self.symbols, "ensembl": self.ensembl})

    def resolve(self, names: Sequence[Any]) -> List[Optional[str]]:
        """Canonical key for each name (``None`` when unknown)."""
        return [self.alias_to_key.get(str(n).strip()) for n in names]


def load_token_vocab(path: Union[str, Path]) -> Dict[str, str]:
    """``{token id: gene name}`` from a JSON vocabulary (``{name: id}`` or ``{id: name}``)."""
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: token vocabulary must be a JSON object")
    sample = next(iter(raw.values()), None)
    if isinstance(sample, int) or (isinstance(sample, str) and sample.isdigit()):
        return {str(v): str(k) for k, v in raw.items()}
    return {str(k): str(v) for k, v in raw.items()}


class TokenResolver:
    """Map stored ``token_ids`` to canonical gene keys (``None`` for special or unknown tokens)."""

    def __init__(self, genes: GeneTable, vocab: Optional[Dict[str, str]] = None,
                 special_tokens: Sequence[Any] = ()):
        self.genes = genes
        self.vocab = vocab or {}
        self.special = {str(t) for t in special_tokens} | SPECIAL_MARKERS
        self._cache: Dict[str, Optional[str]] = {}

    def resolve_one(self, token: Any) -> Optional[str]:
        s = str(token).strip()
        if s in self._cache:
            return self._cache[s]
        key: Optional[str] = None
        if s and s not in self.special:
            name = self.vocab.get(s, s)
            if name not in self.special:
                key = self.genes.alias_to_key.get(name)
                if key is None and name != s:
                    key = self.genes.alias_to_key.get(s)
        self._cache[s] = key
        return key

    def resolve(self, tokens: Sequence[Any]) -> np.ndarray:
        """Object array of keys (``None`` where unresolved)."""
        return np.array([self.resolve_one(t) for t in tokens], dtype=object)

    def resolve_rows(self, tokens: Sequence[Any], key_to_row: Dict[str, int]) -> np.ndarray:
        """Row index of each token's gene in a table keyed by gene (-1 when unresolved)."""
        return np.array([key_to_row.get(k, -1) if k is not None else -1 for k in self.resolve(tokens)], dtype=np.int64)


def resolver_for(cfg, params: Dict[str, Any], store_root: Union[str, Path], genes: GeneTable,
                 stage: str = "concept_eval") -> TokenResolver:
    """The :class:`TokenResolver` of a run: ``stages.<stage>.token_vocab`` (a JSON vocabulary)
    when set, else ``<store_root>/token_vocab.json`` when the extraction wrote one, else
    token ids are taken as gene names. Special tokens come from ``model.special_tokens``."""
    from interpretfm.params import param_path

    vocab: Optional[Dict[str, str]] = None
    given = params.get("token_vocab")
    if given:
        vocab = load_token_vocab(param_path(cfg, given, stage, "token_vocab"))
    elif (Path(store_root) / "token_vocab.json").exists():
        vocab = load_token_vocab(Path(store_root) / "token_vocab.json")
    return TokenResolver(genes, vocab=vocab, special_tokens=cfg.model.special_tokens)
