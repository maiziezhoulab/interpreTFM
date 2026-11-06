"""The C2S-Scale adapter without the model stack (CPU): cell sentences against the vendored
cell2sentence rule, shard positions and the subset invariance of the tie-breaking seeds, the
prompt template, and the specification checks."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import scipy.sparse as sp

from interpretfm.extract import spec_from_config
from interpretfm.genes import GeneTable
from interpretfm.models import get_extractor
from interpretfm.models.base import ExtractionError, ExtractionSpec
from interpretfm.models.c2s_scale import (C2SScaleExtractor, cell_sentences, prompt_template, rank_genes, render_prompt,
                                          shard_positions)
from interpretfm.testing import make_toy_adata

from test_extract import toy_extraction_config

LOG = logging.getLogger("test_c2s")


def counts_with_ties(n_cells: int = 40, n_genes: int = 30, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.8, size=(n_cells, n_genes)).astype(np.float32)  # many 1s and 2s: ties everywhere
    X[0] = 0  # a cell without detected genes
    return sp.csr_matrix(X), [f"g{i}" for i in range(n_genes)]


def test_cell_sentences_follow_the_vendored_rule():
    import anndata as ad
    import pandas as pd

    from interp_pipeline.vendor.c2s_local.util import generate_sentences, generate_vocabulary

    X, names = counts_with_ties()
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=names))
    ref = generate_sentences(adata, generate_vocabulary(adata), random_state=42)  # seed 42 + row index
    ours = cell_sentences(X, [n.upper() for n in names], seeds=[42 + i for i in range(X.shape[0])])
    assert [" ".join(s) for s in ours] == ref
    assert ours[0] == [] and all(len(s) == int((X[i] > 0).sum()) for i, s in enumerate(ours))
    # descending counts, truncation, and a different seed changes only the order inside ties
    row = X[5].toarray().ravel()
    vals = [row[int(n[1:])] for n in ours[5]]
    assert vals == sorted(vals, reverse=True)
    assert cell_sentences(X, names, seeds=[0] * X.shape[0], max_genes=3)[5] == [n.lower() for n in cell_sentences(X, [n.upper() for n in names], [0] * X.shape[0])[5][:3]]
    other = rank_genes(X[5].indices, X[5].data, seed=7)
    assert set(other.tolist()) == set(X[5].indices.tolist()) and sorted(row[other], reverse=True) == row[other].tolist()


def test_shard_positions_and_subset_invariance():
    ids, pos, labels = shard_positions(["shard_3", "shard_0", "shard_3", "shard_3", "shard_0"])
    assert ids.tolist() == [3, 0, 3, 3, 0] and pos.tolist() == [0, 0, 1, 2, 1] and labels == {3: "shard_3", 0: "shard_0"}
    ids, pos, labels = shard_positions(["b", "a", "b"])
    assert ids.tolist() == [0, 1, 0] and pos.tolist() == [0, 0, 1] and labels == {0: "b", 1: "a"}
    adata = make_toy_adata(n_cells=60, n_genes=20)
    rng = np.random.default_rng(0)
    adata.obs["shards"] = [f"shard_{k}" for k in rng.integers(0, 4, size=adata.n_obs)]
    spec = ExtractionSpec(model="c2s_scale", layers=["layer_1"], rows_per_shard=100,
                          options={"shard_column": "shards", "tie_seed": 42})
    ex = C2SScaleExtractor()
    ex.prepare(adata, spec, LOG)
    full = dict(ex._positions)
    sub = adata[rng.choice(adata.n_obs, 15, replace=False)].copy()
    assert all(full[str(c)] == ex._positions[str(c)] for c in sub.obs_names)
    k, p = zip(*(full[str(c)] for c in adata.obs_names))
    assert sorted(set(k)) == [0, 1, 2, 3] and max(p) < adata.n_obs and len({(a, b) for a, b in zip(k, p)}) == adata.n_obs
    # consecutive blocks when no column is given: cells_per_shard from rows_per_shard and the median detected genes
    ex2 = C2SScaleExtractor()
    ex2.prepare(adata, ExtractionSpec(model="c2s_scale", layers=["layer_1"], rows_per_shard=100, options={}), LOG)
    k2, p2 = zip(*(ex2._positions[str(c)] for c in adata.obs_names))
    assert list(k2) == sorted(k2) and p2[0] == 0 and max(p2) < 100
    ex3 = C2SScaleExtractor()
    ex3.prepare(adata, ExtractionSpec(model="c2s_scale", layers=["layer_1"], options={"cells_per_shard": 7}), LOG)
    assert [ex3._positions[str(c)] for c in adata.obs_names[:9]] == [(0, i) for i in range(7)] + [(1, 0), (1, 1)]
    with pytest.raises(ExtractionError, match="shard_column"):
        C2SScaleExtractor().prepare(adata, ExtractionSpec(model="c2s_scale", layers=["layer_1"], options={"shard_column": "nope"}), LOG)


def test_prompt_template_and_rendering():
    t = prompt_template()
    assert "{num_genes}" in t and "{organism}" in t and "Cell sentence: {cell_sentence}." in t
    p = render_prompt(t, ["CD74", "HLA-B", "SPP1"], 256, "Homo sapiens")
    assert p.startswith("The following is a list of 256 gene names") and "in a Homo sapiens cell" in p
    assert "\nCell sentence: CD74 HLA-B SPP1.\nThe cell type corresponding to these genes is:" in p


def fake_checkpoint(root, n_layers=4, hidden=8):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"],
                                                   "num_hidden_layers": n_layers, "hidden_size": hidden, "torch_dtype": "bfloat16"}))
    return root


def test_check_validates_checkpoint_layers_and_options(tmp_path):
    adata = make_toy_adata(n_cells=10, n_genes=12)
    from interpretfm.config import GeneId

    genes = GeneTable.from_adata(adata, GeneId(column="index", kind="symbol"))
    ex = get_extractor("c2s_scale")
    assert isinstance(ex, C2SScaleExtractor) and ex.gene_kind == "symbol"
    with pytest.raises(ExtractionError, match="model.checkpoint is required"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"]), genes)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExtractionError, match="config.json"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"], checkpoint=empty), genes)
    ckpt = fake_checkpoint(tmp_path / "ckpt")
    info = ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1", "layer_3"], checkpoint=ckpt), genes)
    assert info["n_layers"] == 4 and info["hidden_size"] == 8 and info["checkpoint_dtype"] == "bfloat16"
    assert info["coverage"] == {"n_genes": 12, "n_with_symbol": 12, "n_missing": 0, "fraction": 1.0}
    assert info["options"] == {"max_genes": 256, "token_pooling": "last", "model_dtype": "float32", "tie_seed": 42,
                               "shard_column": None, "cells_per_shard": None, "organism": None}
    with pytest.raises(ExtractionError, match="layers 0..3"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_4"], checkpoint=ckpt), genes)
    with pytest.raises(ExtractionError, match="layer_<i>"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1.norm2"], checkpoint=ckpt), genes)
    with pytest.raises(ExtractionError, match="token_pooling"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"], checkpoint=ckpt, options={"token_pooling": "first"}), genes)
    with pytest.raises(ExtractionError, match="model_dtype"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"], checkpoint=ckpt, options={"model_dtype": "int8"}), genes)
    with pytest.raises(ExtractionError, match="max_genes"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"], checkpoint=ckpt, options={"max_genes": 0}), genes)
    no_symbols = GeneTable(keys=["ENSG1", "ENSG2"], kind="ensembl", symbols=[None, None], ensembl=["ENSG1", "ENSG2"])
    with pytest.raises(ExtractionError, match="no panel gene has a symbol"):
        ex.check(ExtractionSpec(model="c2s_scale", layers=["layer_1"], checkpoint=ckpt), no_symbols)


def test_spec_from_config_passes_the_c2s_options(tmp_path):
    cfg = toy_extraction_config(tmp_path, model="c2s_scale", batch_size=4, dtype="float16",
                                c2s_scale={"max_genes": 128, "shard_column": "shards"})
    spec = spec_from_config(cfg)
    assert spec.model == "c2s_scale" and spec.batch_size == 4 and spec.dtype == "float16"
    assert spec.options == {"max_genes": 128, "shard_column": "shards"}
    assert C2SScaleExtractor._options(spec)["max_genes"] == 128 and C2SScaleExtractor._options(spec)["token_pooling"] == "last"
