"""Randomly re-initialized weights for the extraction (``stages.extract.weights``) on small modules
and the toy fixture, and gzip-compressed cell lists given relative to the configuration."""
from __future__ import annotations

import gzip
import json

import pytest
import torch
import yaml

from interpretfm.config import load_config
from interpretfm.extract import read_cell_list, spec_from_config
from interpretfm.models import get_extractor
from interpretfm.models.base import ExtractionError, ExtractionSpec
from interpretfm.models.reinit import WEIGHTS, check_weights, reinitialize
from interpretfm.runner import run
from interpretfm.testing import write_toy_run


# --- re-initialized weights ---------------------------------------------------------------------


class TinyScGPT(torch.nn.Module):
    """Names and initializers like the vendored scGPT model: a gene-embedding table, linear layers,
    attention (``_reset_parameters``), a layer norm and the model's own ``init_weights``."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.embedding = torch.nn.Embedding(30, 8, padding_idx=0)
        self.encoder.enc_norm = torch.nn.LayerNorm(8)
        self.value = torch.nn.Linear(1, 8)
        self.attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)

    def init_weights(self):
        self.encoder.embedding.weight.data.uniform_(-0.1, 0.1)


def _perturbed(module):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.5)
    return module


def test_reinitialize_forms_on_a_small_module():
    m = _perturbed(TinyScGPT())
    trained = {n: p.detach().clone() for n, p in m.named_parameters()}
    assert reinitialize(m, "scgpt", "trained") == {"weights": "trained"}
    assert all(torch.equal(trained[n], p) for n, p in m.named_parameters())

    rec = reinitialize(m, "scgpt", "random_table", seed=0)
    assert rec["n_params"] == len(trained) and rec["n_changed"] == len(trained) - 1
    assert rec["unchanged"] == ["encoder.embedding.weight"] and rec["embedding_table"] == "encoder.embedding.weight"
    table_kept = {n: p.detach().clone() for n, p in m.named_parameters()}
    assert torch.equal(table_kept["encoder.embedding.weight"], trained["encoder.embedding.weight"])
    assert torch.equal(table_kept["encoder.enc_norm.weight"], torch.ones(8))  # layer-norm reset

    m2 = _perturbed(TinyScGPT())
    rec2 = reinitialize(m2, "scgpt", "random_all", seed=0)
    assert rec2["n_changed"] == rec2["n_params"] and rec2["unchanged"] == []
    p2 = dict(m2.named_parameters())
    # the same seed draws the same weights; the table comes from init_weights (uniform in +-0.1)
    for n, p in table_kept.items():
        if n != "encoder.embedding.weight":
            assert torch.equal(p, p2[n]), n
    assert float(p2["encoder.embedding.weight"].abs().max()) <= 0.1
    m3 = _perturbed(TinyScGPT())
    reinitialize(m3, "scgpt", "random_all", seed=1)
    assert not torch.equal(dict(m3.named_parameters())["value.weight"], p2["value.weight"])


def test_reinitialize_the_vendored_scgpt_model():
    pytest.importorskip("transformers")
    from interp_pipeline.vendor.scgpt_local.model import scGPTConfig, scGPTModelClean

    cfg = scGPTConfig(vocab_size=40, embsize=16, d_hid=16, nlayers=2, nheads=2, pad_token_id=39)
    m = _perturbed(scGPTModelClean(cfg))
    rec = reinitialize(m, "scgpt", "random_table", seed=0)
    assert rec["unchanged"] == ["encoder.embedding.weight"] and rec["n_changed"] == rec["n_params"] - 1
    m2 = _perturbed(scGPTModelClean(cfg))
    rec2 = reinitialize(m2, "scgpt", "random_all", seed=0)
    assert rec2["n_changed"] == rec2["n_params"]


def test_reinitialize_a_bert_model():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=64)
    m = _perturbed(transformers.BertForMaskedLM(cfg))
    table = "bert.embeddings.word_embeddings.weight"
    before = dict((n, p.detach().clone()) for n, p in m.named_parameters())
    rec = reinitialize(m, "geneformer", "random_table", seed=0)
    assert rec["unchanged"] == [table] and torch.equal(dict(m.named_parameters())[table], before[table])
    m2 = _perturbed(transformers.BertForMaskedLM(cfg))
    rec2 = reinitialize(m2, "geneformer", "random_all", seed=0)
    assert rec2["n_changed"] == rec2["n_params"]
    same = [n for n, p in m2.named_parameters() if n != table and torch.equal(p, dict(m.named_parameters())[n])]
    assert len(same) == rec2["n_params"] - 1  # every other tensor equals the random_table draw of the same seed


def test_weights_option_is_checked(tmp_path):
    assert WEIGHTS == ("trained", "random_table", "random_all")
    check_weights("scgpt", "random_all")
    check_weights("c2s_scale", "trained")
    with pytest.raises(ExtractionError, match="available for"):
        check_weights("c2s_scale", "random_table")
    with pytest.raises(ExtractionError, match="expected one of"):
        check_weights("scgpt", "random")
    spec = ExtractionSpec(model="c2s_scale", layers=["layer_13"], weights="random_all")
    with pytest.raises(ExtractionError, match="random_all"):
        get_extractor("toy").check(ExtractionSpec(model="toy", layers=["layer_1.norm2"], weights="random_all"), None)
    assert spec.to_dict()["weights"] == "random_all"

    path = write_toy_run(tmp_path, store=False)
    d = yaml.safe_load(open(path))
    d["stages"]["extract"] = {"weights": "random_table", "weights_seed": 3}
    d["model"]["name"] = "scgpt"
    yaml.safe_dump(d, open(path, "w"))
    s = spec_from_config(load_config(path))
    assert (s.weights, s.weights_seed) == ("random_table", 3)
    d["model"]["name"] = "c2s_scale"
    yaml.safe_dump(d, open(path, "w"))
    with pytest.raises(ExtractionError, match="c2s_scale"):
        spec_from_config(load_config(path))
    d["model"]["name"] = "scgpt"
    d["stages"]["extract"]["weights"] = "untrained"
    yaml.safe_dump(d, open(path, "w"))
    with pytest.raises(ValueError, match="stages.extract.weights"):
        spec_from_config(load_config(path))


def test_cached_store_records_the_weights_and_rejects_unknown_values(tmp_path):
    path = write_toy_run(tmp_path, store=True)
    d = yaml.safe_load(open(path))
    d["stages"]["extract"]["weights"] = "random_all"
    d["model"]["name"] = "scgpt"
    yaml.safe_dump(d, open(path, "w"))
    cfg = load_config(path)
    assert [r["status"] for r in run(cfg, stages=["ingest", "extract"])] == ["done", "done"]
    assert json.load(open(cfg.run.out_dir / "02_extract" / "store.json"))["weights"] == "random_all"
    d["model"]["name"] = "c2s_scale"
    d["run"]["out_dir"] = str(tmp_path / "runs" / "c2s")
    yaml.safe_dump(d, open(path, "w"))
    with pytest.raises(Exception, match="available for"):
        run(load_config(path), stages=["ingest", "extract"])


def test_cell_lists_gzip_and_relative_to_the_configuration(tmp_path):
    ids = ["cell_3", "cell_1", "cell_7"]
    with gzip.open(tmp_path / "cells.txt.gz", "wt") as fh:
        fh.write("# a comment\n" + "\n".join(ids) + "\n")
    assert read_cell_list(tmp_path / "cells.txt.gz") == ids
    with gzip.open(tmp_path / "cells.csv.gz", "wt") as fh:
        fh.write("cell,x\n" + "\n".join(f"{c},1" for c in ids) + "\n")
    assert read_cell_list(tmp_path / "cells.csv.gz") == ids
    path = write_toy_run(tmp_path, store=False)
    d = yaml.safe_load(open(path))
    d["stages"]["extract"] = {"cell_subset": "cells.txt.gz"}
    yaml.safe_dump(d, open(path, "w"))
    cfg = load_config(path)
    assert cfg.stages["extract"]["cell_subset"] == str((tmp_path / "cells.txt.gz").resolve())
