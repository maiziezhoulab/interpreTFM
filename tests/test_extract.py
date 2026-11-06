"""The extract stage and the model-environment boundary on toy data (CPU, no weights): the
extraction specification, cell subsets, per-layer manifests, the store comparison used by the
acceptance check, container command construction, the synthetic ``toy`` model in-process and
through a real subprocess, the cached-store mode and the ``interpretfm extract`` verb."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from interpretfm.config import ModelEnv, load_config
from interpretfm.envs import EnvError, child_environment, config_binds, container_command, resolve_env
from interpretfm.extract import compare_projections, compare_stores, layer_manifest, read_cell_list, select_cells, spec_from_config
from interpretfm.manifest import StageManifest
from interpretfm.models import available_models, get_extractor
from interpretfm.models.base import ExtractionError, gene_coverage, layer_index
from interpretfm.runner import StageError, get_stage, model_stage_args, plan, run, stage_dir
from interpretfm.testing import make_toy_adata, make_toy_concepts, toy_stage_params, write_toy_run, write_toy_store

LAYER = "layer_1.norm2"


def toy_extraction_config(root: Path, env=None, model: str = "toy", **extract):
    """A toy run whose extract stage really extracts (no store_root) with the given parameters."""
    path = write_toy_run(root, env=env)
    d = yaml.safe_load(open(path))
    d["model"]["name"] = model
    d["stages"]["extract"] = {"batch_size": 16, "rows_per_shard": 1000, **extract}
    d["stages"].update({k: v for k, v in toy_stage_params().items() if k in ("tis", "sae")})
    yaml.safe_dump(d, open(path, "w"), sort_keys=False)
    return load_config(path)


# --- specification and helpers ----------------------------------------------------------------------


def test_spec_from_config_defaults_and_options(tmp_path):
    cfg = toy_extraction_config(tmp_path, toy={"hidden_dim": 8}, dtype="float16")
    spec = spec_from_config(cfg)
    assert spec.model == "toy" and spec.layers == [LAYER] and spec.checkpoint is None
    assert spec.batch_size == 16 and spec.rows_per_shard == 1000 and spec.max_length == 512 and spec.device == "cuda"
    assert spec.dtype == "float16" and spec.options == {"hidden_dim": 8}
    assert spec.pooling_kind == "token" and spec.pooling_token == "60695" and spec.special_tokens == ["60694", "60695"]
    assert spec.to_dict()["checkpoint"] is None
    cfg2 = toy_extraction_config(tmp_path / "b", device="gpu")
    with pytest.raises(ValueError, match="stages.extract.device"):
        spec_from_config(cfg2)
    cfg3 = toy_extraction_config(tmp_path / "c", dtype="bfloat16")
    with pytest.raises(ValueError, match="stages.extract.dtype"):
        spec_from_config(cfg3)


def test_select_cells(tmp_path):
    names = [f"cell_{i}" for i in range(50)]
    assert select_cells(names, None, 0) == (None, {"mode": "all", "n_cells": 50})
    idx, info = select_cells(names, 10, 0)
    assert len(idx) == 10 and list(idx) == sorted(idx) and info["mode"] == "random"
    assert list(select_cells(names, 10, 0)[0]) == list(idx) and list(select_cells(names, 10, 1)[0]) != list(idx)
    assert select_cells(names, 50, 0)[0] is None and select_cells(names, 500, 0)[1]["mode"] == "all"
    f = tmp_path / "ids.txt"
    f.write_text("# ids\ncell_3\ncell_1\ncell_3\n")
    idx, info = select_cells(names, str(f), 0)
    assert list(idx) == [1, 3] and info == {"mode": "file", "file": str(f), "n_cells": 2, "n_listed": 3}
    c = tmp_path / "ids.csv"
    c.write_text("cell,label\ncell_7,a\ncell_2,b\n")
    assert read_cell_list(c) == ["cell_7", "cell_2"] and list(select_cells(names, str(c), 0)[0]) == [2, 7]
    f.write_text("cell_3\nnot_a_cell\n")
    with pytest.raises(ValueError, match="not in the dataset"):
        select_cells(names, str(f), 0)
    with pytest.raises(FileNotFoundError):
        select_cells(names, str(tmp_path / "missing.txt"), 0)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            select_cells(names, bad, 0)


def test_layer_index_and_coverage():
    assert layer_index("layer_1.norm2") == 1 and layer_index("layer_11", 12) == 11
    with pytest.raises(ExtractionError, match="layer_<i>"):
        layer_index("block_3")
    with pytest.raises(ExtractionError, match="layers 0..11"):
        layer_index("layer_12.norm2", 12, model="scgpt")
    cov = gene_coverage(["A", None, "C", "D"], {"A", "C"})
    assert cov["n_in_vocab"] == 2 and cov["n_missing"] == 2 and cov["fraction"] == 0.5


def test_adapter_registry():
    assert available_models() == ["c2s_scale", "geneformer", "scgpt", "toy"]
    assert get_extractor("toy").name == "toy" and get_extractor("SCGPT").name == "scgpt"
    with pytest.raises(ExtractionError, match="no extraction adapter"):
        get_extractor("llama")
    assert get_extractor("c2s_scale").name == "c2s_scale"


def test_layer_manifest_counts(tmp_path):
    adata = make_toy_adata(n_cells=40)
    root = write_toy_store(tmp_path / "store", adata, rows_per_shard=300)
    m = layer_manifest(root, LAYER, pooling_token="60695", special_tokens=["60694", "60695"])
    assert m["n_cells"] == 40 and m["n_cells_with_pooling_token"] == 40 and m["n_rows_pooling_token"] == 40
    assert m["n_rows"] == m["n_rows_gene"] + m["n_rows_special"] and m["n_rows_special"] == 40
    assert m["hidden_dim"] == 16 and m["dtype"] == "torch.float32" and m["n_shards"] > 1


# --- store comparison -------------------------------------------------------------------------------


def test_compare_stores_and_projections(tmp_path):
    adata = make_toy_adata(n_cells=60)
    concepts = make_toy_concepts(adata)
    a = write_toy_store(tmp_path / "a", adata, concepts=concepts, rows_per_shard=500)
    b = write_toy_store(tmp_path / "b", adata, concepts=concepts, rows_per_shard=333)  # same rows, other sharding
    cells = [str(c) for c in adata.obs_names[:25]]
    r = compare_stores(a, b, LAYER, cells)
    assert r["token_sequence_identical"] and r["n_cells_compared"] == 25 and r["n_rows_identical"] == r["n_rows_compared"]
    assert r["cosine"]["min"] > 0.999999 and r["max_abs_diff"] == 0.0 and r["n_rows_cosine_below_0.999"] == 0
    # a perturbed copy: token order preserved, values differ
    import torch

    from interpretfm.stores import list_shards

    c = tmp_path / "c"
    import shutil

    shutil.copytree(a, c)
    for shard in list_shards(c, LAYER):
        X = torch.load(shard / "activations.pt", weights_only=False)
        torch.save(X * 1.01 + 0.05, shard / "activations.pt")
    r2 = compare_stores(a, c, LAYER, cells)
    assert r2["token_sequence_identical"] and r2["max_abs_diff"] > 0.01 and r2["cosine"]["min"] < 1.0
    # a store missing a cell and with a different token order
    d = write_toy_store(tmp_path / "d", adata[:24].copy(), concepts=concepts)
    r3 = compare_stores(a, d, LAYER, cells)
    assert r3["n_cells_missing_in_ref"] == 1 and not r3["token_sequence_identical"]
    from interpretfm.sae import untrained_sae

    untrained_sae(a, LAYER, 2, 0, tmp_path / "sae.pt")
    p = compare_projections(a, b, LAYER, tmp_path / "sae.pt", cells, special_tokens=["60695"])
    assert p["n_cells"] == 25 and p["n_features"] == 32 and p["store_vs_store"]["cosine"]["min"] > 0.999999
    assert p["store_vs_store"]["frac_entries_identical"] == 1.0


# --- the boundary ---------------------------------------------------------------------------------


def test_cached_store_mode_runs_in_process_without_the_env(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, store=True, env={"python": str(tmp_path / "no" / "python")}))
    assert not get_stage("extract").runs_in_model_env(cfg)
    assert [p["env"] for p in plan(cfg, ["extract"])] == ["core"]
    rows = run(cfg, stages=["ingest", "extract"])
    assert [r["status"] for r in rows] == ["done", "done"]
    assert json.load(open(stage_dir(cfg.run.out_dir, "extract") / "store.json"))["source"] == "cached"


def test_env_current_runs_fresh_extraction_in_process(tmp_path, monkeypatch):
    cfg = toy_extraction_config(tmp_path, env=None, cell_subset=120)
    assert cfg.model.env.kind == "current" and not get_stage("extract").runs_in_model_env(cfg)
    rows = run(cfg, stages=["ingest", "extract", "tis", "sae"])
    assert [r["status"] for r in rows] == ["done"] * 4
    sdir = stage_dir(cfg.run.out_dir, "extract")
    spec = json.load(open(sdir / "store.json"))
    assert spec["source"] == "extracted" and spec["n_cells"] == 120 and spec["cell_pooling"] == {"kind": "token", "token": "60695"}
    assert Path(spec["root"]) == sdir / "store"
    ids = (sdir / "cell_subset.txt").read_text().split()
    assert len(ids) == 120 and ids == sorted(ids, key=lambda s: int(s.split("_")[1]))
    m = json.load(open(sdir / LAYER / "manifest.json"))
    assert m["n_cells"] == 120 and m["n_cells_with_pooling_token"] == 120 and m["hidden_dim"] == 16
    assert m["environment"]["python"] == sys.version.split()[0] and any(p.lower().startswith("numpy==") for p in m["environment"]["packages"])
    assert m["environment"]["torch"]["version"] and m["adapter"]["synthetic"] and m["cell_subset"]["mode"] == "random"
    tis = json.load(open(stage_dir(cfg.run.out_dir, "tis") / "tis.json"))
    assert tis["layers"][LAYER]["n_cells"] == 120
    # the next run reuses the cached stage
    assert [r["status"] for r in run(cfg, stages=["extract"])] == ["cached"]


def test_toy_extraction_through_a_subprocess(tmp_path):
    cfg = toy_extraction_config(tmp_path, env={"python": sys.executable})
    assert get_stage("extract").runs_in_model_env(cfg)
    assert model_stage_args(cfg, get_stage("extract"), cfg.run.out_dir)[:1] == ["extract"]
    rows = run(cfg, stages=["ingest", "extract", "tis"])
    assert [(r["env"], r["status"]) for r in rows] == [("core", "done"), ("model", "done"), ("core", "done")]
    sdir = stage_dir(cfg.run.out_dir, "extract")
    log = (sdir / "log.txt").read_text()
    assert "interpretfm.cli extract --config" in log and "--model toy --layers layer_1.norm2" in log
    m = StageManifest.read(sdir)
    assert m.status == "done" and m.model_env["kind"] == "python" and m.extra["source"] == "extracted"
    # the child wrote exactly the rows the synthetic model defines for every cell
    adata = make_toy_adata()
    ref = write_toy_store(tmp_path / "ref", adata, concepts=make_toy_concepts(adata))
    r = compare_stores(sdir / "store", ref, LAYER, [str(c) for c in adata.obs_names])
    assert r["token_sequence_identical"] and r["n_cells_compared"] == 300 and r["max_abs_diff"] == 0.0


def test_clear_errors_for_missing_weights_and_gpu(tmp_path, monkeypatch):
    cfg = toy_extraction_config(tmp_path, model="scgpt")  # checkpoint: null
    with pytest.raises(StageError, match="model.checkpoint is required"):
        run(cfg, stages=["ingest", "extract"])
    assert "ExtractionError" in StageManifest.read(stage_dir(cfg.run.out_dir, "extract")).error
    import torch

    from interpretfm.models import toy as toy_module

    monkeypatch.setattr(toy_module.ToyExtractor, "uses_device", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg2 = toy_extraction_config(tmp_path / "b", device="cuda")
    with pytest.raises(StageError, match="no GPU is visible"):
        run(cfg2, stages=["ingest", "extract"])
    cfg3 = toy_extraction_config(tmp_path / "c", device="cpu")
    assert [r["status"] for r in run(cfg3, stages=["ingest", "extract"])] == ["done", "done"]


def test_container_commands(tmp_path, monkeypatch):
    binds = ["/data/run", "/data/x.h5ad"]
    cmd = container_command("/usr/bin/apptainer", "img.sif", binds, cwd="/data/run")
    assert cmd[:3] == ["/usr/bin/apptainer", "exec", "--nv"] and cmd[-2:] == ["img.sif", "python"]
    assert cmd.count("--bind") == 2 and "--pwd" in cmd
    cmd = container_command("docker", "ghcr.io/x/y:1", binds)
    assert cmd[:3] == ["docker", "run", "--rm"] and "--gpus" in cmd and cmd.count("-v") == 2 and cmd[-1] == "python"
    assert "nvidia.com/gpu=all" in container_command("podman", "img", [])
    # resolution finds a runtime on PATH (a fake one) and binds only existing paths, files by their directory
    fake = tmp_path / "bin" / "apptainer"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(fake.parent))
    monkeypatch.delenv("INTERPRETFM_CONTAINER_RUNTIME", raising=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.h5ad").write_text("")
    r = resolve_env(ModelEnv(container="img.sif"), binds=[tmp_path / "data" / "x.h5ad", tmp_path / "data", tmp_path / "missing"],
                    cwd=tmp_path / "run")
    assert r["kind"] == "container" and r["binds"] == [str((tmp_path / "data").resolve())] and r["command"][0] == str(fake)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(EnvError, match="no container runtime"):
        resolve_env(ModelEnv(container="img.sif"))


def test_config_binds_and_child_environment(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, store=True))
    binds = [str(b) for b in config_binds(cfg) if b is not None]
    assert str(cfg.dataset.h5ad) in binds and str(cfg.run.out_dir) in binds and str(cfg.stages["extract"]["store_root"]) in binds
    env = child_environment()
    assert env["python"] == sys.version.split()[0] and env["executable"] == sys.executable
    assert isinstance(env["packages"], list) and env["packages"] and env["torch"]["version"]


def test_cli_extract_verb(tmp_path):
    cfg = toy_extraction_config(tmp_path, cell_subset=50)
    cli = [sys.executable, "-m", "interpretfm.cli", "extract", "--config", str(cfg.source)]
    out = subprocess.run(cli + ["--model", "scgpt"], capture_output=True, text=True)
    assert out.returncode == 2 and "does not match model.name" in out.stderr
    out = subprocess.run(cli + ["--layers", "layer_0"], capture_output=True, text=True)
    assert out.returncode == 2 and "do not match model.layers" in out.stderr
    out = subprocess.run(cli + ["--model", "toy", "--layers", LAYER], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (stage_dir(cfg.run.out_dir, "extract") / "store.json").is_file()
    assert np.loadtxt(stage_dir(cfg.run.out_dir, "extract") / "cell_subset.txt", dtype=str).size == 50
