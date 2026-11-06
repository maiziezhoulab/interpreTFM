"""Unit tests for the v1 pipeline core: configuration, runner (caching, resume, force), manifests,
the model-environment boundary and the ``interpretfm`` command. CPU only, toy data only."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from interpretfm import __version__
from interpretfm.config import ConfigError, check_files, load_config, parse_config
from interpretfm.envs import EnvError, resolve_env
from interpretfm.manifest import StageManifest
from interpretfm.runner import (
    StageError, list_stages, plan, register_stage, run, stage_dir, stage_is_current, status, unregister_stage,
)
from interpretfm.testing import TOY_UM_PER_UNIT, make_toy_adata, toy_config_dict, write_toy_run
from interpretfm.units import coords_to_um, um_to_native

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy(tmp_path):
    """A toy dataset and configuration in a temporary directory (paths relative to the YAML)."""
    return load_config(write_toy_run(tmp_path))


# --- configuration -------------------------------------------------------------------------------


def test_toy_config_loads(toy, tmp_path):
    assert toy.dataset.h5ad == (tmp_path / "data" / "toy.h5ad").resolve()
    assert toy.run.out_dir == (tmp_path / "runs" / "toy").resolve()
    assert toy.dataset.is_spatial and toy.dataset.spatial.um_per_unit == TOY_UM_PER_UNIT
    assert toy.model.env.kind == "current"
    assert toy.stage_params("tis") == {"trials": 50}
    assert toy.stage_params("sae") == {}
    assert check_files(toy) == []


def test_config_errors_are_listed_together():
    raw = toy_config_dict("x.h5ad", "runs/x")
    raw["dataset"]["spatial"] = {"obsm": "spatial"}  # no um_per_unit
    raw["model"]["layers"] = []
    raw["model"]["representative_layer"] = "nope"
    raw["stages"]["bogus"] = {}
    raw["typo"] = 1
    with pytest.raises(ConfigError) as e:
        parse_config(raw)
    msg = str(e.value)
    for frag in ("um_per_unit", "model.layers", "stages.bogus", "unknown key(s) ['typo']"):
        assert frag in msg


def test_config_rejects_multiple_envs_and_models_list():
    raw = toy_config_dict("x.h5ad", "runs/x", env={"conda": "a", "python": "/b"})
    raw["models"] = []
    with pytest.raises(ConfigError) as e:
        parse_config(raw)
    assert "only one of conda/prefix/python/container" in str(e.value)
    assert "'models' is not supported" in str(e.value)


def test_config_um_per_unit_positive():
    raw = toy_config_dict("x.h5ad", "runs/x")
    raw["dataset"]["spatial"]["um_per_unit"] = 0
    with pytest.raises(ConfigError, match="um_per_unit: must be > 0"):
        parse_config(raw)


def test_stage_hash_only_depends_on_relevant_sections(toy):
    h = toy.stage_hash("tis")
    other = copy.deepcopy(toy)
    other.stages["sae"] = {"steps": 10}
    other.run.name = "renamed"
    assert other.stage_hash("tis") == h
    other.stages["tis"] = {"trials": 51}
    assert other.stage_hash("tis") != h
    other2 = copy.deepcopy(toy)
    other2.dataset.spatial.um_per_unit = 1.0
    assert other2.stage_hash("tis") != h


def test_resolved_config_round_trips(toy, tmp_path):
    p = toy.write_resolved(tmp_path / "resolved.yaml")
    again = load_config(p)
    assert again.to_dict() == toy.to_dict()


def test_placeholders_expand_from_env_file_and_environment(tmp_path, monkeypatch):
    """``${NAME}`` in any string is filled from the environment, then an env file; unset names are errors."""
    yaml_path = write_toy_run(tmp_path)
    text = yaml_path.read_text().replace("h5ad: ", "h5ad: ${TOY_ROOT}/").replace("out_dir: ", "out_dir: ${TOY_RUNS}/")
    cfg_path = tmp_path / "placeholders.yaml"
    cfg_path.write_text(text + "\nstages:\n  tis: {trials: 3, store_root: '${TOY_STORE}'}\n")
    env_file = tmp_path / "paths.env"
    env_file.write_text("# comment\nexport TOY_ROOT=.\nTOY_RUNS=${TOY_ROOT}/elsewhere\nTOY_STORE='quoted'\n")
    with pytest.raises(ConfigError) as e:
        load_config(cfg_path)
    assert "${TOY_ROOT}" in str(e.value) and "dataset.h5ad" in str(e.value) and "stages.tis.store_root" in str(e.value)
    cfg = load_config(cfg_path, env_file=env_file)
    assert cfg.dataset.h5ad == (tmp_path / "data" / "toy.h5ad").resolve()
    assert cfg.run.out_dir == (tmp_path / "elsewhere" / "runs" / "toy").resolve()
    assert cfg.stage_params("tis")["store_root"] == "quoted"
    monkeypatch.setenv("TOY_RUNS", str(tmp_path / "from_env"))
    assert load_config(cfg_path, env_file=env_file).run.out_dir == (tmp_path / "from_env" / "runs" / "toy").resolve()
    assert load_config(cfg_path, env_file=env_file, env={"TOY_RUNS": "x"}).run.out_dir == (tmp_path / "x" / "runs" / "toy").resolve()


def test_validate_cli_reports_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\ndataset: {h5ad: a.h5ad, spatial: {obsm: spatial}}\nmodel: {name: m}\n")
    out = subprocess.run([sys.executable, "-m", "interpretfm.cli", "validate", str(bad)], capture_output=True, text=True)
    assert out.returncode == 2
    assert "um_per_unit" in out.stderr and "model.layers" in out.stderr and "run.out_dir" in out.stderr


def test_validate_cli_missing_file_and_example_config(tmp_path):
    (tmp_path / "configs").mkdir()
    cfg_path = tmp_path / "configs" / "toy.yaml"  # paths in the file are relative to configs/
    cfg_path.write_text((ROOT / "configs" / "toy.yaml").read_text())
    out = subprocess.run([sys.executable, "-m", "interpretfm.cli", "validate", str(cfg_path)],
                         capture_output=True, text=True)
    assert out.returncode == 3 and "dataset.h5ad: file not found" in out.stderr
    (tmp_path / "data").mkdir()
    make_toy_adata().write_h5ad(tmp_path / "data" / "toy.h5ad")
    out = subprocess.run([sys.executable, "-m", "interpretfm.cli", "validate", str(cfg_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


# --- units ---------------------------------------------------------------------------------------


def test_units():
    c = coords_to_um(np.array([[2.0, 4.0]]), 0.5)
    assert c.tolist() == [[1.0, 2.0]]
    assert um_to_native(21.6, 0.18) == pytest.approx(120.0)
    with pytest.raises(ValueError):
        um_to_native(1.0, 0.0)


# --- runner ---------------------------------------------------------------------------------------


def test_registry_has_all_builtin_stages_in_order():
    names = [s.name for s in list_stages()]
    assert names == ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers",
                     "niches", "interfaces", "lr_screen", "report"]
    assert [s.dir_name for s in list_stages()][:2] == ["01_ingest", "02_extract"]
    assert {s.name: s.env for s in list_stages()}["extract"] == "model"


def test_unknown_stage_is_an_error(toy):
    with pytest.raises(StageError, match="unknown stage 'nope'"):
        plan(toy, ["nope"])


@pytest.fixture
def counting_stage():
    """A trivial stage registered for the test, counting how often it runs; removed afterwards."""
    calls = []

    @register_stage("toy_count", inputs=["ingest"], outputs=["count.json"], order=50)
    def _toy_count(ctx):
        calls.append(ctx.params)
        ctx.write_json("count.json", {"n": len(calls), "n_cells": json.loads(
            (ctx.input_dir("ingest") / "dataset_qc.json").read_text())["n_cells"]})
        return {"runs": len(calls)}

    yield calls
    unregister_stage("toy_count")


def test_dry_run_writes_nothing(toy):
    rows = run(toy, dry_run=True)
    assert [r["status"] for r in rows] == ["dry-run"] * 13
    assert not toy.run.out_dir.exists()


def test_run_caches_resumes_and_forces(toy, counting_stage):
    rows = run(toy, stages=["ingest", "toy_count"])
    assert [r["status"] for r in rows] == ["done", "done"]
    assert len(counting_stage) == 1
    run_dir = toy.run.out_dir
    assert (run_dir / "config.resolved.yaml").is_file() and (run_dir / "manifest.json").is_file()
    assert (run_dir / "log.txt").is_file() and (run_dir / "01_ingest" / "log.txt").is_file()
    m = StageManifest.read(run_dir / "50_toy_count")
    assert m.status == "done" and m.extra == {"runs": 1} and m.inputs == {"ingest": "01_ingest"}
    assert m.config_hash == toy.stage_hash("toy_count") and m.package_version == __version__
    assert m.environment["python"] and m.duration_s is not None and m.ended
    assert json.loads((run_dir / "50_toy_count" / "count.json").read_text())["n_cells"] == 300

    # second run: everything cached
    rows = run(toy, stages=["ingest", "toy_count"])
    assert [r["status"] for r in rows] == ["cached", "cached"] and len(counting_stage) == 1

    # --force re-runs the selected stages only
    rows = run(toy, stages=["toy_count"], force=True)
    assert [r["status"] for r in rows] == ["done"] and len(counting_stage) == 2

    # a changed parameter invalidates the cache; a missing output too
    other = copy.deepcopy(toy)
    other.stages["tis"] = {"trials": 7}
    assert stage_is_current(list_stages()[0], run_dir, other)  # ingest does not read stages.tis
    (run_dir / "50_toy_count" / "count.json").unlink()
    assert not stage_is_current([s for s in list_stages() if s.name == "toy_count"][0], run_dir, toy)
    rows = run(toy, stages=["toy_count"])
    assert rows[0]["status"] == "done" and len(counting_stage) == 3

    st = status(run_dir)
    assert {r["stage"]: r["status"] for r in st}["ingest"] == "done"
    assert {r["stage"]: r["status"] for r in st}["tis"] == "-"


def test_missing_upstream_stage_is_an_error(toy, counting_stage):
    with pytest.raises(StageError, match="needs 'ingest'"):
        run(toy, stages=["toy_count"])


def test_failed_stage_writes_manifest_and_stops(toy):
    @register_stage("toy_fail", inputs=["ingest"], outputs=["never.json"], order=51)
    def _toy_fail(ctx):
        raise NotImplementedError("stage 'toy_fail' is not implemented")

    try:
        with pytest.raises(StageError, match="not implemented"):
            run(toy, stages=["ingest", "toy_fail"])
        m = StageManifest.read(stage_dir(toy.run.out_dir, "toy_fail"))
        assert m.status == "failed" and "NotImplementedError" in m.error
        assert {r["stage"]: r["status"] for r in status(toy.run.out_dir)}["toy_fail"] == "failed"
    finally:
        unregister_stage("toy_fail")

    # stubs of the scientific stages fail with a clear message
    with pytest.raises(StageError, match="needs 'extract', which has not been run"):
        run(toy, stages=["tis"])


def test_spatial_stage_skipped_without_coordinates(tmp_path):
    cfg = load_config(write_toy_run(tmp_path, spatial=False))
    assert not cfg.dataset.is_spatial
    rows = plan(cfg)
    assert {r["stage"]: r["action"] for r in rows}["niches"] == "skip-na"
    run(cfg, stages=["ingest", "niches"])
    assert StageManifest.read(stage_dir(cfg.run.out_dir, "niches")).status == "skipped"
    qc = json.loads((stage_dir(cfg.run.out_dir, "ingest") / "dataset_qc.json").read_text())
    assert qc["spatial"] is None and qc["labels"]["n_labels"] == 3


def test_ingest_converts_coordinates_once(toy):
    run(toy, stages=["ingest"])
    qc = json.loads((stage_dir(toy.run.out_dir, "ingest") / "dataset_qc.json").read_text())
    assert qc["spatial"]["um_per_unit"] == TOY_UM_PER_UNIT
    coords = np.loadtxt(stage_dir(toy.run.out_dir, "ingest") / "coords_um.csv", delimiter=",", skiprows=1)
    import anndata as ad

    native = ad.read_h5ad(toy.dataset.h5ad).obsm["spatial"]
    assert np.allclose(coords[:, 1:], native * TOY_UM_PER_UNIT, atol=1e-3)  # written with 4 decimals


# --- environment boundary -------------------------------------------------------------------------


def test_env_boundary_runs_stage_in_subprocess(tmp_path):
    """With env.python = this interpreter, the model stage runs as a child process via `interpretfm stage`."""
    cfg = load_config(write_toy_run(tmp_path, env={"python": sys.executable}))
    assert cfg.model.env.kind == "python"
    assert resolve_env(cfg.model.env)["command"] == [sys.executable]
    with pytest.raises(StageError, match="failed in the model environment"):
        run(cfg, stages=["ingest", "extract"])
    sdir = stage_dir(cfg.run.out_dir, "extract")
    m = StageManifest.read(sdir)
    assert m.status == "failed" and "ExtractionError" in m.error and "model.checkpoint" in m.error  # written by the child
    log = (sdir / "log.txt").read_text()
    assert "interpretfm.cli extract --config" in log and "--model scgpt" in log


def test_env_boundary_missing_env_is_clear(tmp_path, monkeypatch):
    cfg = load_config(write_toy_run(tmp_path, env={"python": str(tmp_path / "no" / "python")}))
    with pytest.raises(EnvError, match="interpreter not found"):
        resolve_env(cfg.model.env)
    with pytest.raises(StageError, match="interpreter not found"):
        run(cfg, stages=["ingest", "extract"])
    assert StageManifest.read(stage_dir(cfg.run.out_dir, "extract")).status == "failed"

    cfg2 = load_config(write_toy_run(tmp_path / "b", env={"conda": "interpretfm-no-such-env-xyz"}))
    monkeypatch.setenv("INTERPRETFM_CONDA", str(tmp_path / "no" / "conda"))
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(EnvError, match="conda was not found"):
        resolve_env(cfg2.model.env)

    cfg3 = load_config(write_toy_run(tmp_path / "c", env={"container": "ghcr.io/x/y"}))
    monkeypatch.delenv("INTERPRETFM_CONTAINER_RUNTIME", raising=False)
    with pytest.raises(EnvError, match="no container runtime"):  # PATH holds no apptainer/docker/podman
        resolve_env(cfg3.model.env)


@pytest.mark.skipif(not __import__("interpretfm.envs").envs.find_conda(), reason="conda not available")
def test_env_boundary_unknown_conda_env_is_clear():
    cfg = parse_config(toy_config_dict("x.h5ad", "runs/x", env={"conda": "interpretfm-no-such-env-xyz"}))
    with pytest.raises(EnvError, match="does not exist"):
        resolve_env(cfg.model.env)


# --- command line ----------------------------------------------------------------------------------


def test_cli_end_to_end(tmp_path):
    cfg_path = write_toy_run(tmp_path)
    cli = [sys.executable, "-m", "interpretfm.cli"]
    out = subprocess.run(cli + ["version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == __version__
    out = subprocess.run(cli + ["run", str(cfg_path), "--dry-run"], capture_output=True, text=True)
    assert out.returncode == 0 and "dry-run" in out.stdout and not (tmp_path / "runs").exists()
    out = subprocess.run(cli + ["run", str(cfg_path), "--stages", "ingest"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "runs" / "toy" / "01_ingest" / "manifest.json").is_file()
    out = subprocess.run(cli + ["run", str(cfg_path), "--stages", "ingest,extract"], capture_output=True, text=True)
    assert out.returncode == 1 and "model.checkpoint is required" in out.stderr  # env {}: in-process, no weights
    assert "model.checkpoint is required" in (tmp_path / "runs" / "toy" / "02_extract" / "log.txt").read_text()
    out = subprocess.run(cli + ["status", str(tmp_path / "runs" / "toy")], capture_output=True, text=True)
    assert out.returncode == 0 and "failed" in out.stdout and "cached" not in out.stdout
    out = subprocess.run(cli + ["report", str(tmp_path / "runs" / "toy")], capture_output=True, text=True)
    assert out.returncode == 1 and "needs 'layers'" in out.stderr  # the report needs the finished layer summary
    for sub in ("validate", "run", "status", "report"):
        out = subprocess.run(cli + [sub, "--help"], capture_output=True, text=True)
        assert out.returncode == 0 and "usage:" in out.stdout


def test_console_script_installed():
    """The console script is registered in pyproject (its availability on PATH depends on the install)."""
    text = (ROOT / "pyproject.toml").read_text()
    assert 'interpretfm = "interpretfm.cli:main"' in text
    resolved = yaml.safe_load((ROOT / "configs" / "toy.yaml").read_text())
    assert resolved["dataset"]["spatial"]["um_per_unit"] == TOY_UM_PER_UNIT
