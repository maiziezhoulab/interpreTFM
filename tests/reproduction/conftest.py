"""Reproduction tests: the pipeline on the manuscript's cached data against the published numbers.

Every test in this directory carries the ``reproduction`` marker and is skipped unless the
data locations are configured (the ``INTERPRETFM_*`` variables of
``configs/paper/paths.env.example``, read from the environment or from the file named by
``INTERPRETFM_ENV_FILE``, default ``configs/paper/paths.env``). Tests that need a finished
pipeline run skip until ``make repro CONFIG=<name>`` has produced it (hours on a CPU), unless
``INTERPRETFM_REPRO_RUN=1`` lets the test run the pipeline itself. See ``docs/reproducing.md``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_DIR = REPO / "configs" / "paper"
DEFAULT_ENV_FILE = CONFIG_DIR / "paths.env"
REQUIRED = ("INTERPRETFM_DATA", "INTERPRETFM_STORES", "INTERPRETFM_MODELS", "INTERPRETFM_PAPER_CACHE", "INTERPRETFM_RUNS")
MODEL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout"]


def _load_env() -> List[str]:
    """Fill the environment from the env file (never overriding set variables); return the unset names."""
    path = os.environ.get("INTERPRETFM_ENV_FILE") or (str(DEFAULT_ENV_FILE) if DEFAULT_ENV_FILE.is_file() else None)
    if path:
        from interpretfm.config import ConfigError, load_env_file

        try:
            for k, v in load_env_file(path).items():
                os.environ.setdefault(k, v)
        except ConfigError as e:
            return [f"({e})"] + [k for k in REQUIRED if not os.environ.get(k)]
    return [k for k in REQUIRED if not os.environ.get(k)]


def pytest_configure(config):
    config.addinivalue_line("markers", "reproduction: compares results on the manuscript's cached data with the published numbers")


def pytest_collection_modifyitems(config, items):
    missing = _load_env()
    for item in items:
        if Path(str(item.fspath)).resolve().parent != HERE:
            continue
        item.add_marker(pytest.mark.reproduction)
        if missing:
            item.add_marker(pytest.mark.skip(
                reason=f"reproduction data not configured: set {missing} (copy configs/paper/paths.env.example "
                       "to configs/paper/paths.env; see docs/reproducing.md)"))


class PaperRun:
    """Access to a paper configuration and its (finished) run directory."""

    def __init__(self):
        self._configs: Dict[str, object] = {}

    def config(self, name: str):
        if name not in self._configs:
            from interpretfm.config import load_config

            path = CONFIG_DIR / f"{name}.yaml"
            if not path.is_file():
                pytest.skip(f"no paper configuration {path.name}")
            self._configs[name] = load_config(path)
        return self._configs[name]

    def run_dir(self, name: str, stages=MODEL_STAGES) -> Path:
        """The run directory with ``stages`` finished (runs them when ``INTERPRETFM_REPRO_RUN=1``)."""
        from interpretfm.runner import run, status

        cfg = self.config(name)
        rd = cfg.run.out_dir
        done = {r["stage"]: r["status"] for r in status(rd)} if rd.is_dir() else {}
        missing = [s for s in stages if done.get(s) not in ("done", "skipped")]
        if missing:
            if os.environ.get("INTERPRETFM_REPRO_RUN") == "1":
                run(cfg, stages=list(stages))
            else:
                pytest.skip(f"{name}: stage(s) {missing} not finished in {rd}; run `make repro CONFIG={name}` "
                            "(hours on a CPU) or set INTERPRETFM_REPRO_RUN=1 to let the test run them")
        return rd


@pytest.fixture(scope="session")
def paper() -> PaperRun:
    return PaperRun()
