"""Toy fixtures for tests, examples and the tutorial: a tiny synthetic dataset and configuration.

Nothing here is a real dataset; the values are random counts with a few cell populations and a
regular spatial layout, enough to exercise the pipeline mechanics on a laptop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

#: Micrometres per coordinate unit of the toy dataset (coordinates are stored in "pixels").
TOY_UM_PER_UNIT = 0.5


def make_toy_adata(n_cells: int = 300, n_genes: int = 50, n_types: int = 3, seed: int = 0, spatial: bool = True):
    """A small AnnData with raw counts in ``X``, ``obs['cell_type']`` and (optionally) ``obsm['spatial']``.

    Gene names are ``G000``..., cells ``cell_0``...; each cell type over-expresses its own block of genes.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    types = rng.integers(0, n_types, size=n_cells)
    base = rng.gamma(shape=2.0, scale=1.0, size=n_genes)
    X = np.empty((n_cells, n_genes), dtype=np.float32)
    block = max(1, n_genes // n_types)
    for i in range(n_cells):
        mu = base.copy()
        t = types[i]
        mu[t * block:(t + 1) * block] *= 4.0
        X[i] = rng.poisson(mu)
    obs = pd.DataFrame({"cell_type": pd.Categorical([f"type_{t}" for t in types])},
                       index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"G{i:03d}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if spatial:
        side = int(np.ceil(np.sqrt(n_cells)))
        grid = np.array([(i % side, i // side) for i in range(n_cells)], dtype=np.float64)
        adata.obsm["spatial"] = grid * 20.0 + rng.normal(scale=2.0, size=grid.shape)  # "pixels", 20 px apart
    return adata


def toy_config_dict(h5ad: Union[str, Path], out_dir: Union[str, Path], spatial: bool = True,
                    env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The configuration mapping for a toy run (paths as given; use absolute paths or paths relative to the YAML)."""
    cfg: Dict[str, Any] = {
        "run": {"name": "toy", "out_dir": str(out_dir), "seed": 0},
        "dataset": {
            "h5ad": str(h5ad),
            "counts_layer": None,
            "gene_id": {"column": "index", "kind": "symbol"},
            "labels": {"column": "cell_type"},
            "qc": {"min_transcripts": 10, "min_genes": 5},
        },
        "model": {
            "name": "scgpt",
            "checkpoint": None,
            "env": env or {},
            "layers": ["layer_1.norm2"],
            "cell_pooling": {"kind": "token", "token": "60695"},
            "special_tokens": ["60694", "60695"],
            "representative_layer": "layer_1.norm2",
        },
        "stages": {"extract": {"batch_size": 16, "rows_per_shard": 1024}, "tis": {"trials": 50}},
        "output": {"html": True, "json": True},
    }
    if spatial:
        cfg["dataset"]["spatial"] = {"obsm": "spatial", "um_per_unit": TOY_UM_PER_UNIT}
    return cfg


def write_toy_run(root: Union[str, Path], n_cells: int = 300, n_genes: int = 50, spatial: bool = True,
                  env: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<root>/data/toy.h5ad`` and ``<root>/toy.yaml`` (run directory ``<root>/runs/toy``); returns the YAML path."""
    import yaml

    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    make_toy_adata(n_cells=n_cells, n_genes=n_genes, spatial=spatial).write_h5ad(root / "data" / "toy.h5ad")
    cfg = toy_config_dict("data/toy.h5ad", "runs/toy", spatial=spatial, env=env)
    path = root / "toy.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path
