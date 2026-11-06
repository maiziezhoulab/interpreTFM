"""Toy fixtures for tests, examples and the tutorial: a tiny synthetic dataset and configuration.

Nothing here is a real dataset; the values are random counts with a few cell populations and a
regular spatial layout, enough to exercise the pipeline mechanics on a laptop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
                    env: Optional[Dict[str, Any]] = None, store_root: Optional[Union[str, Path]] = None,
                    concepts_csv: Optional[Union[str, Path]] = None, lr_database: Optional[Union[str, Path]] = None,
                    lr_complexes: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """The configuration mapping for a toy run (paths as given; use absolute paths or paths relative to the YAML).

    With ``store_root`` the model-level stages are configured too (:func:`toy_stage_params`):
    ``extract`` uses that cached store, ``concepts`` loads ``concepts_csv`` offline and ``lr_screen``
    screens the pairs of ``lr_database`` (with the complexes of ``lr_complexes``).
    """
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
    if store_root is not None:
        cfg["stages"]["extract"]["store_root"] = str(store_root)
        cfg["stages"].update(toy_stage_params(concepts_csv, lr_database, lr_complexes))
    return cfg


def toy_stage_params(concepts_csv: Optional[Union[str, Path]] = None, lr_database: Optional[Union[str, Path]] = None,
                     lr_complexes: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Tiny settings for the model-level stages so that the toy pipeline runs on a CPU in seconds."""
    params: Dict[str, Dict[str, Any]] = {
        "tis": {"trials": 40, "k_ref": 8, "controls": ["shuffle", "pca"]},
        "sae": {"width_multiplier": 2, "lambda": {"grid": [1.0, 3.0], "rule": {"l0_max": 0.3, "dead_max": 0.25}},
                "steps": 300, "lr": 1.0e-2, "warmup": 5, "resample_every": 100, "batch_size": 64, "device": "cpu",
                "qc": {"max_shards": 4, "rows_per_shard": 256}},
        "concepts": {},
        "concept_eval": {"top_m": 20, "batch_size": 1024},
        "gene_readout": {"n_splits": 2, "controls": ["random", "coexpression", "coexpression_pc50", "celltype", "trivial"]},
        "cells": {"retain": {"min_features": 3}, "specificity": {"mhr_k": 10, "n_assoc": 10, "n_null": 5}},
        "niches": {"radii_um": [12.0, 25.0], "k": [2, 3], "methods": ["kmeans"], "seeds": [0, 1], "min_cluster": 5,
                   "n_pcs": 8, "n_null": 3, "label_composition": {"radius_um": 12.0, "k": 2, "n_init": 3},
                   "seed_stability": {"k": [3]}, "concepts": {"top_k": 3, "min_cells": 5},
                   "fixed_configs": [{"radius_um": 12.0, "space": "m", "method": "kmeans", "k": 2}]},
        "interfaces": {"edge_radius_um": 12.0, "block_um": 25.0, "min_edges": 5, "n_perm": 20, "calib_perm": 10, "calib_reps": 1,
                       "legacy_perm": 5, "markers": {"top_n": 3}, "fired_on": {"max_shards": 2}},
        "lr_screen": {"radius_um": 12.0, "thresholds": [1, 2], "n_perm": 20, "seed": 0, "min_positive": 10},
    }
    if concepts_csv is not None:
        params["concepts"]["offline_matrix"] = str(concepts_csv)
    if lr_database is not None:
        params["lr_screen"]["database"] = str(lr_database)
        params["lr_screen"]["complexes"] = str(lr_complexes) if lr_complexes is not None else False
    return params


def make_toy_lr_database(adata, n_pairs: int = 6, seed: int = 0):
    """A small ligand–receptor table over the toy panel (``pandas.DataFrame`` in the CellChatDB layout) and a
    complex table with one two-subunit receptor complex (``RC``), so that complex resolution is exercised."""
    import pandas as pd

    rng = np.random.default_rng(seed + 23)
    genes = [str(g) for g in adata.var_names]
    pick = rng.choice(len(genes), size=2 * n_pairs + 2, replace=False)
    rows = []
    for k in range(n_pairs):
        lg, rg = genes[pick[2 * k]], genes[pick[2 * k + 1]]
        rows.append({"interaction_name": f"{lg}_{rg}", "pathway_name": f"TOY{k % 2}", "ligand": lg, "receptor": rg, "annotation": "Cell-Cell Contact"})
    rows.append({"interaction_name": f"{genes[pick[0]]}_RC", "pathway_name": "TOY2", "ligand": genes[pick[0]], "receptor": "RC", "annotation": "Secreted Signaling"})
    rows.append({"interaction_name": "NOTONPANEL_RC", "pathway_name": "TOY2", "ligand": "NOTONPANEL", "receptor": "RC", "annotation": "Secreted Signaling"})
    db = pd.DataFrame(rows)
    cx = pd.DataFrame([{"complex_name": "RC", "subunit_1": genes[pick[-2]], "subunit_2": genes[pick[-1]], "subunit_3": ""}])
    return db, cx


def make_toy_concepts(adata, n_concepts: int = 8, size_range: Tuple[int, int] = (5, 12), seed: int = 0):
    """A gene x concept 0/1 table (``pandas.DataFrame``, genes as index) over the toy panel.

    Concepts are random gene sets of ``size_range`` genes; concept ``j`` is named ``TOY:j``.
    """
    import pandas as pd

    rng = np.random.default_rng(seed + 11)
    genes = [str(g) for g in adata.var_names]
    mat = np.zeros((len(genes), n_concepts), dtype=np.int8)
    for j in range(n_concepts):
        size = int(rng.integers(size_range[0], min(size_range[1], len(genes)) + 1))
        mat[rng.choice(len(genes), size=size, replace=False), j] = 1
    return pd.DataFrame(mat, index=genes, columns=[f"TOY:{j:04d}" for j in range(n_concepts)])


def write_toy_store(root: Union[str, Path], adata, layers: Sequence[str] = ("layer_1.norm2",), hidden_dim: int = 16,
                    concepts=None, seed: int = 0, rows_per_shard: int = 1500, cls_token: str = "60695",
                    duplicate_cells: int = 0) -> Path:
    """Write a synthetic activation store (0.9 sharded format) for the toy dataset.

    Every cell gets one classification-token row (``token_id == cls_token``) followed by one
    gene-token row per detected gene (``token_id`` = gene name, ``example_id`` = cell name).
    Gene rows carry a concept direction (from ``concepts``, a gene x concept table), a cell-type
    direction and noise, so that concept-aligned features exist; the classification row
    carries the cell-type direction plus the mean gene embedding so that TIS is above its
    shuffled control. ``duplicate_cells`` > 0 writes that many cells a second time (to test
    de-duplication). Returns the store root.
    """
    import torch

    from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

    root = Path(root)
    rng = np.random.default_rng(seed)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    genes = [str(g) for g in adata.var_names]
    cells = [str(c) for c in adata.obs_names]
    types = adata.obs["cell_type"].astype(str).values if "cell_type" in adata.obs else np.array(["t"] * len(cells))
    type_names = sorted(set(types))
    n_concepts = 0 if concepts is None else int(concepts.shape[1])
    store = ActivationStore(ActivationStoreSpec(root=str(root)))
    for li, layer in enumerate(layers):
        lrng = np.random.default_rng(seed + 101 * (li + 1))
        U = lrng.normal(size=(max(n_concepts, 1), hidden_dim)).astype(np.float32) * 2.0
        V = lrng.normal(size=(len(type_names), hidden_dim)).astype(np.float32)
        E = lrng.normal(scale=0.5, size=(len(genes), hidden_dim)).astype(np.float32)
        if concepts is not None:
            M = concepts.reindex(genes).fillna(0).values.astype(np.float32)
            E += M @ U
        order = list(range(len(cells))) + list(range(min(duplicate_cells, len(cells))))
        acts, ex, tok, shard_id = [], [], [], 0
        for ci in order:
            t = type_names.index(types[ci])
            nz = np.flatnonzero(X[ci] > 0)
            gene_rows = E[nz] + 0.5 * V[t] + rng.normal(scale=0.3, size=(len(nz), hidden_dim)).astype(np.float32)
            cls_row = V[t] + 0.5 * E[nz].mean(axis=0) + rng.normal(scale=0.3, size=hidden_dim).astype(np.float32)
            acts.append(cls_row[None, :])
            acts.append(gene_rows)
            ex.extend([cells[ci]] * (len(nz) + 1))
            tok.append(cls_token)
            tok.extend(genes[g] for g in nz)
            if len(ex) >= rows_per_shard:
                store.write_token_shard(layer, shard_id, torch.from_numpy(np.concatenate(acts)), ex, tok, "gene")
                acts, ex, tok, shard_id = [], [], [], shard_id + 1
        if ex:
            store.write_token_shard(layer, shard_id, torch.from_numpy(np.concatenate(acts)), ex, tok, "gene")
    return root


def write_toy_run(root: Union[str, Path], n_cells: int = 300, n_genes: int = 50, spatial: bool = True,
                  env: Optional[Dict[str, Any]] = None, store: bool = False, hidden_dim: int = 16,
                  duplicate_cells: int = 0) -> Path:
    """Write ``<root>/data/toy.h5ad`` and ``<root>/toy.yaml`` (run directory ``<root>/runs/toy``); returns the YAML path.

    ``store=True`` also writes a synthetic activation store (``data/store``), a concept table
    (``data/concepts.csv``) and a toy ligand–receptor table (``data/lr_pairs.csv``, ``data/lr_complexes.csv``)
    and configures the model-level stages with tiny settings.
    """
    import yaml

    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    adata = make_toy_adata(n_cells=n_cells, n_genes=n_genes, spatial=spatial)
    adata.write_h5ad(root / "data" / "toy.h5ad")
    store_root = concepts_csv = lr_db = lr_cx = None
    if store:
        concepts = make_toy_concepts(adata)
        concepts_csv = "data/concepts.csv"
        concepts.to_csv(root / concepts_csv)
        store_root = root.resolve() / "data" / "store"
        write_toy_store(store_root, adata, hidden_dim=hidden_dim, concepts=concepts, duplicate_cells=duplicate_cells)
        concepts_csv = root.resolve() / concepts_csv  # stage parameters hold absolute paths
        db, cx = make_toy_lr_database(adata)
        lr_db, lr_cx = root.resolve() / "data" / "lr_pairs.csv", root.resolve() / "data" / "lr_complexes.csv"
        db.to_csv(lr_db, index=False)
        cx.to_csv(lr_cx, index=False)
    cfg = toy_config_dict("data/toy.h5ad", "runs/toy", spatial=spatial, env=env, store_root=store_root,
                          concepts_csv=concepts_csv, lr_database=lr_db, lr_complexes=lr_cx)
    path = root / "toy.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path
