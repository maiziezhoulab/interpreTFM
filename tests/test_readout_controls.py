"""The post hoc controls of the gene-level readout: gene-symbol representations, the symbol-removed
variants, further stores, a fixed gene universe and a given first split (toy fixture)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import yaml

from interpretfm.config import load_config
from interpretfm.gene_readout import (
    make_splits,
    params_from_mapping,
    read_split_file,
    remove_component,
    symbol_representations,
    symbol_trigrams,
)
from interpretfm.runner import run
from interpretfm.testing import make_toy_adata, write_toy_run, write_toy_store

MODEL_STAGES = ["ingest", "extract", "sae", "concepts", "gene_readout"]


# --- gene-symbol controls and splits ------------------------------------------------------------


def test_symbol_trigrams_and_representations():
    assert symbol_trigrams("krt5") == {"^KR", "KRT", "RT5", "T5$"}
    symbols = ["KRT5", "KRT8", "CD3E", "CD3D", "MS4A1"]
    tri, pc, grams = symbol_representations(symbols, n_pcs=3)
    assert grams == sorted(grams) and tri.shape == (5, len(grams)) and set(np.unique(tri)) <= {0.0, 1.0}
    assert tri[0, grams.index("KRT")] == 1 and tri[2, grams.index("KRT")] == 0
    assert pc.shape == (5, 3) and np.allclose(pc.mean(0), 0, atol=1e-6)
    # the components are the centred matrix's principal-component scores
    C = tri - tri.mean(0)
    U, S, _ = np.linalg.svd(C.astype(np.float64), full_matrices=False)
    assert np.allclose(np.abs(pc), np.abs(U[:, :3] * S[:3]), atol=1e-5)


def test_remove_component_fits_on_the_given_rows_only():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((60, 3))
    X = (P @ rng.standard_normal((3, 4)) + 2.0 + 0.01 * rng.standard_normal((60, 4))).astype(np.float32)
    rows = np.arange(40)
    R = remove_component(X, P, rows)
    assert R.dtype == np.float32 and R.shape == X.shape
    assert np.abs(R).max() < 0.1  # X is (almost) linear in P with an intercept
    A = np.hstack([np.ones((60, 1)), P])
    B = np.linalg.lstsq(A[rows], X[rows].astype(np.float64), rcond=None)[0]
    assert np.allclose(R, X - A @ B, atol=1e-6)
    X2 = X.copy()
    X2[50] += 100.0  # a test row does not move the fit
    assert np.allclose(remove_component(X2, P, rows)[:40], R[:40], atol=1e-5)


def test_make_splits_with_a_given_first_split(tmp_path):
    genes = [f"g{i}" for i in range(100)]
    plain = make_splits(genes, 3, 0.2, 0.1, seed=0)
    first = (genes[10:95] + ["not_in_universe"], genes[:10])
    split = make_splits(genes[:95] + genes[96:], 3, 0.2, 0.1, seed=0, first=first)
    assert split["s0"] == (sorted(genes[10:95]), sorted(genes[:10]))
    assert split["s1"][1] == make_splits(genes[:95] + genes[96:], 3, 0.2, 0.1, seed=0)["s1"][1]
    assert plain["s1"][1] != split["s1"][1]  # the universe differs, so the draw differs
    (tmp_path / "split.json").write_text(json.dumps({"train_genes": ["a"], "valid_genes": ["b"], "test_genes": ["c"]}))
    assert read_split_file(tmp_path / "split.json") == (["a", "b"], ["c"])
    (tmp_path / "bad.json").write_text(json.dumps({"test_genes": ["c"]}))
    with pytest.raises(ValueError, match="train_genes"):
        read_split_file(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="both"):
        make_splits(genes, 2, 0.2, 0.1, 0, first=(genes[:50], genes[40:60]))


def test_readout_params_validation():
    p = params_from_mapping({"controls": ["names_tri", "names_pc50"], "symbol_residuals": ["hidden", "extra"],
                             "stores": {"extra": "/some/store"}, "representations": ["hidden"], "universe": "genes.txt",
                             "split": {"first_from": "split.json"}})
    assert p.stores == {"extra": "/some/store"} and p.first_split == "split.json" and list(p.representations) == ["hidden"]
    with pytest.raises(ValueError, match="stores.hidden"):
        params_from_mapping({"stores": {"hidden": "/x"}})
    with pytest.raises(ValueError, match="symbol_residuals"):
        params_from_mapping({"symbol_residuals": ["nothing"]})
    with pytest.raises(ValueError, match="representations"):
        params_from_mapping({"representations": ["dense"]})


def test_gene_readout_stage_with_the_post_hoc_controls(tmp_path):
    path = write_toy_run(tmp_path, store=True, env={"python": sys.executable})
    d = yaml.safe_load(open(path))
    adata = make_toy_adata()
    second = write_toy_store(tmp_path / "data" / "store_b", adata, hidden_dim=16, seed=5)
    genes = [str(g) for g in adata.var_names]
    (tmp_path / "genes.txt").write_text("# a subset of the panel\n" + "\n".join(genes[:45]) + "\n")
    (tmp_path / "first.json").write_text(json.dumps({"train_genes": genes[5:30], "valid_genes": genes[30:45], "test_genes": genes[:5]}))
    d["stages"]["gene_readout"] = {
        "n_splits": 2, "scaling": ["standardize"], "representations": ["hidden"],
        "controls": ["coexpression_pc50", "names_tri", "names_pc50"], "symbol_residuals": ["hidden", "coexpression_pc50"],
        "stores": {"other": str(second)}, "universe": "genes.txt", "n_pcs": 4,
        "split": {"valid": 0.2, "test": 0.1, "first_from": str(tmp_path / "first.json")}}
    yaml.safe_dump(d, open(path, "w"), sort_keys=False)
    cfg = load_config(path)
    rows = run(cfg, stages=MODEL_STAGES)
    assert [r["status"] for r in rows] == ["done"] * len(MODEL_STAGES)
    out = cfg.run.out_dir / "07_gene_readout"
    reps = json.load(open(out / "gene_readout.json"))["representations"]
    assert set(reps) == {"hidden|layer_1.norm2", "other|layer_1.norm2", "coexpression_pc50|-", "names_tri|-", "names_pc50|-",
                         "hidden|layer_1.norm2|symbol_removed", "coexpression_pc50|-|symbol_removed"}
    assert all(r["n_splits"] == 2 for r in reps.values())
    assert "mean_probe_test_auroc_center_mean" not in reps["hidden|layer_1.norm2"]
    splits = json.load(open(out / "splits.json"))
    assert len(splits["universe"]) == 45 and splits["n_listed"] == 45 and splits["listed_not_observed"] == []
    assert splits["splits"]["s0"]["test"] == sorted(genes[:5]) and splits["splits"]["s0"]["n_train_valid"] == 40
    sym = np.load(out / "symbol_reps.npz", allow_pickle=True)
    assert sym["names_tri"].shape[0] == 45 and sym["names_pc50"].shape == (45, 4)
    assert (out / "layer_1.norm2" / "gene_means_other.npz").is_file()
    means = np.load(out / "layer_1.norm2" / "gene_means.npz", allow_pickle=True)
    assert means["hidden"].shape == (50, 16) and "sae" not in means.files
