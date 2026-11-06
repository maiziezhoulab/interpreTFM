"""Model-level stages (tis, sae, concepts, concept_eval, gene_readout) on the toy fixture, plus
unit tests of the split assignment, the λ rule, the concept-F1 summary and the readout controls."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from interpretfm.concept_eval import HeldoutParams, HeldoutResult, per_concept_best, summarize_concept_f1
from interpretfm.config import load_config
from interpretfm.gene_readout import ReadoutParams, make_splits, probe_auroc, readout, single_dim_auroc
from interpretfm.genes import GeneTable, TokenResolver, load_token_vocab
from interpretfm.runner import run
from interpretfm.sae import select_lambda
from interpretfm.testing import make_toy_adata, make_toy_concepts, write_toy_run, write_toy_store

MODEL_STAGES = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout"]


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """The toy pipeline through gene_readout (with 20 duplicated cells in the store)."""
    root = tmp_path_factory.mktemp("toy")
    yaml_path = write_toy_run(root, store=True, env={"python": sys.executable}, duplicate_cells=20)
    cfg = load_config(yaml_path)
    rows = run(cfg, stages=MODEL_STAGES)
    assert [r["status"] for r in rows] == ["done"] * len(MODEL_STAGES)
    return cfg, cfg.run.out_dir


# --- end to end ---------------------------------------------------------------------------------


def test_tis_stage_outputs(toy_run):
    cfg, run_dir = toy_run
    tis = json.load(open(run_dir / "03_tis" / "tis.json"))
    layer = tis["layers"]["layer_1.norm2"]
    assert layer["n_cells"] == 300 and layer["hidden_dim"] == 16
    assert layer["gain"] > 0.1 and 0.35 < layer["shuffled"]["mean"] < 0.65  # shuffled control at chance
    per_dim = pd.read_csv(run_dir / "03_tis" / "layer_1.norm2" / "tis_per_dim.csv")
    assert list(per_dim.columns) == ["dim", "tis", "tis_shuffled", "tis_pca"] and len(per_dim) == 16
    A = np.load(run_dir / "03_tis" / "layer_1.norm2" / "cell_activations.npy")
    assert A.shape == (300, 16)


def test_sae_stage_selects_lambda_and_writes_qc(toy_run):
    cfg, run_dir = toy_run
    sel = json.load(open(run_dir / "04_sae" / "selected.json"))
    entry = sel["layers"]["layer_1.norm2"]
    assert entry["lambda"] in (1.0, 3.0) and entry["rule"].startswith("l0_frac<=")
    for lam in ("1", "3"):
        qc = entry["qc"][lam]
        assert 0 < qc["fvu"] < 1 and 0 < qc["l0_frac"] < 1 and qc["n_latents"] == 32
        ckpt = torch.load(qc["checkpoint"], map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        assert set(sd) == {"encoder.weight", "encoder.bias", "bias", "decoder.weight"}
        assert sd["encoder.weight"].shape == (32, 16)
        assert torch.allclose(sd["decoder.weight"].norm(dim=0) * ckpt["input_scale"], torch.ones(32), atol=1e-4)
    assert entry["untrained"]["n_latents"] == 32 and (run_dir / "04_sae" / "layer_1.norm2" / "untrained" / "sae.pt").exists()
    hist = pd.read_csv(run_dir / "04_sae" / "layer_1.norm2" / "lambda_1" / "history.csv")
    assert len(hist) == 300 and hist["loss"].iloc[-1] < hist["loss"].iloc[0]


def test_concepts_stage_offline_universe(toy_run):
    cfg, run_dir = toy_run
    u = json.load(open(run_dir / "05_concepts" / "universe.json"))
    assert u["n_concepts"] == 8 and u["manifest"]["source"] == "offline_matrix"
    m = pd.read_csv(run_dir / "05_concepts" / "universe" / "gene_by_concept.csv", index_col=0)
    assert m.shape == (50, 8) and list(m.index) == [f"G{i:03d}" for i in range(50)]
    genes = pd.read_csv(run_dir / "05_concepts" / "genes.csv")
    assert list(genes["key"]) == list(m.index)


def test_concept_eval_stage(toy_run):
    cfg, run_dir = toy_run
    table = pd.read_csv(run_dir / "06_concept_eval" / "concept_f1.csv")
    assert set(table["unit"]) == {"row", "gene"} and set(table["representation"]) == {"sae", "raw", "untrained"}
    row = table[table["unit"] == "row"].set_index("representation")
    assert row.loc["sae", "n_concepts"] == 8 and row.loc["raw", "n_units"] == 32
    assert row.loc["sae", "concept_f1"] > row.loc["raw", "concept_f1"]  # planted concept directions
    split = json.load(open(run_dir / "06_concept_eval" / "layer_1.norm2" / "f1_row_sae" / "split.json"))
    assert split["dedup_cells"] and split["rows_dropped_duplicate_cells"] > 0 and split["n_cells"] == 300
    gene_split = json.load(open(run_dir / "06_concept_eval" / "layer_1.norm2" / "f1_gene_sae" / "split.json"))
    assert gene_split["n_valid_genes"] == 10 and gene_split["n_test_genes"] == 5
    assert not (set(gene_split["test_genes"]) & set(gene_split["valid_genes"]))
    valid = pd.read_csv(run_dir / "06_concept_eval" / "layer_1.norm2" / "f1_row_sae" / "valid_concept_f1.csv")
    assert sorted(valid["threshold"].unique()) == [0.0, 0.15, 0.3, 0.6]
    assert valid.groupby(["concept", "threshold"]).size().max() <= 20  # top_m of the toy settings


def test_gene_readout_stage(toy_run):
    cfg, run_dir = toy_run
    s = json.load(open(run_dir / "07_gene_readout" / "gene_readout.json"))
    reps = s["representations"]
    assert {"hidden|layer_1.norm2", "sae|layer_1.norm2", "untrained|layer_1.norm2", "random_16|-", "coexpression|-",
            "coexpression_pc50|-", "celltype|-", "trivial|-"} <= set(reps)
    assert reps["hidden|layer_1.norm2"]["n_splits"] == 2
    for key in ("mean_probe_test_auroc_standardize_mean", "mean_probe_test_auroc_center_mean", "mean_single_test_auroc_mean"):
        assert 0 <= reps["hidden|layer_1.norm2"][key] <= 1
    means = np.load(run_dir / "07_gene_readout" / "layer_1.norm2" / "gene_means.npz", allow_pickle=True)
    assert means["hidden"].shape == (50, 16) and means["sae"].shape == (50, 32) and (means["count"] > 0).all()
    pc = pd.read_csv(run_dir / "07_gene_readout" / "per_concept.csv.gz")
    assert {"split", "representation", "concept", "single_test_auroc", "probe_test_auroc_standardize"} <= set(pc.columns)


# --- splits -------------------------------------------------------------------------------------


def test_row_split_is_deterministic_and_has_the_right_fractions():
    from interp_pipeline.get_annotation.f1_alignment import _assign_gene_token_row_split

    def split(seed):
        return np.array([_assign_gene_token_row_split(seed=seed, shard_name="shard_3", row_index=i, token_id=f"G{i % 50}",
                                                      valid_frac=0.2, test_frac=0.1) for i in range(20000)])

    a, b = split(0), split(0)
    assert (a == b).all()
    assert abs((a == 0).mean() - 0.2) < 0.01 and abs((a == 1).mean() - 0.1) < 0.01 and abs((a == -1).mean() - 0.7) < 0.01
    assert (split(1) != a).mean() > 0.3  # another seed is another split


def test_gene_split_keeps_all_rows_of_a_gene_together(tmp_path):
    from interpretfm.concept_eval import assign_rows

    adata = make_toy_adata(n_cells=60)
    concepts = make_toy_concepts(adata)
    store = write_toy_store(tmp_path / "store", adata, concepts=concepts)
    genes = GeneTable.from_adata(adata, type("G", (), {"column": "index", "kind": "symbol"})())
    resolver = TokenResolver(genes, special_tokens=["60695"])
    key_to_row = {k: i for i, k in enumerate(genes.keys)}
    params = HeldoutParams(seed=0)
    gene = assign_rows(store, "layer_1.norm2", resolver, key_to_row, "gene", params)
    test_rows = {r for name in gene.gt_rows for r in gene.gt_rows[name][gene.split[name] == 1]}
    valid_rows = {r for name in gene.gt_rows for r in gene.gt_rows[name][gene.split[name] == 0]}
    assert test_rows and valid_rows and not (test_rows & valid_rows)
    assert {genes.keys[r] for r in test_rows} == set(gene.genes_by_split["test"])
    row = assign_rows(store, "layer_1.norm2", resolver, key_to_row, "row", params)
    mixed = {r for name in row.gt_rows for r in row.gt_rows[name][row.split[name] == 1]} & \
            {r for name in row.gt_rows for r in row.gt_rows[name][row.split[name] == 0]}
    assert mixed  # the row split puts rows of one gene in both partitions
    assert all((row.gt_rows[name] >= 0).sum() == (row.gt_rows[name] >= 0).sum() for name in row.gt_rows)
    assert all(((row.gt_rows[n] < 0) == (gene.gt_rows[n] < 0)).all() for n in row.gt_rows)
    assert (gene.gt_rows[next(iter(gene.gt_rows))][0]) == -1  # the classification-token row maps to nothing


# --- the λ rule ---------------------------------------------------------------------------------


def _qc(fvu, l0, dead):
    return {"fvu": fvu, "l0_frac": l0, "dead_frac": dead}


def test_select_lambda_rule():
    qc = {0.1: _qc(0.03, 0.08, 0.0), 0.3: _qc(0.05, 0.03, 0.0), 1.0: _qc(0.058, 0.0108, 0.0),
          3.0: _qc(0.09, 0.005, 0.1), 10.0: _qc(0.2, 0.001, 0.4)}
    lam, rule = select_lambda(qc)
    assert lam == 1.0 and rule.startswith("l0_frac<=0.02")  # lowest FVU among L0 <= 2 %, dead <= 0.25
    lam, _ = select_lambda(qc, l0_max=0.1)  # with a looser L0 bound the lowest FVU wins
    assert lam == 0.1
    lam, _ = select_lambda(qc, dead_max=0.05)  # 3.0 excluded by the dead-feature bound; 1.0 still qualifies
    assert lam == 1.0
    lam, rule = select_lambda({1.0: _qc(0.1, 0.05, 0.0), 3.0: _qc(0.15, 0.03, 0.0), 10.0: _qc(0.5, 0.01, 0.4)})
    assert lam == 3.0 and rule.startswith("fallback")  # none qualifies: lowest L0 with FVU <= 0.2
    lam, rule = select_lambda({1.0: _qc(0.5, 0.05, 0.0)})
    assert lam is None and rule == "none"
    lam, _ = select_lambda({1.0: _qc(0.05, 0.01, 0.0), 3.0: _qc(0.05, 0.005, 0.0)})
    assert lam == 1.0  # ties on FVU go to the smaller λ


# --- the concept-F1 summary ---------------------------------------------------------------------


def _f1_frame(rows):
    return pd.DataFrame(rows, columns=["concept", "feature", "threshold", "f1"])


def test_concept_f1_summary_rule():
    valid = _f1_frame([
        ("A", 0, 0.15, 0.9), ("A", 7, 0.15, 0.9), ("A", 3, 0.15, 0.5),   # tie at 0.9 -> feature 0 (lowest index)
        ("B", 5, 0.15, 0.7), ("B", 2, 0.15, 0.6),
        ("C", 1, 0.15, 0.8),                                              # too large a concept
        ("D", 4, 0.15, 0.8),                                              # no test support
        ("A", 3, 0.3, 0.95),                                              # another threshold is ignored
    ])
    test = _f1_frame([("A", 0, 0.15, 0.4), ("A", 7, 0.15, 0.99), ("A", 3, 0.15, 0.1), ("B", 5, 0.15, 0.6),
                      ("B", 2, 0.15, 0.9), ("C", 1, 0.15, 0.7), ("D", 4, 0.15, 0.7), ("A", 3, 0.3, 0.2)])
    support = pd.DataFrame({"split": ["valid"] * 4 + ["test"] * 4, "concept": list("ABCD") * 2,
                            "true_pos": [3, 3, 3, 3, 2, 2, 2, 0]})
    res = HeldoutResult(valid=valid, test=test, support=support, counts={}, split_meta={})
    best = per_concept_best(valid, test, 0.15).set_index("concept")
    assert best.loc["A", "feature"] == 0 and best.loc["A", "test_f1"] == 0.4
    assert best.loc["B", "feature"] == 5 and best.loc["B", "test_f1"] == 0.6
    summary, pc = summarize_concept_f1(res, {"A": 10, "B": 5, "C": 200, "D": 8}, tau=0.15, size_range=(5, 100))
    assert summary["n_concepts"] == 2 and summary["concept_f1"] == pytest.approx(0.5)
    assert summary["share_test_f1_ge_0.5"] == pytest.approx(0.5)
    assert set(pc[pc["supported"] & pc["in_size_range"]]["concept"]) == {"A", "B"}


# --- the gene-level readout ---------------------------------------------------------------------


def test_readout_controls_and_probe():
    rng = np.random.default_rng(0)
    n_genes, d = 400, 20
    X = rng.standard_normal((n_genes, d)).astype(np.float32)
    Y = np.zeros((n_genes, 3), dtype=np.int8)
    Y[:, 0] = (X[:, 0] + 0.3 * rng.standard_normal(n_genes) > 0.8)      # decodable from dimension 0
    Y[:, 1] = (-X[:, 1] > 1.0)                                          # negative sign of dimension 1
    Y[:, 2] = rng.random(n_genes) < 0.1                                 # unrelated
    tv, te = np.arange(0, 300), np.arange(300, 400)
    d_best, sign, a_tv, a_te = single_dim_auroc(X[tv], X[te], Y[tv], Y[te], both_signs=True)
    assert d_best[0] == 0 and sign[0] == 1 and a_te[0] > 0.8
    assert d_best[1] == 1 and sign[1] == -1 and a_te[1] > 0.95
    assert abs(a_te[2] - 0.5) < 0.25
    probe = probe_auroc(X[tv], X[te], Y[tv], Y[te], C=1.0, scaling="standardize")
    assert probe[0] > 0.8 and probe[1] > 0.95 and abs(probe[2] - 0.5) < 0.25
    # a random representation is at chance on average; the planted one is not
    genes = [f"g{i}" for i in range(n_genes)]
    splits = make_splits(genes, n_splits=3, valid_frac=0.2, test_frac=0.1, seed=0)
    assert all(not (set(tv_) & set(te_)) and len(te_) == 40 for tv_, te_ in splits.values())
    reps = {"planted": (X, "raw"), "random": (rng.standard_normal((n_genes, d)).astype(np.float32), "random_20")}
    pc = readout(reps, Y, ["c0", "c1", "c2"], [10, 10, 10], {g: i for i, g in enumerate(genes)}, splits,
                 ReadoutParams(n_splits=3, scaling=["standardize"]))
    means = pc.groupby("representation")["probe_test_auroc_standardize"].mean()
    assert means["planted"] > 0.7 and abs(means["random"] - 0.5) < 0.15
    # center-only scaling differs from standardization only by the per-dimension scale
    Z1, _ = __import__("interpretfm.gene_readout", fromlist=["probe_prepare"]).probe_prepare(X[tv], X[te], "center")
    assert Z1.shape == (300, d)


# --- gene keys and tokens -----------------------------------------------------------------------


def test_gene_table_and_token_resolver(tmp_path):
    import anndata as ad

    var = pd.DataFrame({"gene_symbol": ["TP53", "EGFR", "KRAS"]}, index=["ENSG1", "ENSG2", "ENSG3"])
    adata = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32), var=var)
    gid = type("G", (), {"column": "index", "kind": "ensembl"})()
    genes = GeneTable.from_adata(adata, gid)
    assert genes.keys == ["ENSG1", "ENSG2", "ENSG3"] and genes.symbols == ["TP53", "EGFR", "KRAS"]
    assert genes.resolve(["EGFR", "ENSG3", "nope"]) == ["ENSG2", "ENSG3", None]
    assert genes.names_for_query("symbol") == ["TP53", "EGFR", "KRAS"]
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"TP53": 10, "EGFR": 11, "<cls>": 60695, "<pad>": 60694}))
    vocab = load_token_vocab(vocab_path)
    assert vocab["10"] == "TP53" and vocab["60695"] == "<cls>"
    r = TokenResolver(genes, vocab=vocab, special_tokens=["60694", "60695"])
    assert list(r.resolve(["10", 11, "60695", "KRAS", "ENSG2", "99"])) == ["ENSG1", "ENSG2", None, "ENSG3", "ENSG2", None]
    assert r.resolve_rows(["10", "60695"], {"ENSG1": 0}).tolist() == [0, -1]


# --- parallel concept evaluation -----------------------------------------------------------------


def _concept_eval_tables(run_dir, layer="layer_1.norm2"):
    out = {"summary": pd.read_csv(run_dir / "06_concept_eval" / "concept_f1.csv")}
    for d in sorted((run_dir / "06_concept_eval" / layer).glob("f1_*")):
        for name in ("valid_concept_f1.csv", "test_concept_f1.csv", "per_concept.csv", "support.csv"):
            out[f"{d.name}/{name}"] = pd.read_csv(d / name)
        out[f"{d.name}/counts.json"] = json.load(open(d / "counts.json"))
        out[f"{d.name}/split.json"] = json.load(open(d / "split.json"))
    return out


def test_concept_eval_n_jobs_gives_identical_results(toy_run, tmp_path):
    """``n_jobs: 2`` (separate processes) writes exactly the tables of the sequential stage, and
    ``n_jobs`` / ``threads_per_job`` do not enter the stage hash."""
    import shutil

    from interpretfm.runner import run_stage_inprocess

    cfg, run_dir = toy_run
    seq = _concept_eval_tables(run_dir)
    cfg2 = load_config(run_dir / "config.resolved.yaml")
    assert cfg2.stage_hash("concept_eval") == cfg.stage_hash("concept_eval")
    cfg2.stages["concept_eval"] = dict(cfg2.stages["concept_eval"], n_jobs=2, threads_per_job=1)
    assert cfg2.stage_hash("concept_eval") == cfg.stage_hash("concept_eval"), "n_jobs / threads_per_job must not change the hash"
    rd2 = tmp_path / "parallel"
    shutil.copytree(run_dir, rd2, symlinks=True, ignore=shutil.ignore_patterns("06_concept_eval", "07_gene_readout"))
    cfg2.run.out_dir = rd2
    run_stage_inprocess(cfg2, "concept_eval")
    par = _concept_eval_tables(rd2)
    assert set(par) == set(seq) and len(seq) > 5
    for key in seq:
        a, b = seq[key], par[key]
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            assert a == b, key
    meta = json.load(open(rd2 / "06_concept_eval" / "concept_f1.json"))
    assert meta["n_jobs"] == 2 and meta["threads_per_job"] == 1
