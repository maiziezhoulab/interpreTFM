"""Fresh-extraction acceptance (GPU): 2,000 CosMx cells per model, extracted in the model's own
environment through the runner's subprocess path, compared with the manuscript's cached store.

Skipped unless ``INTERPRETFM_REPRO_GPU=1`` (the extraction needs the model weights, the
model environments named by ``INTERPRETFM_SCGPT_ENV`` / ``INTERPRETFM_GENEFORMER_ENV`` and a
GPU; about a minute per model). Thresholds: the token sequence of every cell must be
identical to the cached one, every token's cosine >= 0.999, and the cell-level projection
through the cached SAE must agree with the cached store (and, for scGPT, with the ``cells``
stage of the paper run when it exists) with a cosine >= 0.999 per cell. The differences that
remain come from batch composition (padding length) and float16 rounding; the reports are
written next to the run (``02_extract/check_*.json``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repro_support import report

from interpretfm.config import ConfigError
from interpretfm.extract import compare_projections, compare_stores
from interpretfm.runner import run, stage_dir

CASES = {
    "extract_cosmx_scgpt": {
        "store": "acts_extraction_scgpt_cosmx", "layer": "layer_1.norm2", "ref_exclude": ["60694"],  # padding rows of the cached store
        "special": ["60694", "60695"], "paper_run": "cosmx_scgpt_layer2",
        "sae": "sae_v2/cosmx/scgpt/layer_1_norm2/sae/v2biased__steps_8000__lam_1/sae_layer_1.norm2_best.pt"},
    "extract_cosmx_geneformer": {
        "store": "acts_extraction_geneformer_cosmx", "layer": "layer_1", "ref_exclude": [],
        "special": ["<cls>", "<eos>"], "paper_run": "cosmx_geneformer_layer2",
        "sae": "sae_v2/cosmx/geneformer/layer_1/sae/v2biased__steps_8000__lam_1/sae_layer_1_best.pt"},
}

gpu = pytest.mark.skipif(os.environ.get("INTERPRETFM_REPRO_GPU") != "1",
                         reason="fresh extraction needs the model environments and a GPU; set INTERPRETFM_REPRO_GPU=1")


@gpu
@pytest.mark.parametrize("name", list(CASES))
def test_fresh_extraction_matches_cached_store(paper, name):
    case = CASES[name]
    try:
        cfg = paper.config(name)
    except ConfigError as e:
        pytest.skip(f"{name}: {str(e).splitlines()[0]} (set the INTERPRETFM_*_ENV / INTERPRETFM_GENEFORMER_MODEL variables)")
    assert cfg.model.env.kind != "current", "the acceptance extraction must cross the environment boundary"
    rows = run(cfg, stages=["ingest", "extract"])
    assert all(r["status"] in ("done", "cached") for r in rows), rows
    sdir = stage_dir(cfg.run.out_dir, "extract")
    layer = case["layer"]
    manifest = json.load(open(sdir / layer / "manifest.json"))
    report(f"{name} extracted", f"{manifest['n_cells']} cells, {manifest['n_rows']} rows, {manifest['n_shards']} shards, "
                               f"{manifest['dtype']}, torch {manifest['environment']['torch']['version']}, "
                               f"cuda {manifest['environment']['torch']['cuda_available']}")
    assert manifest["n_cells"] == 2000
    if manifest["cell_pooling"]["kind"] == "token":
        assert manifest["n_cells_with_pooling_token"] == 2000
    assert manifest["environment"]["torch"]["cuda_available"], "the extraction did not run on a GPU"
    cells = (sdir / "cell_subset.txt").read_text().split()
    cached = Path(os.environ["INTERPRETFM_STORES"]) / case["store"]
    if not (cached / "activations" / layer).is_dir():
        pytest.skip(f"cached store {cached} has no layer {layer}")

    rows_cmp = compare_stores(sdir / "store", cached, layer, cells, ref_exclude_tokens=case["ref_exclude"])
    json.dump(rows_cmp, open(sdir / "check_rows_vs_cached.json", "w"), indent=1)
    report(f"{name} token sequences identical", rows_cmp["n_cells_token_sequence_identical"], manuscript=len(cells))
    report(f"{name} per-token cosine min / median", (rows_cmp["cosine"]["min"], rows_cmp["cosine"]["median"]), tol=">= 0.999")
    report(f"{name} max abs diff / relative to the cached scale", (rows_cmp["max_abs_diff"], rows_cmp["max_abs_diff_over_ref_scale"]))
    assert rows_cmp["token_sequence_identical"], rows_cmp["token_mismatch_examples"]
    assert rows_cmp["n_rows_compared"] > 0 and rows_cmp["n_rows_cosine_below_0.999"] == 0
    assert rows_cmp["cosine"]["min"] >= 0.999

    ckpt = Path(os.environ["INTERPRETFM_PAPER_CACHE"]) / case["sae"]
    if not ckpt.is_file():
        pytest.skip(f"cached SAE checkpoint {ckpt} not found")
    paper_cells = Path(os.environ["INTERPRETFM_RUNS"]) / case["paper_run"] / "08_cells" / layer / "cells.h5ad"
    proj = compare_projections(sdir / "store", cached, layer, ckpt, cells, special_tokens=case["special"],
                               ref_exclude_tokens=case["ref_exclude"], cells_h5ad=paper_cells if paper_cells.is_file() else None)
    json.dump(proj, open(sdir / "check_projection_vs_cached.json", "w"), indent=1)
    s = proj["store_vs_store"]
    report(f"{name} projected features ({proj['n_features']}): per-cell cosine min, max abs diff, active-set Jaccard",
           (s["cosine"]["min"], s["max_abs_diff"], s["active_set_jaccard_mean"]), tol=">= 0.999")
    assert s["n"] == len(cells) and s["cosine"]["min"] >= 0.999 and s["active_set_jaccard_mean"] >= 0.99
    if "cells_h5ad" in proj:
        h = proj["cells_h5ad"]["new_store_vs_h5ad"]
        report(f"{name} vs the paper run's cells.h5ad ({proj['cells_h5ad']['n_features']} retained features): cosine min, max abs diff",
               (h["cosine"]["min"], h["max_abs_diff"]), tol=">= 0.999")
        assert h["cosine"]["min"] >= 0.999
    else:
        report(f"{name} vs the paper run's cells.h5ad", "not available (cells stage not finished)")
