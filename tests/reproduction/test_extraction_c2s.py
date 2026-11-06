"""Fresh-extraction acceptance (GPU) for C2S-Scale: 2,000 CosMx cells extracted in the C2S
environment through the runner's subprocess path, compared with the manuscript's cached store
(``acts_extraction_c2s_cosmx_store``, ``layer_13``) and the cached SAE, with the same checks
and thresholds as ``test_extraction.py`` (identical token sequences, per-token cosine >= 0.999,
per-cell projected cosine >= 0.999).

Skipped unless ``INTERPRETFM_REPRO_GPU=1``; needs ``INTERPRETFM_C2S_ENV`` and
``INTERPRETFM_C2S_MODEL`` (a few minutes: the Gemma-2 2B model loads in float32).
The rows differ from the cached ones only by float16 rounding of a float32 forward pass whose
batch composition (left padding) differs; cells re-extracted in their original batches are
bit-identical.
"""
from __future__ import annotations

from test_extraction import CASES, gpu
from test_extraction import test_fresh_extraction_matches_cached_store as check_case

CASES["extract_cosmx_c2s"] = {
    "store": "acts_extraction_c2s_cosmx_store", "layer": "layer_13", "ref_exclude": [], "special": [],
    "paper_run": "cosmx_c2s_layer14",
    "sae": "sae_v2/cosmx/c2sscale/layer_13/sae/v2biased__steps_8000__lam_1/sae_layer_13_best.pt"}


@gpu
def test_fresh_c2s_extraction_matches_cached_store(paper):
    check_case(paper, "extract_cosmx_c2s")
