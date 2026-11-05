"""T6-T13 (AUDIT §4): cell-level and spatial numbers. Placeholders until the ``cells`` stage
(feature retention, GSI/MHR) and the ``spatial`` stage (niches, interfaces, LR screen) land;
the expected values are pinned in ``expected.py`` so that the tests only need the readers.

TODO (after the cells stage merges):
- T6: features kept by the spatial feature rule (validation F1 >= 0.40 at τ 0.15) = Table S1 "Features kept";
  cached source ``SPS/tables/stageB_summary.csv`` (``n_features``; 925 concepts kept at >= 0.40).
- T7: GSI excess and MHR@100 of sparse features, raw dimensions and single genes (same 40,000 cells,
  populations >= 30 cells, 5 label shuffles); cached source
  ``revision_metrics/gsi_mhr_full/gsi_mhr_<ds>_<model>_<layer>_v2biased__steps_8000__lam_1.json``.

TODO (after the spatial stages merge): T8 niche sweep winner and validation, T9 expression-only and
label-composition baselines, T10 calibrated interface statistic (n_perm 100, or a 20-permutation
smoke variant comparing counts), T11 LR co-localization screen, T12 matched-neighbour radius,
T13 niche concept labels; values in ``expected.SPATIAL_COSMX_SCGPT``.
"""
from __future__ import annotations

import pytest

from expected import EXPECTED, PRIMARY, SPATIAL_COSMX_SCGPT

CELLS_TODO = "TODO: needs the cells stage (P1-C); expected values are in expected.py"
SPATIAL_TODO = "TODO: needs the spatial stages (P2); expected values are in expected.SPATIAL_COSMX_SCGPT"


@pytest.mark.skip(reason=CELLS_TODO)
@pytest.mark.parametrize("name", list(EXPECTED))
def test_t6_features_kept(paper, name):
    exp = EXPECTED[name]
    assert exp["features_kept"] > 0  # pragma: no cover - placeholder


@pytest.mark.skip(reason=CELLS_TODO)
@pytest.mark.parametrize("name", list(EXPECTED))
def test_t7_gsi_excess_and_mhr(paper, name):
    exp = EXPECTED[name]
    assert set(exp["gsi_excess"]) == {"raw", "sae", "genes"} and set(exp["mhr100"]) == {"raw", "sae", "genes", "chance"}


@pytest.mark.skip(reason=SPATIAL_TODO)
@pytest.mark.parametrize("key", ["niche", "expression_baseline", "label_composition", "interfaces", "lr_screen",
                                 "matched_radius", "niche_concepts"])
def test_t8_to_t13_spatial(paper, key):
    assert key in SPATIAL_COSMX_SCGPT and PRIMARY in EXPECTED
