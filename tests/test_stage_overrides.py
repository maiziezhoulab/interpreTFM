"""Stage inputs replaced by a stage parameter (``cells_h5ad``, ``spatial_only``) and markers with single-cell labels."""
import numpy as np
import pytest

from interpretfm.config import parse_config
from interpretfm.runner import StageError, _check_inputs, get_stage
from interpretfm.testing import make_toy_adata, toy_config_dict


def _cfg(tmp_path, stages):
    adata = make_toy_adata(n_cells=40, n_genes=20)
    adata.write_h5ad(tmp_path / "toy.h5ad")
    raw = toy_config_dict(tmp_path / "toy.h5ad", tmp_path / "run")
    raw["stages"].update(stages)
    return parse_config(raw, base=tmp_path)


def test_input_override_skips_missing_input(tmp_path):
    st = get_stage("niches")  # inputs ingest and cells; cells is replaced by stages.niches.cells_h5ad
    with pytest.raises(StageError, match="needs 'cells'"):
        _check_inputs(st, tmp_path / "run", {"ingest"}, _cfg(tmp_path, {}))
    _check_inputs(st, tmp_path / "run", {"ingest"}, _cfg(tmp_path, {"niches": {"cells_h5ad": {"layer_1.norm2": "x.h5ad"}}}))
    st = get_stage("report")
    with pytest.raises(StageError, match="needs 'layers'"):
        _check_inputs(st, tmp_path / "run", {"ingest"}, _cfg(tmp_path, {}))
    _check_inputs(st, tmp_path / "run", {"ingest"}, _cfg(tmp_path, {"report": {"spatial_only": True}}))


def test_builtin_overrides_declared():
    assert get_stage("niches").input_overrides == {"cells": "cells_h5ad"}
    assert get_stage("interfaces").input_overrides == {"cells": "cells_h5ad", "extract": "cells_h5ad"}
    assert get_stage("report").input_overrides == {"layers": "spatial_only"}


def test_partner_markers_skip_single_cell_label():
    from interpretfm.interfaces import partner_markers

    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(30, 12)).astype(np.float32)
    labels = ["a"] * 15 + ["b"] * 14 + ["single"]
    M = partner_markers(counts, [f"g{i}" for i in range(12)], labels, top_n=3)
    assert set(M["label"]) == {"a", "b"}
    assert (M.groupby("label").size() == 3).all()
