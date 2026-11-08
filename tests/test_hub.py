"""Hugging Face export and reload of a trained SAE (``interpretfm hub``)."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from interpretfm import hub
from interpretfm.cli import main
from interpretfm.sae import Encoder, save_checkpoint

pytest.importorskip("safetensors")


def _fake_run(root: Path, layer: str = "layer_1.norm2", model: str = "scgpt") -> Path:
    """A run directory with the files the export reads: the resolved configuration, the selected SAE and a feature table."""
    run = root / "run"
    (run / "04_sae").mkdir(parents=True)
    (run / "08_cells" / layer).mkdir(parents=True)
    torch.manual_seed(0)
    d_in, n_latents = 8, 16
    state = {"encoder.weight": torch.randn(n_latents, d_in), "encoder.bias": torch.randn(n_latents), "bias": torch.randn(d_in),
             "decoder.weight": torch.randn(d_in, n_latents)}
    ckpt = save_checkpoint(run / "04_sae" / layer / "lambda_1" / "sae.pt", state, d_in, n_latents,
                           spec={"l1": 1.0, "steps": 10}, summary={"step": 9}, input_scale=1.25)
    cfg = {"run": {"name": "cosmx_toy", "out_dir": str(run), "seed": 0},
           "dataset": {"h5ad": str(root / "private" / "data.h5ad"), "spatial": {"um_per_unit": 0.18}, "labels": {"column": "ct"}},
           "model": {"name": model, "checkpoint": str(root / "private" / "ckpt"), "layers": [layer], "representative_layer": layer},
           "stages": {"sae": {"width_multiplier": 2, "steps": 10, "lambda": {"grid": [1.0], "rule": {"l0_max": 0.02, "dead_max": 0.25}},
                              "checkpoints": {layer: {"1.0": str(ckpt)}}},
                      "concept_eval": {"concepts_csv": str(root / "private" / "concepts.csv")}}}
    with open(run / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    with open(run / "04_sae" / "selected.json", "w") as fh:
        json.dump({"layers": {layer: {"layer": layer, "lambda": 1.0, "checkpoint": str(ckpt), "grid": [1.0],
                                      "qc": {"1": {"fvu": 0.1, "l0_frac": 0.01, "dead_frac": 0.0, "checkpoint_step": 9}},
                                      "untrained": {"fvu": 3.0, "l0_frac": 0.5, "dead_frac": 0.01}}}}, fh)
    (run / "08_cells" / layer / "features.csv").write_text("concept,feature,valid_f1,test_f1,var_name\nGO:1,3,0.8,0.7,GO:1|feat3\n")
    with open(run / "08_cells" / layer / "selection.json", "w") as fh:
        json.dump({"layer": layer, "n_features": 1, "n_concepts_kept": 1, "rule": {"min_valid_f1": 0.4, "unit": "row"}}, fh)
    return run


def test_export_reload_identical_and_public(tmp_path):
    run = _fake_run(tmp_path)
    out = tmp_path / "export"
    assert main(["hub", "export", str(run), str(out), "--repo-id", "user/interpretfm-test"]) == 0
    d = out / "scgpt" / "cosmx" / "layer_1.norm2"
    assert {p.name for p in d.iterdir()} == set(hub.LAYER_FILES)
    assert (out / "README.md").exists()
    ckpt = hub.load_sae(out, "scgpt", "layer_1.norm2")
    ref = torch.load(run / "04_sae" / "layer_1.norm2" / "lambda_1" / "sae.pt", weights_only=False)
    assert set(ckpt["state_dict"]) == set(ref["state_dict"])
    for k, v in ref["state_dict"].items():
        assert torch.equal(v, ckpt["state_dict"][k])
    assert ckpt["d_in"] == 8 and ckpt["n_latents"] == 16 and ckpt["input_scale"] == 1.25
    enc = Encoder(ckpt)
    assert enc.encode(torch.zeros(2, 8)).shape == (2, 16)
    # nothing machine-specific leaves the run directory: no absolute path of the private tree in any exported text file
    for p in (d / hub.CONFIG_FILE, d / hub.METRICS_FILE, d / hub.CARD_FILE, out / "README.md"):
        assert str(tmp_path) not in p.read_text()
    assert main(["hub", "verify", str(run), str(out)]) == 0
    assert hub.load_features(out, "scgpt", "layer_1.norm2").shape[0] == 1
    assert hub.layer_dir(out, "scgpt", "layer_1.norm2", dataset="cosmx") == d
    with pytest.raises(FileNotFoundError):
        hub.layer_dir(out, "scgpt", "layer_1.norm2", dataset="xenium")


def test_load_sae_missing_layer(tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.load_sae(tmp_path, "scgpt", "layer_9")
    assert hub.DEFAULT_REPO == "YunfeiHu/interpreTFM"  # the release repository (no network access here)
