"""Export of trained SAEs to a Hugging Face-style folder and loading from it (``interpretfm hub``).

One exported layer is a folder ``<out_dir>/<model>/<dataset>/<layer>/`` (the dataset is the
prefix of the run name, ``cosmx`` or ``xenium`` for the manuscript's runs) with

- ``sae.safetensors``: the checkpoint's tensors (``encoder.weight``, ``encoder.bias``, ``bias``,
  ``decoder.weight``), float32 as trained;
- ``config.json``: ``d_in``, ``n_latents``, ``architecture``, ``input_scale``, the training
  specification, the λ rule, the model, layer and dataset, and the package version;
- ``features.csv``: the retained features with their concepts (``08_cells/<layer>/features.csv``);
- ``metrics.json``: SAE quality control with the untrained control, held-out concept F1 of SAE
  features beside raw dimensions and the untrained SAE on both held-out units, the gene-level
  readout, cell specificity against raw dimensions and single genes, the layer-table row;
- ``README.md``: the model card.

The folder is what ``huggingface-cli upload <repo-id> <out_dir>`` publishes; nothing here uploads.
:func:`load_sae` reads a layer back from a local folder or, with ``huggingface_hub`` installed,
from the repository, as a checkpoint dict that :class:`interpretfm.sae.Encoder` accepts.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from interp_pipeline import __version__

#: Hugging Face repository of the release (trained SAEs and the tutorial files); versions are tags (``v1.0.0``).
DEFAULT_REPO = "YunfeiHu/interpreTFM"

SAE_FILE = "sae.safetensors"
CONFIG_FILE = "config.json"
FEATURES_FILE = "features.csv"
METRICS_FILE = "metrics.json"
CARD_FILE = "README.md"
LAYER_FILES = (SAE_FILE, CONFIG_FILE, FEATURES_FILE, METRICS_FILE, CARD_FILE)

#: Public sources of the foundation models whose activations the SAEs decompose.
MODEL_SOURCES: Dict[str, Dict[str, str]] = {
    "scgpt": {"display": "scGPT", "source": "https://github.com/bowang-lab/scGPT (whole-human checkpoint)",
              "license": "MIT"},
    "geneformer": {"display": "Geneformer", "source": "https://huggingface.co/ctheodoris/Geneformer (V2, 104M parameters)",
                   "license": "Apache-2.0"},
    "c2s_scale": {"display": "C2S-Scale", "source": "https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-2B",
                  "license": "CC BY 4.0 for the C2S-Scale weights; built on Gemma 2, subject to the Gemma Terms of Use"},
}

#: Datasets of the manuscript, keyed by the prefix of the run name.
DATASETS: Dict[str, str] = {
    "cosmx": "CosMx human non-small-cell lung cancer section (81,236 cells, 960-gene panel; public NanoString data)",
    "xenium": "Xenium human renal cell carcinoma section (465,220 cells, 399-gene panel)",
}


def layer_folder(model: str, layer: str, dataset: str) -> str:
    """Relative folder of one exported layer inside the export (``<model>/<dataset>/<layer>``)."""
    return f"{model}/{dataset}/{layer}"


def dataset_key(run_name: str) -> str:
    """The dataset folder of a run: the known dataset prefix of the run name, else its first ``_``-separated part."""
    low = run_name.lower()
    return next((k for k in DATASETS if low.startswith(k)), low.split("_")[0] or low)


# --- reading a run directory -----------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _stage_dirs(run_dir: Path) -> Dict[str, Path]:
    from interpretfm.runner import list_stages

    return {s.name: run_dir / s.dir_name for s in list_stages()}


def _layer_table_row(layers_dir: Path, layer: str) -> Optional[Dict[str, Any]]:
    path = layers_dir / "layer_table.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("layer") == layer:
                return {k: _number(v) for k, v in row.items()}
    return None


def _number(v: str) -> Any:
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def _pick(d: Optional[Dict[str, Any]], *keys: str) -> Dict[str, Any]:
    return {k: d[k] for k in keys if d and k in d}


def _qc_at(qc: Dict[str, Any], lam: Any) -> Dict[str, Any]:
    """The QC entry of λ, whose key may be written ``"1"`` or ``"1.0"``."""
    if lam is None:
        return {}
    for key in (str(lam), str(float(lam)), str(int(lam)) if float(lam).is_integer() else None):
        if key is not None and key in qc:
            return qc[key]
    return {}


def _public(obj: Any) -> Any:
    """Copy of a parameter tree with every absolute path reduced to its file name (nothing machine-specific is exported)."""
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/") and len(obj) > 1:
        return Path(obj).name
    return obj


def collect_layer(run_dir: Union[str, Path], layer: str) -> Dict[str, Any]:
    """Everything the export needs for one layer of a finished run (checkpoint path, configuration, metrics)."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.resolved.yaml") as fh:
        cfg = yaml.safe_load(fh)
    dirs = _stage_dirs(run_dir)
    model = cfg["model"]["name"]
    if layer not in cfg["model"]["layers"]:
        raise KeyError(f"{run_dir}: layer {layer!r} is not in model.layers {cfg['model']['layers']}")
    sae_sel = (_read_json(dirs["sae"] / "selected.json") or {}).get("layers", {}).get(layer)
    if not sae_sel or not sae_sel.get("checkpoint"):
        raise FileNotFoundError(f"{run_dir}: no selected SAE for {layer} in {dirs['sae'] / 'selected.json'}")
    ckpt_path = Path(sae_sel["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{run_dir}: SAE checkpoint {ckpt_path} missing")
    lam = sae_sel.get("lambda")
    qc = _qc_at(sae_sel.get("qc") or {}, lam)
    f1 = (_read_json(dirs["concept_eval"] / "concept_f1.json") or {})
    evals = f1.get("layers", {}).get(layer, {}).get("evaluations", {})
    cells_sel = _read_json(dirs["cells"] / layer / "selection.json") or {}
    spec = _read_json(dirs["cells"] / layer / "specificity.json") or {}
    readout = (_read_json(dirs["gene_readout"] / "gene_readout.json") or {}).get("representations") or {}
    readout = {k.split("|")[0]: _pick(v, "mean_probe_test_auroc_standardize_mean", "mean_probe_test_auroc_standardize_std",
                                      "mean_probe_test_auroc_center_mean", "mean_single_test_auroc_mean", "n_splits")
               for k, v in readout.items() if k.endswith("|-") or k.endswith(f"|{layer}")}
    tis = (_read_json(dirs["tis"] / "tis.json") or {}).get("layers", {}).get(layer, {})
    row = _layer_table_row(dirs["layers"], layer)
    features_csv = dirs["cells"] / layer / "features.csv"
    sae_params = (cfg.get("stages") or {}).get("sae") or {}
    sae_params = _public({k: v for k, v in sae_params.items() if k not in ("checkpoints", "device")})
    run_name = cfg["run"]["name"]
    ds_key = dataset_key(run_name)
    metrics = {
        "sae_qc": {"lambda": lam, "rule": sae_sel.get("rule"), "grid": sae_sel.get("grid"),
                   **_pick(qc, "n_rows", "fvu", "l0_mean", "l0_frac", "dead_frac", "checkpoint_step"),
                   "untrained": _pick(sae_sel.get("untrained") or {}, "fvu", "l0_mean", "l0_frac", "dead_frac")},
        "concept_f1": {k: _pick(v, "concept_f1", "n_concepts", "n_concepts_scored", "share_test_f1_ge_0.5", "unit", "representation")
                       for k, v in evals.items()},
        "concept_f1_summary": f1.get("summary"),
        "features": _pick(cells_sel, "rule", "n_eligible_concepts", "n_concepts_kept", "n_features", "enough_features",
                          "valid_f1_mean", "test_f1_mean", "test_f1_median"),
        "specificity": {name: _pick(e, "gsi", "mhr") for name, e in (spec.get("representations") or {}).items()},
        "specificity_params": _pick(spec, "n_cells_sampled", "shards_read", "n_shards"),
        "tis": {**_pick(tis, "n_cells", "hidden_dim", "gain"), **_pick(row, "tis_mean", "tis_shuffled_mean", "tis_pca_mean")},
        "readout": readout or None,
        "layer_table": row,
    }
    return {
        "run_name": run_name, "model": model, "layer": layer, "representative_layer": cfg["model"].get("representative_layer"),
        "checkpoint": ckpt_path, "features_csv": features_csv if features_csv.exists() else None,
        "dataset": {"name": DATASETS.get(ds_key, run_name), "key": ds_key, "file": Path(cfg["dataset"]["h5ad"]).name,
                    "um_per_unit": (cfg["dataset"].get("spatial") or {}).get("um_per_unit"),
                    "labels": (cfg["dataset"].get("labels") or {}).get("column")},
        "model_checkpoint": Path(cfg["model"]["checkpoint"]).name if cfg["model"].get("checkpoint") else None,
        "sae_params": sae_params, "concept_eval_params": _public((cfg.get("stages") or {}).get("concept_eval") or {}),
        "cells_params": _public((cfg.get("stages") or {}).get("cells") or {}),
        "metrics": metrics,
    }


# --- export ------------------------------------------------------------------------------------


def _tensor_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.detach().cpu().contiguous() for k, v in ckpt["state_dict"].items()}


def export_layer(run_dir: Union[str, Path], layer: str, out_dir: Union[str, Path], repo_id: str = DEFAULT_REPO) -> Path:
    """Write the hub folder of one layer of ``run_dir`` under ``out_dir``; returns the folder."""
    from safetensors.torch import save_file

    from interpretfm.sae import load_checkpoint

    info = collect_layer(run_dir, layer)
    ckpt = load_checkpoint(info["checkpoint"])
    dest = Path(out_dir) / layer_folder(info["model"], layer, info["dataset"]["key"])
    dest.mkdir(parents=True, exist_ok=True)
    state = _tensor_state(ckpt)
    save_file(state, str(dest / SAE_FILE), metadata={"format": "pt", "interpretfm": __version__, "model": info["model"],
                                                     "layer": layer, "architecture": str(ckpt.get("architecture"))})
    config = {
        "interpretfm_version": __version__,
        "model": info["model"], "layer": layer, "representative_layer": info["representative_layer"],
        "model_checkpoint": info["model_checkpoint"], "dataset": info["dataset"], "run_name": info["run_name"],
        "d_in": int(ckpt["d_in"]), "n_latents": int(ckpt["n_latents"]), "architecture": ckpt.get("architecture"),
        "input_scale": ckpt.get("input_scale"), "spec": ckpt.get("spec"), "summary": ckpt.get("summary"),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in state.items()},
        "training": info["sae_params"], "lambda_selected": info["metrics"]["sae_qc"].get("lambda"),
        "concept_eval": info["concept_eval_params"], "cells": info["cells_params"],
        "repo_id": repo_id, "files": list(LAYER_FILES),
    }
    with open(dest / CONFIG_FILE, "w") as fh:
        json.dump(config, fh, indent=2, default=str)
    with open(dest / METRICS_FILE, "w") as fh:
        json.dump(info["metrics"], fh, indent=2, default=str)
    if info["features_csv"] is not None:
        shutil.copyfile(info["features_csv"], dest / FEATURES_FILE)
    (dest / CARD_FILE).write_text(model_card(info, config), encoding="utf-8")
    return dest


def export_run(run_dir: Union[str, Path], out_dir: Union[str, Path], layers: Optional[Sequence[str]] = None,
               repo_id: str = DEFAULT_REPO) -> List[Path]:
    """Export every selected layer of a run (default: ``model.layers``); returns the folders written."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.resolved.yaml") as fh:
        cfg = yaml.safe_load(fh)
    out = []
    for layer in (layers or cfg["model"]["layers"]):
        out.append(export_layer(run_dir, layer, out_dir, repo_id=repo_id))
    return out


def write_index(out_dir: Union[str, Path], repo_id: str = DEFAULT_REPO) -> Path:
    """Top-level ``README.md`` of the export: one row per exported layer."""
    out_dir = Path(out_dir)
    rows = []
    for cfg_path in sorted(out_dir.glob(f"*/*/*/{CONFIG_FILE}")):
        with open(cfg_path) as fh:
            c = json.load(fh)
        m = _read_json(cfg_path.parent / METRICS_FILE) or {}
        f1 = m.get("concept_f1", {})
        rows.append((c["model"], c["dataset"]["key"], c["layer"], c["dataset"]["name"].split(" (")[0], c["d_in"], c["n_latents"],
                     _fmt(m.get("sae_qc", {}).get("fvu")), _fmt(f1.get("row_sae", {}).get("concept_f1")),
                     _fmt(f1.get("row_raw", {}).get("concept_f1")), m.get("features", {}).get("n_features") or "–"))
    lines = ["---", "license: gpl-3.0", "library_name: interpretfm", "tags:", "  - sparse-autoencoder", "  - single-cell",
             "  - spatial-transcriptomics", "  - interpretability", "---", "", "# interpreTFM: trained sparse autoencoders",
             "",
             "Sparse autoencoders (SAEs) trained on layer activations of transcriptomic foundation models by interpreTFM "
             f"({__version__}), one folder per model, dataset and layer (`<model>/<dataset>/<layer>/`). Each folder holds the weights "
             "(`sae.safetensors`), `config.json`, the retained feature–concept table (`features.csv`), `metrics.json` and a "
             "model card. Load a layer with `interpretfm.hub.load_sae`:", "", "```python",
             "from interpretfm.hub import load_sae", "from interpretfm.sae import Encoder", "",
             f'encoder = Encoder(load_sae("{repo_id}", model="{rows[0][0] if rows else "scgpt"}", '
             f'layer="{rows[0][2] if rows else "layer_1.norm2"}", dataset="{rows[0][1] if rows else "cosmx"}"))',
             "```", "", "| Model | Dataset | Layer | Section | d_in | Features | FVU | Concept F1 (row split) | Raw dimensions | Retained features |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", "Licence: GPL-3.0-or-later (the licence of interpreTFM). The base models and datasets keep their own licences; "
              "see each model card.", ""]
    path = out_dir / CARD_FILE
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def model_card(info: Dict[str, Any], config: Dict[str, Any]) -> str:
    """The model card of one layer (Hugging Face front matter plus Markdown)."""
    m = info["metrics"]
    src = MODEL_SOURCES.get(info["model"], {"display": info["model"], "source": "–", "license": "see the model's source"})
    qc = m["sae_qc"]
    f1 = m["concept_f1"]
    feats = m["features"]
    spec = m["specificity"]
    tr = config.get("training") or {}
    rule = (tr.get("lambda") or {}).get("rule") or {}
    ds = info["dataset"]
    lines = [
        "---", "license: gpl-3.0", "library_name: interpretfm", "tags:", "  - sparse-autoencoder", "  - single-cell",
        "  - spatial-transcriptomics", "  - interpretability", f"  - {info['model']}", "---", "",
        f"# interpreTFM SAE: {src['display']} `{info['layer']}` on {ds['name'].split(' (')[0]}", "",
        f"A sparse autoencoder trained by interpreTFM {__version__} on the `{info['layer']}` hidden states of "
        f"{src['display']} over the cells of the {ds['name']}. It is the SAE of this layer used for the manuscript's analyses.", "",
        "## Source model and data", "",
        f"- **Model:** {src['display']}, {src['source']}" + (f"; checkpoint `{info['model_checkpoint']}`" if info["model_checkpoint"] else "") + ".",
        f"- **Layer:** `{info['layer']}` (stored layer name; display index as in the manuscript).",
        f"- **Dataset:** {ds['name']}; file `{ds['file']}`" + (f"; labels `{ds['labels']}`" if ds.get("labels") else "") + ".",
        f"- **Activations:** one row per gene token of every cell (the activation store of the `extract` stage); "
        f"`input_scale` {_fmt(config.get('input_scale'), 4)} rescales rows to unit mean norm before encoding.", "",
        "## Training", "",
        f"- Architecture `{config.get('architecture')}`: ReLU encoder with a pre-encoder bias, tied input bias, linear decoder; "
        f"`d_in` {config['d_in']}, `n_latents` {config['n_latents']} (width multiplier {tr.get('width_multiplier', '–')}).",
        f"- Steps {tr.get('steps', '–')}, learning rate {tr.get('lr', '–')}, warm-up {tr.get('warmup', '–')}, dead-feature resampling every "
        f"{tr.get('resample_every', '–')} steps, batch size {tr.get('batch_size', '–')}, seed {tr.get('seed', '–')}; "
        f"best checkpoint at step {qc.get('checkpoint_step', '–')}.",
        f"- λ rule: ℓ1 weight chosen from the grid {qc.get('grid')} as the smallest FVU among checkpoints with L0 fraction ≤ "
        f"{rule.get('l0_max', '–')} and dead-feature fraction ≤ {rule.get('dead_max', '–')}; selected λ = {qc.get('lambda')}.", "",
        "## Metrics (with baselines)", "",
        "| Quantity | SAE | Raw hidden dimensions | Untrained SAE |", "|---|---|---|---|",
        f"| FVU | {_fmt(qc.get('fvu'))} | – | {_fmt(qc.get('untrained', {}).get('fvu'))} |",
        f"| L0 fraction | {_fmt(qc.get('l0_frac'), 4)} | – | {_fmt(qc.get('untrained', {}).get('l0_frac'), 4)} |",
        f"| Dead-feature fraction | {_fmt(qc.get('dead_frac'), 4)} | – | {_fmt(qc.get('untrained', {}).get('dead_frac'), 4)} |",
        f"| Concept F1, held-out gene-token rows ({f1.get('row_sae', {}).get('n_concepts', '–')} concepts) | "
        f"{_fmt(f1.get('row_sae', {}).get('concept_f1'))} | {_fmt(f1.get('row_raw', {}).get('concept_f1'))} | {_fmt(f1.get('row_untrained', {}).get('concept_f1'))} |",
        f"| Concept F1, held-out whole genes ({f1.get('gene_sae', {}).get('n_concepts', '–')} concepts) | "
        f"{_fmt(f1.get('gene_sae', {}).get('concept_f1'))} | {_fmt(f1.get('gene_raw', {}).get('concept_f1'))} | {_fmt(f1.get('gene_untrained', {}).get('concept_f1'))} |",
    ]
    if spec:
        g = {k: v.get("gsi", {}).get("excess_mean") for k, v in spec.items()}
        h = {k: v.get("mhr", {}).get("mhr") for k, v in spec.items()}
        lines += [f"| GSI excess (cell populations) | {_fmt(g.get('sae'))} | {_fmt(g.get('raw'))} | single genes {_fmt(g.get('genes'))} |",
                  f"| MHR@100 | {_fmt(h.get('sae'))} | {_fmt(h.get('raw'))} | single genes {_fmt(h.get('genes'))} |"]
    lines += [
        "",
        f"Retained features: {feats.get('n_features', '–')} features for {feats.get('n_concepts_kept', '–')} concepts "
        f"(rule: validation F1 ≥ {(feats.get('rule') or {}).get('min_valid_f1', '–')} on the {(feats.get('rule') or {}).get('unit', '–')} split, "
        f"concept size {(feats.get('rule') or {}).get('concept_size', '–')} panel genes); mean test F1 {_fmt(feats.get('test_f1_mean'))}. "
        "`features.csv` lists them with their concept, validation and test F1. `metrics.json` holds every number above with "
        "the full layer-table row.", "",
        "## Intended use", "",
        "- Decompose the same layer of the same model on single-cell or targeted spatial data: encode gene-token activations "
        "(`interpretfm.sae.Encoder`), project the retained features to cells and run the cell-level and spatial stages of the "
        "pipeline (`interpretfm run`).",
        "- Compare feature-level results with their baselines (raw dimensions, untrained SAE, single genes), as in the table above.", "",
        "## Limits", "",
        f"- Trained on one section of one platform ({ds['name'].split(' (')[0]}); concepts are panel-observable (the measured genes), not genome-wide.",
        "- Concept F1 on held-out whole genes is low for every representation; the features recognise known member genes in new cells "
        "better than they generalise to unseen genes.",
        "- The SAE is specific to the stored layer name and to the model checkpoint named above; activations from another checkpoint, "
        "tokenisation or pooling are out of distribution.",
        "- Metrics are from one section; replicate across sections or donors before generalising.", "",
        "## Licence", "",
        f"GPL-3.0-or-later (the SAE weights, tables and this card; the licence of interpreTFM). The base model keeps its own licence "
        f"({src['license']}); check it and the dataset's terms before use.", "",
    ]
    if info["model"] == "c2s_scale":
        lines += [
            "The SAE reads the hidden states of C2S-Scale-Gemma-2-2B by the van Dijk lab "
            "(<https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-2B>), whose weights are licensed under CC BY 4.0 "
            "(<https://creativecommons.org/licenses/by/4.0/>); the model itself was not changed and is not redistributed here.", "",
            "C2S-Scale-Gemma-2-2B is built on Gemma 2. Gemma is provided under and subject to the Gemma Terms of Use found at "
            "<https://ai.google.dev/gemma/terms>.", "",
        ]
    lines += [
        "## Citation", "",
        "Cite the interpreTFM manuscript (in preparation) and the software release (`CITATION.cff` of the repository, "
        "<https://github.com/OliiverHu/interpreTFM>).", "",
        "## Loading", "", "```python", "from interpretfm.hub import load_sae", "from interpretfm.sae import Encoder", "",
        f'ckpt = load_sae("{config.get("repo_id", DEFAULT_REPO)}", model="{info["model"]}", layer="{info["layer"]}", '
        f'dataset="{ds["key"]}")  # or a local export folder',
        "encoder = Encoder(ckpt)          # encoder.encode(x) -> features", "```", "",
    ]
    return "\n".join(lines)


# --- loading -----------------------------------------------------------------------------------


def _find_layer(root: Path, model: str, layer: str, dataset: Optional[str]) -> Path:
    if dataset is not None:
        local = root / layer_folder(model, layer, dataset)
        if not local.is_dir():
            raise FileNotFoundError(f"{root}: no exported layer at {local}")
        return local
    found = sorted(p for p in root.glob(f"{model}/*/{layer}") if p.is_dir())
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"{root}: no exported layer {model}/*/{layer}")
    raise ValueError(f"{root}: {model}/{layer} exists for several datasets ({', '.join(p.parent.name for p in found)}); pass dataset=...")


def layer_dir(repo_or_path: Union[str, Path], model: str, layer: str, dataset: Optional[str] = None,
              revision: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Local folder of one exported layer: ``<repo_or_path>/<model>/<dataset>/<layer>`` when ``repo_or_path`` is a
    folder, otherwise the layer's files of the Hugging Face repository ``repo_or_path`` (downloaded with
    ``huggingface_hub``). ``dataset`` may be left out when the model's layer was exported for one dataset only."""
    root = Path(repo_or_path)
    if root.exists():
        return _find_layer(root, model, layer, dataset)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError("loading from the Hugging Face Hub needs the huggingface_hub package (pip install huggingface_hub)") from e
    pattern = layer_folder(model, layer, dataset or "*")
    snap = snapshot_download(repo_id=str(repo_or_path), revision=revision, cache_dir=cache_dir, allow_patterns=[f"{pattern}/*"])
    return _find_layer(Path(snap), model, layer, dataset)


def load_sae(repo_or_path: Union[str, Path], model: str, layer: str, dataset: Optional[str] = None,
             revision: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load one exported SAE as a checkpoint dict (``state_dict``, ``d_in``, ``n_latents``, ``spec``,
    ``input_scale``, ``architecture``, ``summary``) that :class:`interpretfm.sae.Encoder` accepts."""
    from safetensors.torch import load_file

    d = layer_dir(repo_or_path, model, layer, dataset=dataset, revision=revision, cache_dir=cache_dir)
    with open(d / CONFIG_FILE) as fh:
        config = json.load(fh)
    state = load_file(str(d / SAE_FILE))
    return {"state_dict": state, "d_in": int(config["d_in"]), "n_latents": int(config["n_latents"]), "spec": config.get("spec"),
            "input_scale": config.get("input_scale"), "architecture": config.get("architecture"), "summary": config.get("summary"),
            "hub": {"model": config["model"], "layer": config["layer"], "dataset": config["dataset"].get("key"), "dir": str(d)}}


def load_features(repo_or_path: Union[str, Path], model: str, layer: str, dataset: Optional[str] = None, **kw):
    """The retained feature–concept table of one exported layer (a pandas DataFrame)."""
    import pandas as pd

    return pd.read_csv(layer_dir(repo_or_path, model, layer, dataset=dataset, **kw) / FEATURES_FILE)


TUTORIAL_FILES = ("tutorial/*", "scgpt/cosmx/layer_1.norm2/*")


def tutorial_bundle(source: str = "local", path: Optional[Union[str, Path]] = None, repo_id: str = DEFAULT_REPO,
                    revision: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Root folder of the tutorial data (``tutorial/`` with the CosMx crop, its projected features, a store slice
    and the label groups, beside the exported SAE ``scgpt/cosmx/layer_1.norm2/``).

    ``source="local"`` uses the folder ``path`` (a downloaded or self-built bundle); ``source="hub"`` downloads the
    tutorial files of the Hugging Face repository ``repo_id`` with ``huggingface_hub``.
    """
    if source == "local":
        if path is None:
            raise ValueError("tutorial_bundle(source='local') needs path=<bundle folder>")
        root = Path(path)
        if not (root / "tutorial" / "cosmx_crop.h5ad").exists():
            raise FileNotFoundError(f"{root}: not a tutorial bundle (tutorial/cosmx_crop.h5ad missing)")
        return root
    if source != "hub":
        raise ValueError("source must be 'local' or 'hub'")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=list(TUTORIAL_FILES)))


def verify_export(out_dir: Union[str, Path], run_dirs: Sequence[Union[str, Path]]) -> List[Dict[str, Any]]:
    """Reload every exported layer of ``run_dirs`` and compare it tensor by tensor with the run's checkpoint."""
    import torch

    from interpretfm.sae import load_checkpoint

    out = []
    for rd in run_dirs:
        with open(Path(rd) / "config.resolved.yaml") as fh:
            cfg = yaml.safe_load(fh)
        for layer in cfg["model"]["layers"]:
            info = collect_layer(rd, layer)
            ref = load_checkpoint(info["checkpoint"])
            got = load_sae(out_dir, info["model"], layer, dataset=info["dataset"]["key"])
            same = set(ref["state_dict"]) == set(got["state_dict"]) and all(
                torch.equal(ref["state_dict"][k].cpu(), got["state_dict"][k]) for k in ref["state_dict"])
            meta = int(ref["d_in"]) == got["d_in"] and int(ref["n_latents"]) == got["n_latents"] and ref.get("input_scale") == got["input_scale"]
            out.append({"run": str(rd), "model": info["model"], "dataset": info["dataset"]["key"], "layer": layer, "identical": bool(same and meta),
                        "tensors": sorted(ref["state_dict"])})
    return out
