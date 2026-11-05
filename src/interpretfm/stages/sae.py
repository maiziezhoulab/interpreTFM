"""Stage ``sae``: SAE training per layer with the fixed lambda-selection rule, plus the untrained baseline.

For every layer and every λ of the grid an SAE is trained on the stored activations
(:func:`interpretfm.sae.train_sae`), QC'd on evenly spaced evaluation rows
(:func:`interpretfm.sae.sae_qc`) and the λ rule (:func:`interpretfm.sae.select_lambda`)
picks the checkpoint; an untrained SAE of the same width is written as the baseline.
Training needs only the stored activations, so the stage runs in the core environment
(on the GPU when ``device`` allows it).

Parameters (``stages.sae``): ``width_multiplier`` (8); ``lambda`` as a number (fixed) or
``{grid: [0.1, 0.3, 1, 3, 10], rule: {l0_max: 0.02, dead_max: 0.25, fvu_fallback: 0.2}}``;
``steps`` (8000), ``lr`` (1e-4), ``warmup`` (1000), ``resample_every`` (2000), ``batch_size``
(1024), ``seed`` (``run.seed``), ``device`` (``auto``), ``exclude_special_tokens`` (false:
the published SAEs saw every stored row), ``untrained_baseline`` (true), ``qc``
(``{max_shards: 300, rows_per_shard: 512}``), ``checkpoints`` (``{<layer>: path}`` or
``{<layer>: {<λ>: path}}``: cached checkpoints instead of training) and ``store_root``.

Outputs: ``selected.json`` (per layer: the chosen λ, the rule, the checkpoint paths and QC
of every λ and of the untrained SAE) and per layer ``<layer>/lambda_<λ>/{sae_best.pt,
sae_last.pt, history.csv, summary.json, qc.json, qc_features.csv}``,
``<layer>/untrained/{sae.pt, qc.json}`` and ``<layer>/selected.json``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from interpretfm.params import get_mapping, get_param, param_path
from interpretfm.runner import StageContext, register_stage
from interpretfm.sae import LAMBDA_GRID, SAETrainParams, sae_qc, select_lambda, train_sae, untrained_sae
from interpretfm.stores import resolve_store, sample_rows

OUTPUTS = ["selected.json"]


def lambda_tag(l1: float) -> str:
    return f"lambda_{float(l1):g}"


def _lambda_settings(params: Dict[str, Any]):
    lam = params.get("lambda", {"grid": list(LAMBDA_GRID)})
    if isinstance(lam, (int, float)) and not isinstance(lam, bool):
        return [float(lam)], None
    if not isinstance(lam, dict):
        raise ValueError("stages.sae.lambda: expected a number or {grid: [...], rule: {...}}")
    grid = lam.get("grid", list(LAMBDA_GRID))
    if not isinstance(grid, (list, tuple)) or not grid or any(isinstance(g, bool) or not isinstance(g, (int, float)) for g in grid):
        raise ValueError(f"stages.sae.lambda.grid: expected a non-empty list of numbers, got {grid!r}")
    rule = dict(lam.get("rule") or {})
    for k, default in (("l0_max", 0.02), ("dead_max", 0.25), ("fvu_fallback", 0.2)):
        v = rule.get(k, default)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise ValueError(f"stages.sae.lambda.rule.{k}: expected a number, got {v!r}")
        rule[k] = None if v is None else float(v)
    return [float(g) for g in grid], rule


def _cached_checkpoint(cfg, cached: Dict[str, Any], layer: str, l1: float) -> Optional[Path]:
    entry = cached.get(layer)
    if entry is None:
        return None
    if isinstance(entry, dict):
        for k, v in entry.items():
            if float(k) == float(l1):
                return param_path(cfg, v, "sae", f"checkpoints.{layer}.{k}")
        return None
    return param_path(cfg, entry, "sae", f"checkpoints.{layer}")


@register_stage("sae", inputs=["extract"], outputs=OUTPUTS)
def run_sae(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """SAE training per layer with the fixed lambda-selection rule, plus the untrained baseline."""
    cfg, p = ctx.config, ctx.params
    store = resolve_store(ctx)
    grid, rule = _lambda_settings(p)
    width = get_param(p, "width_multiplier", 8, "sae", int)
    seed = get_param(p, "seed", cfg.run.seed, "sae", int)
    device = get_param(p, "device", "auto", "sae", str)
    exclude = cfg.model.special_tokens if get_param(p, "exclude_special_tokens", False, "sae", bool) else ()
    qc_cfg = get_mapping(p, "qc", "sae")
    qc_shards = get_param(qc_cfg, "max_shards", 300, "sae.qc", int)
    qc_rows = get_param(qc_cfg, "rows_per_shard", 512, "sae.qc", int)
    untrained = get_param(p, "untrained_baseline", True, "sae", bool)
    cached = get_mapping(p, "checkpoints", "sae")
    train = SAETrainParams(width_multiplier=width, lr=get_param(p, "lr", 1e-4, "sae", float),
                           steps=get_param(p, "steps", 8000, "sae", int), warmup=get_param(p, "warmup", 1000, "sae", int),
                           resample_every=get_param(p, "resample_every", 2000, "sae", int),
                           batch_size=get_param(p, "batch_size", 1024, "sae", int), seed=seed, device=device,
                           exclude_tokens=exclude)
    if train.steps <= 0 or train.batch_size <= 0 or width <= 0:
        raise ValueError("stages.sae: steps, batch_size and width_multiplier must be > 0")
    qc_device = "cuda" if device in ("auto", "cuda") and train.resolved_device().type == "cuda" else "cpu"

    selected: Dict[str, Any] = {}
    for layer in cfg.model.layers:
        X = sample_rows(store, layer, max_shards=qc_shards, rows_per_shard=qc_rows, gene_rows_only=bool(exclude),
                        special_tokens=exclude)
        ctx.log.info("sae: layer %s, %d QC rows, lambda grid %s", layer, X.shape[0], grid)
        per_lambda: Dict[str, Any] = {}
        qc_by_lambda: Dict[float, Dict[str, Any]] = {}
        for l1 in grid:
            ldir = ctx.path(layer, lambda_tag(l1), "qc.json").parent
            ckpt = _cached_checkpoint(cfg, cached, layer, l1)
            if ckpt is None:
                ckpt_path = ldir / "sae_best.pt"
                if not ckpt_path.exists():
                    params_l = SAETrainParams(**{**train.__dict__, "l1": float(l1)})
                    train_sae(store, layer, params_l, ldir, label=f"{layer}:{lambda_tag(l1)}", log=ctx.log)
                source = "trained"
            else:
                ckpt_path, source = ckpt, "cached"
            qc, feats = sae_qc(ckpt_path, X, device=qc_device)
            qc.update({"lambda": float(l1), "checkpoint": str(ckpt_path), "source": source})
            ctx.write_json(f"{layer}/{lambda_tag(l1)}/qc.json", qc)
            feats.to_csv(ldir / "qc_features.csv", index=False)
            per_lambda[f"{l1:g}"] = qc
            qc_by_lambda[float(l1)] = qc
            ctx.log.info("sae: %s λ=%g (%s): fvu %.4f l0 %.3f%% dead %.3f", layer, l1, source, qc["fvu"] or float("nan"),
                         100 * qc["l0_frac"], qc["dead_frac"])
        if rule is None:
            chosen, how = grid[0], "fixed"
        else:
            chosen, how = select_lambda(qc_by_lambda, rule["l0_max"], rule["dead_max"], rule["fvu_fallback"])
        entry: Dict[str, Any] = {"layer": layer, "lambda": chosen, "rule": how, "grid": grid,
                                 "checkpoint": None if chosen is None else per_lambda[f"{chosen:g}"]["checkpoint"],
                                 "qc": per_lambda, "n_qc_rows": int(X.shape[0])}
        if untrained:
            upath = ctx.path(layer, "untrained", "sae.pt")
            if not upath.exists():
                untrained_sae(store, layer, width, seed, upath)
            uqc, ufeats = sae_qc(upath, X, device=qc_device)
            uqc.update({"checkpoint": str(upath), "source": "untrained"})
            ctx.write_json(f"{layer}/untrained/qc.json", uqc)
            ufeats.to_csv(upath.parent / "qc_features.csv", index=False)
            entry["untrained"] = uqc
        ctx.write_json(f"{layer}/selected.json", entry)
        selected[layer] = entry
        if chosen is None:
            ctx.log.warning("sae: %s: no λ satisfies the rule; downstream stages will skip this layer", layer)
        else:
            ctx.log.info("sae: %s: λ=%g (%s)", layer, chosen, how)
    ctx.write_json("selected.json", {"rule": rule, "grid": grid, "width_multiplier": width, "layers": selected})
    return {"layers": {k: {"lambda": v["lambda"], "rule": v["rule"]} for k, v in selected.items()}}
