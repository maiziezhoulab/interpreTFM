"""Sparse autoencoders on stored activations: training, the untrained baseline, QC and the λ rule.

The SAE is a ReLU autoencoder with an encoder bias and a bias shared by the input and the
output (``z = relu(W_enc (x - b) + b_enc)``, ``x_hat = W_dec z + b``), a dictionary of
``width_multiplier`` times the hidden dimension, trained on activations rescaled to mean
norm ``sqrt(h)`` with squared-error reconstruction plus ``λ * ||z||_1``, unit-norm decoder
columns (constrained Adam), warm-up, dead-feature resampling and the lowest-training-loss
checkpoint. Checkpoints are written in the original activation scale with the keys
``encoder.weight``, ``encoder.bias``, ``bias`` and ``decoder.weight``.

The λ rule (:func:`select_lambda`): among the weights whose SAE has at most ``l0_max`` of
its features active per token and a dead-feature fraction of at most ``dead_max``, the one
with the lowest fraction of variance unexplained.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from interpretfm.stores import iter_shards, list_shards, torch_load

LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
ARCHITECTURE = "relu_sae_with_biases_raw_space"


@dataclass
class SAETrainParams:
    """Training schedule of one SAE (``stages.sae``)."""

    width_multiplier: int = 8
    l1: float = 1.0
    lr: float = 1e-4
    steps: int = 8000
    warmup: int = 1000
    resample_every: Optional[int] = 2000
    batch_size: int = 1024
    seed: int = 0
    device: str = "auto"
    exclude_tokens: Sequence[str] = ()

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class BiasedReLUSAE(nn.Module):
    """``z = relu(W_enc (x - b) + b_enc)``, ``x_hat = W_dec z + b``; decoder columns unit norm at init."""

    def __init__(self, d_in: int, n_latents: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            w = torch.randn(d_in, n_latents)
            w /= w.norm(dim=0, keepdim=True)
            self.decoder.weight.copy_(w)
            self.encoder.weight.copy_(w.T)
            self.encoder.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.encoder(x - self.bias))
        return self.decoder(z) + self.bias, z


class ConstrainedAdam(torch.optim.Adam):
    """Adam that removes the gradient component along each constrained column and renormalizes it."""

    def __init__(self, params, constrained_params, lr: float):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    @torch.no_grad()
    def step(self, closure=None):
        for p in self.constrained_params:
            if p.grad is None:
                continue
            normed = p / p.norm(dim=0, keepdim=True).clamp_min(1e-12)
            p.grad -= (p.grad * normed).sum(dim=0, keepdim=True) * normed
        super().step(closure=closure)
        for p in self.constrained_params:
            p /= p.norm(dim=0, keepdim=True).clamp_min(1e-12)


def warmup_lr_multiplier(step: int, warmup_steps: int, resample_steps: Optional[int]) -> float:
    """Linear warm-up, restarted after every resampling period."""
    if warmup_steps <= 0:
        return 1.0
    if resample_steps is None or resample_steps <= 0:
        return min((step + 1) / warmup_steps, 1.0)
    local_step = (step + 1) % resample_steps
    return min(max(local_step, 1) / warmup_steps, 1.0)


@torch.no_grad()
def resample_dead_features(ae: BiasedReLUSAE, optimizer: torch.optim.Optimizer, dead: torch.Tensor,
                           x: torch.Tensor) -> int:
    """Re-initialize dead features from poorly reconstructed inputs (unit-norm directions)."""
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return 0
    x_hat, _ = ae(x)
    losses = (x - x_hat).norm(dim=-1)
    n_resample = min(n_dead, int(losses.shape[0]))
    if n_resample == 0 or float(losses.sum()) <= 0:
        return 0
    idx = torch.multinomial(losses, num_samples=n_resample, replacement=False)
    unit = x[idx] / x[idx].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    alive = ~dead
    alive_norm = ae.encoder.weight[alive].norm(dim=-1).mean() if alive.any() else torch.tensor(1.0, device=x.device)
    dead_idx = dead.nonzero(as_tuple=False).squeeze(-1)[:n_resample]
    ae.encoder.weight[dead_idx] = unit * alive_norm * 0.2
    ae.decoder.weight[:, dead_idx] = unit.T
    ae.encoder.bias[dead_idx] = 0.0
    n_latents = ae.encoder.weight.shape[0]
    for state in optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if value.ndim == 1 and value.shape[0] == n_latents:
                value[dead_idx] = 0.0
            elif value.ndim == 2:
                if value.shape[0] == n_latents:
                    value[dead_idx] = 0.0
                if value.shape[1] == n_latents:
                    value[:, dead_idx] = 0.0
    return n_resample


def raw_space_state(ae: BiasedReLUSAE, scale: float) -> Dict[str, torch.Tensor]:
    """Fold the training-time input scale into the weights (latents unchanged)."""
    return {
        "encoder.weight": (ae.encoder.weight.detach() * scale).cpu().clone(),
        "encoder.bias": ae.encoder.bias.detach().cpu().clone(),
        "bias": (ae.bias.detach() / scale).cpu().clone(),
        "decoder.weight": (ae.decoder.weight.detach() / scale).cpu().clone(),
    }


def save_checkpoint(path: Union[str, Path], state: Dict[str, torch.Tensor], d_in: int, n_latents: int,
                    spec: Dict[str, Any], summary: Dict[str, Any], architecture: str = ARCHITECTURE,
                    input_scale: Optional[float] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "d_in": int(d_in), "n_latents": int(n_latents), "spec": spec,
                "input_scale": input_scale, "architecture": architecture, "summary": summary}, path)
    return path


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    ckpt = torch_load(path)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"{path}: SAE checkpoint must be a dict")
    state = ckpt.get("state_dict", ckpt if all(torch.is_tensor(v) for v in ckpt.values()) else None)
    if state is None or "encoder.weight" not in state:
        raise RuntimeError(f"{path}: no 'state_dict' with 'encoder.weight'")
    return ckpt


class Encoder:
    """Encoder-only view of a checkpoint: ``encode(x)`` gives the latents on any device."""

    def __init__(self, ckpt: Dict[str, Any], device: Union[str, torch.device] = "cpu"):
        sd = ckpt["state_dict"]
        self.device = torch.device(device)
        self.w_enc = sd["encoder.weight"].float().to(self.device)
        self.b_enc = sd.get("encoder.bias")
        self.x_bias = sd.get("bias")
        self.w_dec = sd.get("decoder.weight")
        self.b_enc = None if self.b_enc is None else self.b_enc.float().to(self.device)
        self.x_bias = None if self.x_bias is None else self.x_bias.float().to(self.device)
        self.w_dec = None if self.w_dec is None else self.w_dec.float().to(self.device)
        self.n_latents, self.d_in = int(self.w_enc.shape[0]), int(self.w_enc.shape[1])

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(self.device)
        if self.x_bias is not None:
            x = x - self.x_bias
        z = x @ self.w_enc.T
        if self.b_enc is not None:
            z = z + self.b_enc
        return torch.relu(z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.w_dec is None:
            raise RuntimeError("checkpoint has no decoder")
        x_hat = z @ self.w_dec.T
        return x_hat + self.x_bias if self.x_bias is not None else x_hat


def raw_dims_checkpoint(d_in: int) -> Dict[str, Any]:
    """The 'raw' baseline as an encoder: ``[I; -I]``, the positive and negative part of each dimension."""
    eye = torch.eye(int(d_in))
    return {"state_dict": {"encoder.weight": torch.cat([eye, -eye])}, "d_in": int(d_in), "n_latents": 2 * int(d_in),
            "architecture": "raw_rectified_dims", "spec": {}, "summary": {}}


# --- data ---------------------------------------------------------------------------------------


def iter_batches(store_root: Union[str, Path], layer: str, batch_size: int, seed: int,
                 exclude_tokens: Sequence[str] = ()) -> Iterator[torch.Tensor]:
    """Endless batches of activation rows; shards visited in a seeded random order per pass."""
    rng = random.Random(int(seed))
    shards = list_shards(store_root, layer)
    special = {str(t) for t in exclude_tokens}
    while True:
        order = list(shards)
        rng.shuffle(order)
        for shard in order:
            X = torch_load(shard / "activations.pt").float()
            if special:
                idx = torch_load(shard / "index.pt")
                toks = idx.get("token_ids")
                if toks is not None and len(toks) == X.shape[0]:
                    keep = np.array([str(t) not in special for t in toks], dtype=bool)
                    X = X[torch.as_tensor(np.flatnonzero(keep))]
            for start in range(0, X.shape[0], batch_size):
                b = X[start:start + batch_size]
                if b.shape[0] > 0:
                    yield b


def infer_d_in(store_root: Union[str, Path], layer: str) -> int:
    for _name, X, _idx in iter_shards(store_root, layer, max_shards=1):
        return int(X.shape[1])
    raise RuntimeError(f"no shard for layer {layer!r} under {store_root}")


# --- training -----------------------------------------------------------------------------------


def train_sae(store_root: Union[str, Path], layer: str, params: SAETrainParams, out_dir: Union[str, Path],
              label: str = "sae", log=None) -> Dict[str, Any]:
    """Train one SAE on the stored activations of ``layer``; writes ``sae_best.pt``, ``sae_last.pt``,
    ``history.csv`` and ``summary.json`` under ``out_dir`` and returns the summary."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = params.resolved_device()
    d_in = infer_d_in(store_root, layer)
    n_latents = int(params.width_multiplier) * d_in
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)

    batches = iter_batches(store_root, layer, int(params.batch_size), params.seed, params.exclude_tokens)
    warm = [next(batches).to(dev) for _ in range(16)]
    sample = torch.cat(warm)
    scale = float(math.sqrt(d_in) / sample.norm(dim=-1).mean().clamp_min(1e-12).item())
    ae = BiasedReLUSAE(d_in, n_latents).to(dev)
    with torch.no_grad():
        ae.bias.copy_((sample * scale).mean(dim=0))
    del sample
    opt = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=params.lr)
    resample = int(params.resample_every) if params.resample_every else 0
    steps_since_active = torch.zeros(n_latents, dtype=torch.long, device=dev) if resample else None
    resampled_total = 0
    best_path, last_path = out_dir / "sae_best.pt", out_dir / "sae_last.pt"
    best_value, best_step = float("inf"), -1
    spec = asdict(params)
    spec["exclude_tokens"] = list(params.exclude_tokens)
    spec.update({"d_in": d_in, "n_latents": n_latents, "layer": layer})
    history: List[Dict[str, Any]] = []
    for step in range(int(params.steps)):
        x = (warm.pop(0) if warm else next(batches).to(dev)) * scale
        mult = warmup_lr_multiplier(step, int(params.warmup), resample or None)
        for group in opt.param_groups:
            group["lr"] = params.lr * mult
        opt.zero_grad(set_to_none=True)
        x_hat, z = ae(x)
        sq_err = (x - x_hat).pow(2).sum(dim=-1)
        recon = sq_err.mean()
        sparsity = z.sum(dim=-1).mean()
        loss = recon + params.l1 * sparsity
        loss.backward()
        opt.step()
        if steps_since_active is not None:
            dead = (z == 0).all(dim=0)
            steps_since_active[dead] += 1
            steps_since_active[~dead] = 0
            if (step + 1) % resample == 0:
                to_resample = steps_since_active > max(resample // 2, 1)
                n = resample_dead_features(ae, opt, to_resample, x)
                resampled_total += n
                if n > 0:
                    steps_since_active[to_resample] = 0
        with torch.no_grad():
            fvu = float(sq_err.sum() / (x - x.mean(dim=0)).pow(2).sum().clamp_min(1e-12))
            l0 = float((z > 0).float().sum(dim=-1).mean().item())
        row = {"step": step + 1, "loss": float(loss.item()), "recon": float(recon.item()),
               "sparsity": float(sparsity.item()), "fvu_batch": fvu, "l0": l0, "l0_frac": l0 / n_latents,
               "lr": float(opt.param_groups[0]["lr"]), "resampled_total": int(resampled_total)}
        history.append(row)
        if row["loss"] < best_value:
            best_value, best_step = row["loss"], step + 1
            save_checkpoint(best_path, raw_space_state(ae, scale), d_in, n_latents, spec,
                            {**row, "checkpoint": "best_loss", "label": label}, input_scale=scale)
        if log is not None and (step + 1) % 500 == 0:
            log.info("[%s:%s] step %d loss %.4f fvu %.4f l0 %.1f (%.2f%%) best %.4f@%d", label, layer, step + 1,
                     row["loss"], fvu, l0, 100 * row["l0_frac"], best_value, best_step)
    last = history[-1]
    save_checkpoint(last_path, raw_space_state(ae, scale), d_in, n_latents, spec,
                    {**last, "checkpoint": "last", "best_step": best_step, "best_loss": best_value, "label": label},
                    input_scale=scale)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"label": label, "layer": layer, "d_in": d_in, "n_latents": n_latents, "l1": params.l1,
               "best_path": str(best_path), "last_path": str(last_path), "best_step": best_step,
               "best_loss": best_value, "final": last, "resampled_total": resampled_total, "input_scale": scale,
               "device": str(dev), "params": spec}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def untrained_sae(store_root: Union[str, Path], layer: str, width_multiplier: int, seed: int,
                  out_path: Union[str, Path]) -> Dict[str, Any]:
    """The untrained baseline: a ReLU SAE of the same width at its seeded default initialization
    (:class:`interp_pipeline.sae.sae_base.AutoEncoder`, no biases, no training step)."""
    from interp_pipeline.sae.sae_base import AutoEncoder

    d_in = infer_d_in(store_root, layer)
    n_latents = int(width_multiplier) * d_in
    torch.manual_seed(int(seed))
    ae = AutoEncoder(d_in=d_in, n_latents=n_latents)
    state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
    spec = {"width_multiplier": int(width_multiplier), "seed": int(seed), "d_in": d_in, "n_latents": n_latents,
            "layer": layer, "untrained": True}
    save_checkpoint(out_path, state, d_in, n_latents, spec, {"untrained": True, "step": 0},
                    architecture="relu_sae_untrained")
    return {"layer": layer, "d_in": d_in, "n_latents": n_latents, "path": str(out_path), "untrained": True}


# --- QC and the λ rule --------------------------------------------------------------------------


def sae_qc(ckpt: Union[str, Path, Dict[str, Any]], X: torch.Tensor, device: Union[str, torch.device] = "cpu",
           chunk: int = 8192, near_dead_rate: float = 1e-4):
    """Reconstruction and sparsity QC of a checkpoint on evaluation rows ``X``.

    Returns ``(summary, per_feature)``: fraction of variance unexplained (``fvu``), mean
    number and fraction of active features per row (``l0_mean``, ``l0_frac``), the fraction
    of never-active features (``dead_frac``) and of features active on fewer than
    ``near_dead_rate`` of the rows (``near_dead_frac``); per feature the active rate, mean
    and maximum activation.
    """
    import pandas as pd

    if not isinstance(ckpt, dict):
        ckpt = load_checkpoint(ckpt)
    enc = Encoder(ckpt, device)
    X = X.float()
    mu = X.mean(0, keepdim=True).to(enc.device)
    fires = torch.zeros(enc.n_latents, device=enc.device)
    zsum = torch.zeros(enc.n_latents, device=enc.device)
    zmax = torch.zeros(enc.n_latents, device=enc.device)
    err = tot = l0 = 0.0
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].to(enc.device)
        z = enc.encode(x)
        if enc.w_dec is not None:
            err += float(((x - enc.decode(z)) ** 2).sum())
        tot += float(((x - mu) ** 2).sum())
        l0 += float((z > 0).sum())
        fires += (z > 0).sum(0).float()
        zsum += z.sum(0)
        zmax = torch.maximum(zmax, z.max(0).values)
    n = len(X)
    rate = (fires / n).cpu().numpy()
    summary = {"n_rows": int(n), "d_in": enc.d_in, "n_latents": enc.n_latents,
               "fvu": (err / tot) if (enc.w_dec is not None and tot > 0) else None,
               "l0_mean": l0 / n, "l0_frac": l0 / n / enc.n_latents,
               "dead_frac": float((rate == 0).mean()), "near_dead_frac": float((rate < near_dead_rate).mean()),
               "near_dead_rate": near_dead_rate, "checkpoint_step": ckpt.get("summary", {}).get("step")}
    per_feature = pd.DataFrame({"feature": np.arange(enc.n_latents), "active_rate": rate,
                                "mean_activation": (zsum / n).cpu().numpy(), "max_activation": zmax.cpu().numpy()})
    return summary, per_feature


def select_lambda(qc_by_lambda: Dict[float, Dict[str, Any]], l0_max: float = 0.02, dead_max: float = 0.25,
                  fvu_fallback: Optional[float] = 0.2) -> Tuple[Optional[float], str]:
    """The fixed λ rule.

    ``qc_by_lambda`` maps each λ to its QC summary (:func:`sae_qc`). Among the λ with
    ``l0_frac <= l0_max`` and ``dead_frac <= dead_max`` the lowest ``fvu`` wins; when none
    qualifies, the lowest ``l0_frac`` among those with ``fvu <= fvu_fallback`` (if a fallback
    is set); otherwise ``None``. Ties go to the smaller λ. Returns ``(λ, rule)``.
    """
    rows = sorted(((float(l), q) for l, q in qc_by_lambda.items()), key=lambda t: t[0])
    ok = [(l, q) for l, q in rows if q["fvu"] is not None and q["l0_frac"] <= l0_max and q["dead_frac"] <= dead_max]
    if ok:
        return min(ok, key=lambda t: (t[1]["fvu"], t[0]))[0], f"l0_frac<={l0_max}, dead_frac<={dead_max}, min fvu"
    if fvu_fallback is not None:
        fb = [(l, q) for l, q in rows if q["fvu"] is not None and q["fvu"] <= fvu_fallback]
        if fb:
            return min(fb, key=lambda t: (t[1]["l0_frac"], t[0]))[0], f"fallback: min l0_frac with fvu<={fvu_fallback}"
    return None, "none"
