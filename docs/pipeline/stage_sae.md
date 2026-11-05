# Stage `sae`: sparse autoencoders

Directory `04_sae/`. Input: the activation store. Core environment; training uses the GPU when `device` allows it, the toy settings run on a CPU in seconds. The model itself is not needed, only its stored activations.

## What it computes

Per layer and per λ of the grid, a ReLU SAE with an encoder bias and a bias shared by the input and the output,

```
z = relu(W_enc (x - b) + b_enc),   x_hat = W_dec z + b,
```

with `width_multiplier` × hidden-dimension features, trained on the stored activation rows rescaled to mean norm √h, minimizing the squared reconstruction error plus λ‖z‖₁ with unit-norm decoder columns (constrained Adam), linear warm-up (restarted after every resampling period), dead-feature resampling and the checkpoint with the lowest training loss kept. Checkpoints are stored in the original activation scale (`encoder.weight`, `encoder.bias`, `bias`, `decoder.weight`).

Every checkpoint is QC'd on evenly spaced evaluation rows: fraction of variance unexplained (`fvu`), active features per row (`l0_mean`, `l0_frac`), never-active features (`dead_frac`) and features active on fewer than 10⁻⁴ of the rows (`near_dead_frac`), plus a per-feature table.

The **λ rule** (`interpretfm.sae.select_lambda`): among the λ whose SAE has `l0_frac ≤ 0.02` and `dead_frac ≤ 0.25`, the lowest `fvu`; if none qualifies, the lowest `l0_frac` among those with `fvu ≤ 0.2`; otherwise no SAE is selected for the layer (downstream stages skip it). Ties go to the smaller λ.

The **untrained baseline** is a ReLU SAE of the same width at its seeded default initialization (no biases, no training step), QC'd and written next to the trained ones.

## Parameters (`stages.sae`)

| key | default | meaning |
|---|---|---|
| `width_multiplier` | 8 | features = multiplier × hidden dimension |
| `lambda` | `{grid: [0.1, 0.3, 1, 3, 10], rule: {l0_max: 0.02, dead_max: 0.25, fvu_fallback: 0.2}}` | a number fixes λ instead |
| `steps`, `lr`, `warmup`, `resample_every`, `batch_size` | 8000, 1e-4, 1000, 2000, 1024 | the training schedule |
| `seed` | `run.seed` | initialization, shard order and resampling |
| `device` | `auto` | `auto`, `cpu`, `cuda`, ... |
| `exclude_special_tokens` | false | train on gene rows only (the published SAEs saw every stored row) |
| `untrained_baseline` | true | write the untrained SAE |
| `qc` | `{max_shards: 300, rows_per_shard: 512}` | evaluation rows (evenly spaced shards and rows) |
| `checkpoints` | `{}` | `{<layer>: path}` or `{<layer>: {<λ>: path}}`: cached checkpoints to QC and select instead of training |
| `store_root` | null | a cached activation store |

## Outputs

```
04_sae/
  selected.json                 per layer: lambda, rule, checkpoint, qc of every λ, untrained qc
  <layer>/selected.json
  <layer>/lambda_<λ>/sae_best.pt  sae_last.pt  history.csv  summary.json  qc.json  qc_features.csv
  <layer>/untrained/sae.pt  qc.json  qc_features.csv
```

A checkpoint is a `torch.save` dict with `state_dict`, `d_in`, `n_latents`, `spec`, `input_scale`, `architecture` and `summary`; `interpretfm.sae.Encoder` encodes activations with it, `interpretfm.sae.load_checkpoint` reads it (the 0.9 `.pt` files load unchanged).
