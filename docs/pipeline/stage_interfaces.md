# Stage `interfaces`: calibrated cell–cell interface test

Directory `11_interfaces/`. Inputs: `ingest`, `extract` (the activation store, for the fired-on genes), `cells` (the projected features `08_cells/<layer>/cells.h5ad`) and `niches` (the niche labels); the dataset is read again for the coordinates in their native unit, the raw counts (covariates, partner markers) and the labels. Core environment, CPU; about 30 s per layer on the CosMx section (81,236 cells, 100 permutations). Applies only when `dataset.spatial` is set and needs `dataset.labels`; a configuration without coordinates records the stage as `skipped`.

## What it computes

For every projected layer, on the cells that have a feature profile (`has_row`):

1. **Labels.** The exact labels of `dataset.labels` (or of the stage's own `labels` set) are normalised (lower case; spaces, slashes and hyphens become `_`; `labels.normalize: false` keeps them verbatim) and mapped to groups with `dataset.labels.groups` (a CSV with columns `label`/`cell_type` and `group`) or `dataset.labels.group_map` (inline); a label without a group is its own group. Pairs are tested between **groups**; the permutation strata use the **exact** labels. Groups listed in `exclude_groups` (default `low_quality`, `unresolved`) are dropped from the pairs.
2. **Covariates** from the dataset: log1p total counts, log1p detected genes and log1p cell area (`covariates.area`: `{column: <obs column>}`, or `{product: [<a>, <b>]}` for a bounding box; without it the score is adjusted on counts and genes only, and the summary records which covariates were used).
3. **Program score.** `P` is the mean over features of the positive part of the feature z-scores, each feature clipped at its `clip_quantile` (0.995) quantile. `P_adj` residualises `P` on the covariates (with an intercept) within each exact label.
4. **Pairs and statistic.** Edges join cells closer than `edge_radius_um`. For every unordered pair of groups, within each niche (the calibrated test) or over the whole section (the niche-free test), the statistic is the mean over the pair's edges of the two cells' scores; pairs with fewer than `min_edges` (50) edges are not tested.
5. **Null.** The score is permuted within strata = square block of side `block_um` × niche (niche test only) × exact label, `n_perm` (100) times; `z = (observed - null mean) / null sd`, with one-sided empirical p values. The **legacy** test (raw `P`, strata = `legacy_tile_um` × niche × label, `legacy_perm` = 20 permutations) is the manuscript's earlier uncalibrated statistic and is reported beside the calibrated one.
6. **Calibration.** Contact-free smooth random fields (Gaussian bumps at `length_scales_um`, `calib_reps` fields each, mixed with white noise so that the field's neighbour correlation matches that of `P_adj`) are pushed through the legacy null, the calibrated niche null and the niche-free null with `calib_perm` (50) permutations; an i.i.d. normal field is pushed through the niche null as well. `z* = max(2, 95th percentile of |z|)` pooled over fields and pairs, separately for the niche and the niche-free test. A pair is **enriched** when `z ≥ z*`, **depleted** when `z ≤ -z*`, otherwise **not supported**.
7. **Effect** = `100 · (observed - null mean) / mean edge P` (the unadjusted score on the same edges), in percent.
8. **Partner markers** of every exact label (top `markers.top_n` genes by a one-vs-rest Wilcoxon test on log1p CP10k, `tie_correct=False`) and their union per group; **fired-on genes** of every projected feature from the first `fired_on.max_shards` shards of the store: a feature fires on a gene-token row when its latent exceeds `fired_on.threshold` (0.15); its fired-on set is the genes whose rows fire it in ≥ `fired_on.rate` (20 %) of their occurrences, plus the smallest set of top genes carrying `fired_on.mass` (80 %) of its activation mass. Both are inputs of the two cross-model checks below.

### What the calibration controls, and what it does not

The calibration sets the call threshold so that a **smooth, contact-free** score field (one whose spatial autocorrelation matches the adjusted score's) is called in about 5 % of the tested pairs: `z*` is the 95th percentile of `|z|` under such fields, never below 2. It controls the per-pair false-positive rate under that family of nulls (the summary reports the shares of `|z| > 2` under the smooth fields and under i.i.d. noise, for the legacy and the calibrated null). It is **not** a false-discovery rate over the pairs tested (no multiplicity correction is applied), it does not account for nulls outside that family (for example label-dependent score shifts beyond the covariate adjustment), and its **power is not evaluated**: a pair that is "not supported" may be a true interface that the test cannot see. The legacy tile null is reported beside it because the manuscript shows how the share of `|z| > 2` drops from the uncalibrated to the calibrated test.

## Units

Every length is given in µm and converted once to the dataset's coordinate unit with `dataset.spatial.um_per_unit` (rounded to 1e-6 units). The manifest and `summary.json` record both values. The manuscript's values are 120 native units for the edge radius and the stratum block, 400 for the legacy tile and 60 / 150 / 400 for the smooth-field length scales, in both datasets: for CosMx (0.18 µm per pixel) that is `edge_radius_um: 21.6`, `block_um: 21.6`, `legacy_tile_um: 72`, `length_scales_um: [10.8, 27, 72]`; for Xenium (coordinates in µm) 120 / 120 / 400 / [60, 150, 400]. The CosMx block of 21.6 µm holds few cells per stratum (the permutation null is correspondingly tight); the paper configurations keep the published value for reproduction rather than a per-dataset choice.

## Parameters (`stages.interfaces`)

| key | default | meaning |
|---|---|---|
| `edge_radius_um` | required | edge radius in µm |
| `block_um` | = edge radius | side of the stratum block |
| `legacy_tile_um` | 10/3 × block | tile of the legacy uncalibrated null |
| `length_scales_um` | `[0.5, 1.25, 10/3] × block` | smooth-field length scales |
| `n_perm` | 100 | permutations of the calibrated and niche-free tests |
| `calib_perm` | 50 | permutations per calibration field |
| `calib_reps` | 2 | fields per length scale |
| `legacy_perm` | 20 | permutations of the legacy test |
| `min_edges` | 50 | smallest tested pair |
| `clip_quantile` | 0.995 | feature clipping of the program score |
| `exclude_groups` | `[low_quality, unresolved]` | groups dropped from the pairs |
| `signal_fractions` | `[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]` | candidate signal fractions of the matched fields |
| `seeds` | the reference seeds (`interpretfm.interfaces.DEFAULT_SEEDS`) | `calibration` (fields), `noise`, `calib_legacy_rep` / `calib_niche_rep` / `calib_free_rep` (+ replicate), `calib_noise`, `legacy`, `legacy_adjusted`, `niche`, `free`, `legacy_free` |
| `labels` | `{normalize: true}` | `normalize`, plus an optional label set for this stage with the keys of `dataset.labels` (`column`, `file`, `file_column`, `groups`, `group_map`) used instead of the dataset labels (the manuscript's Xenium interface test uses marker-derived labels from a per-cell CSV, exact label = group, while its cell-level numbers use the dataset's `cell_type` column) |
| `covariates` | `{}` | `area: {column: <obs column>}` or `{product: [<a>, <b>]}` |
| `niche` | `{column: niche}` | the niche partition: a column of `10_niches/<layer>/labels.csv` (`niche` = the sweep winner, `niche_at_<config>` = a fixed configuration), or `{config: <label>}` |
| `niche_labels` | `{}` | `{<layer>: csv}` with columns `cell, <niche column>` instead of the `niches` stage's labels |
| `cells_h5ad` | `{}` | `{<layer>: path}`: projected features instead of the `cells` stage's |
| `markers` | `{enabled: true, top_n: 10}` | |
| `fired_on` | `{enabled: true, max_shards: null, threshold: 0.15, rate: 0.2, mass: 0.8, batch_size: 8192}` | `max_shards: null` reads every shard; the manuscript read the first 40 scGPT shards (483,512 gene-token rows) and the first 4 Geneformer / C2S shards |
| `store_root`, `token_vocab`, `checkpoints` | | as in the other stages (the store, the token vocabulary and an SAE checkpoint per layer; the default checkpoint is the one the `cells` stage used) |

The paper configurations carry the manuscript's values (`configs/paper/*.yaml`, section `interfaces`), the CosMx author-type → group map (`configs/paper/cosmx_author_groups.csv`) and the manuscript's niche configuration per model (`niche.config`). The permutation nulls draw one random vector per permutation and sort within strata, so the results are a deterministic function of the seeds; with the reference seeds the manuscript's tables are reproduced to floating-point precision (the empirical p values of pairs whose null statistic ties the observed one can differ, the z values and calls do not).

## Outputs

```
11_interfaces/
  summary.json                   parameters (µm and native), label scheme, covariates, markers; every layer's summary
  markers.csv                    label, n_cells, rank, gene, wilcoxon_score, logfoldchange, pval_adj, symbol
  markers_by_group.csv           group, markers (;-joined gene keys), markers_symbol, n_markers
  <layer>/niche_pairs.csv        model, niche, group_a, group_b, n_edges, z_v1, effect_pct_v1, z_v1_adjusted_score, z_v2, effect_pct_v2, p_upper_v2, p_lower_v2, zstar_v2, call_v2, mean_edge_P
  <layer>/free_pairs.csv         model, group_a, group_b, n_edges, z_v1_free, z_v2_free, effect_pct_v2_free, zstar_v2_free, call_v2_free
  <layer>/scores.csv             cell, P, P_adj, exact, group, niche
  <layer>/summary.json           calibration (fields, false-positive shares, quantiles per length scale, z*), counts, shares of |z| > 2, effect sizes, the flat `row`
  <layer>/fired_on.csv           feature, var_name, concept, fired_on (;-joined gene keys), n_fired, n_rate, n_mass, total_mass, n_firing_rows, fired_on_symbol
  <layer>/fired_on.json          rows and shards used, thresholds, checkpoint
```

`report.json` carries, per layer (`interfaces.<layer>`), the share of `|z| > 2` under the legacy tile null beside the calibrated null, `z*`, the smooth-field and i.i.d.-noise false-positive shares, the enriched / depleted counts of the niche and niche-free tests and the median and maximum effect; the HTML page shows the same table.

## Partner-marker tercile and marker-excluded checks (several runs)

Both checks need the niche-free calls of several models on the same section, so they are library recipes over finished runs (one per model, same dataset and labels):

```python
import anndata as ad
from interpretfm.config import load_config
from interpretfm.ingest import dataset_labels, label_group_map
from interpretfm.interfaces import (InterfaceParams, cell_area, covariates, fired_on_sets, group_labels,
                                    marker_excluded_calls, partner_marker_tercile)
from interpretfm.stages.interfaces import read_interface_run

runs = {"scgpt": ("<run A>", "layer_1.norm2"), "geneformer": ("<run B>", "layer_1"), "c2sscale": ("<run C>", "layer_13")}
R = {m: read_interface_run(rd, layer) for m, (rd, layer) in runs.items()}
cfg = load_config("<run A>/config.resolved.yaml")
a = ad.read_h5ad(cfg.dataset.h5ad)
cells = R["scgpt"]["scores"].index                      # the cells with features (same for every run)
a = a[cells].copy()
exact, group = group_labels(dataset_labels(cfg.dataset, a), label_group_map(cfg.dataset))
cov = covariates(a.X, cell_area(a.obs, {"product": ["Width", "Height"]}))
params = InterfaceParams.from_dict(R["scgpt"]["summary"]["params"])
markers = {r.group: set(r.markers.split(";")) for r in R["scgpt"]["markers_by_group"].itertuples()}
genes = [str(g) for g in a.var_names]                   # the gene keys the markers and fired-on tables use

# 1. partner-marker tercile: g = (z_t3 - z_t1) / |z_all| per called pair and model; block bootstrap over pairs
free = {m: R[m]["free"] for m in R}
rows, summary = partner_marker_tercile(free, {m: R[m]["scores"]["P_adj"].to_numpy() for m in R}, a.obsm["spatial"], exact, group,
                                       a.X, genes, markers, params, raw_scores={m: R[m]["scores"]["P"].to_numpy() for m in R})
summary.iloc[0]                                          # median g, 95 % CI, verdict ("CI includes 0": calls are not spillover-patterned)

# 2. marker-excluded calls: drop the features firing on a partner marker, re-test, recalibrate z*
feats = {m: ad.read_h5ad(f"{rd}/08_cells/{layer}/cells.h5ad")[cells].X for m, (rd, layer) in runs.items()}
fired = {m: fired_on_sets(R[m]["fired_on"]) for m in R}
res = marker_excluded_calls(free, feats, fired, a.obsm["spatial"], exact, group, cov, a.X, genes, markers, params)
res["verdicts"]                                          # one row per called pair: spillover_robust (|z| >= z* with the original sign in >= 2 models)
res["tercile_summary"]                                   # the tercile gradient of the reduced score, and of the full score on the same permutations
```

The tercile check splits each called pair's cross-group edges into terciles of the partner-marker fraction (raw counts of the partner group's markers over total counts, averaged over the edge's two cells; ties broken at random with seed 0), tests all edges and each tercile with a within-block × exact-label null (100 permutations, seed 5000 + pair index, the models permuted together) and reports the median of `g` over (pair, model) rows with a 95 % interval from 2,000 bootstrap resamples of whole pairs (seed 0). The marker-excluded check removes, per pair and model, every feature whose fired-on genes include a marker of either group, recomputes `P` and `P_adj` on the remaining features, re-runs the niche-free test on the same permutations (seed `free`) and recalibrates `z*` on the same smooth fields; the controls (`res["controls"]`) show that the full feature set reproduces the run's `z_v2_free` and `z*`.

Python: `interpretfm.interfaces` (`InterfaceParams`, `PairTest`, `program_score`, `covariates`, `adjust_score`, `calibrate`, `interface_test`, `partner_markers`, `markers_by_group`, `partner_marker_tercile`, `fired_on_genes`, `marker_excluded_calls`) and `interpretfm.stages.interfaces` (`interface_params`, `read_interface_run`).
