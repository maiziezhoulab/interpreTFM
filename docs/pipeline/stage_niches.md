# Stage `niches`: spatial niches, baselines, seed stability, concept labels

Directory `10_niches/`. Inputs: `ingest` and `cells` (the projected features `08_cells/<layer>/cells.h5ad`); the dataset is read again for the coordinates in their native unit, the raw counts (expression baseline) and the labels. Core environment, CPU. Applies only when `dataset.spatial` is set; a configuration without coordinates records the stage as `skipped`.

## What it computes

For every projected layer, on the cells that have a feature profile (`has_row`):

1. **Niche sweep.** For each cell the mean feature vector of its neighbours within a radius, the cell included (`m`), optionally concatenated with the cell's own features (`xm`); standardized; reduced to `n_pcs` principal components (clamped to the number of features); clustered with k-means (one k-means++ initialisation) or a full-covariance Gaussian mixture, once per seed. Configurations whose smallest cluster has fewer than `min_cluster` cells are dropped. Each remaining configuration is scored with

   `S = score.ari · ARI_seed + score.purity_z · clip(z, ±purity_z_clip) + score.sep · separability`

   where `ARI_seed` is the mean adjusted Rand index between the first seed's labels and the other seeds', `z` the neighbourhood purity (fraction of a cell's neighbours with its label, averaged over cells) against a null that permutes labels within square tiles of `tile_mult` × radius (`n_null` draws, `null_seed`), and separability the mean cosine distance between cluster centroids in the PCA space.
2. **Selection and validation.** The highest score wins (`k_report` restricts the number of clusters). Its labels (first seed, numbered by decreasing size) are written with their sizes, purity, tile null and z, separability, seed ARI and, with `dataset.labels`, ARI and NMI against the labels.
3. **Baselines**, each reported next to the model's numbers:
   - `expression_svd50`: `n_components` truncated-SVD components of log-normalized expression (`log1p` of counts scaled to 10,000 per cell, uncentred SVD), run through the same sweep and score (its own winner) *and* at the model's selected configuration (and at every fixed configuration);
   - `label_composition`: for every cell the fraction of each label among its neighbours within the radius (the cell itself excluded), PCA to `label_composition.n_pcs` components, k-means with `n_init` initialisations; at `label_composition.radius_um` / `k` (the manuscript's published parameters) and at the model's selected radius and k. Needs `dataset.labels`;
   - optional `expression_variants`: four preprocessing variants of the expression features (CP10k log1p with genes standardized and clipped at ±10 then PCA; the same without scaling; uncentred truncated SVD; median library size, scaled, PCA) clustered at the model's configuration with `expression_variants.seeds`.
   - the tile-shuffle null of the purity is the model's own spatial null.
4. **Seed stability**: median seed-to-seed ARI over the configurations with `seed_stability.k` clusters (the finer partitions), model versus expression-only, with the other k values as a control.
5. **Concept labels.** Every feature is tested inside versus outside each niche (Wilcoxon rank-sum, Benjamini–Hochberg over the features of a niche); effect = difference of the mean activation, active-fraction difference = difference of the fraction of cells with activation above `concepts.active_threshold`, `score` = effect × max(active-fraction difference, 0). The top `concepts.top_k` positive-effect features per niche are ranked by effect, then active-fraction difference, then score (`concepts.rank_by: effect`, the manuscript's ranking); `rank_by: score` ranks by the product instead. Cross-model agreement needs two runs and is a library recipe (`interpretfm.niche_concepts`, see below).

## Units

Radii are given in micrometres (`radii_um`, `label_composition.radius_um`) and converted once to the dataset's coordinate unit with `dataset.spatial.um_per_unit` (the converted value is rounded to 1e-6 units to absorb floating-point error). Neighbourhoods and tiles are computed in that unit, which is equivalent to working in µm. The manifest, `selected.json` and `sweep.csv` record both values: for CosMx (0.18 µm per pixel) the manuscript's radii 100 / 120 / 150 pixels are `radii_um: [18, 21.6, 27]`; for Xenium (coordinates in µm) `radii_um: [100, 120, 150]`. Configuration labels read `r18um/xm/kmeans/k3 (r100 native)`.

## Parameters (`stages.niches`)

| key | default | meaning |
|---|---|---|
| `labels` | `dataset.labels` | a label set for this stage with the keys of `dataset.labels` (`column`, `file`, `file_column`), used for the ARI/NMI, the label-composition baseline and the niche matching instead of the dataset labels (the manuscript's Xenium niches are compared with marker-derived labels from a per-cell CSV while its cell-level numbers use the dataset's `cell_type` column) |
| `radii_um` | required | neighbourhood radii in µm |
| `k` | `[3, 4, 5]` | numbers of clusters |
| `methods` | `[kmeans, gmm]` | k-means (one initialisation per seed) and full-covariance GMM |
| `spaces` | `[m, xm]` | neighbourhood mean only, or cell + neighbourhood |
| `seeds` | `[0, 1, 2]` | clustering seeds; the first seed's labels are reported |
| `min_cluster` | 150 | smallest allowed cluster |
| `n_pcs` | 50 | principal components (clamped to the number of features) |
| `kernel` | `uniform` | neighbourhood weights (`gaussian` with `sigma_frac` × radius) |
| `tile_mult` | 4 | tile size of the purity null, in radii |
| `n_null` | 10 | null draws |
| `null_seed` | 0 | |
| `score` | `{ari: 1.5, purity_z: 0.1, sep: 1.0, purity_z_clip: 10}` | score weights and the z clip |
| `k_report` | null | restrict the selection to this number of clusters |
| `baselines` | `[expression_svd50, label_composition]` | |
| `expression` | `{n_components: 50, seed: 0}` | |
| `label_composition` | `{radius_um: <first radius>, k: <first k>, n_pcs: 15, n_init: 20, seed: 0}` | the published parameters; the baseline is also computed at the model's configuration |
| `expression_variants` | `{enabled: false, seeds: [0, 1, 2], variants: [all four]}` | |
| `seed_stability` | `{k: [4, 5]}` | |
| `concepts` | `{top_k: 25, active_threshold: 0.0, min_cells: 20, rank_by: effect}` | |
| `fixed_configs` | `[]` | `[{radius_um, space, method, k}, ...]`: configurations validated besides the winner (sizes, purity, null, z, ARI/NMI, their sweep rank and score, ARI against the selected labels); the paper configurations list the manuscript's selected configuration here so that its numbers are reported whatever the sweep picks |
| `cells_h5ad` | `{}` | `{<layer>: path}`: projected features to use instead of the `cells` stage's (cells × features with `obsm[<dataset.spatial.obsm>]` or the `cells` layout) |

The manuscript's CosMx configuration (`configs/paper/cosmx_scgpt_layer2.yaml`) uses the defaults with `radii_um: [18, 21.6, 27]` and `label_composition: {radius_um: 21.6, k: 3}`. The dev driver's argparse defaults (radii 120/150/180 pixels, five seeds, 20 null draws) are not the manuscript grid; the configuration is the only source of the grid.

## Outputs

```
10_niches/
  selected.json                     grid (radii in µm and native units), every layer's selected configuration, validation, baselines, stability, top concepts
  <layer>/sweep.csv                 one row per configuration: radius_um, radius (native), space, method, n_clusters, ari_mean, purity, purity_null_mean/std, purity_z, separability, fragmentation, score, config
  <layer>/selected.json             the layer's summary (same content as its entry in selected.json)
  <layer>/validation.json           sizes, purity, null, z, separability, ARI/NMI of the selected labels; the same for every fixed configuration
  <layer>/labels.csv                cell, niche (cells with a feature profile)[, niche_at_<config> per fixed configuration]
  <layer>/baselines/expression_sweep.csv
  <layer>/baselines/baselines.json  expression-only winner and at-model-configuration results, label composition (published and matched), expression variants
  <layer>/baselines/labels.csv      cell, expression_winner, expression_at_model_config, label_composition_published, label_composition_at_model_config[, expression_<variant>_seed<s>]
  <layer>/stability.json            median seed ARI over the finer configurations, model vs expression
  <layer>/concepts.csv              niche x feature tests (n_in, n_out, means, effect, active fractions, score, pval, qval, rank)
  <layer>/top_concepts.csv          the top_k features per niche
  <layer>/top_concepts_at_<config>.csv   the same for the niches of each fixed configuration
```

`report.json` carries the niche numbers beside their baselines (`niches.<layer>`: model ARI/NMI, seed ARI, purity with its tile null; expression-only winner and at the model's configuration; label composition; seed stability; top concepts).

## Cross-model concept agreement (several runs)

The manuscript compares the top-25 concept labels of the niche-defining features between models (Supplementary Note S1): niches of two runs are matched by label composition on the cells both runs labelled (Hungarian assignment minimising the maximum absolute difference of label fractions), and the Jaccard index of the top-concept sets is reported for matched and unmatched niche pairs. `cross_model_concepts` does this for every pair of finished runs (one row per pair: the matching, the shared labels and Jaccard of every matched niche, the matched and unmatched means):

```python
import pandas as pd
from interpretfm.niche_concepts import cross_model_concepts, read_niche_run
runs = {"scgpt": read_niche_run("<run of scGPT>", "layer_1.norm2", "r18um/xm/kmeans/k3"),      # a fixed configuration
        "geneformer": read_niche_run("<run of Geneformer>", "layer_1", "r21.6um/m/kmeans/k3"),
        "c2s_scale": read_niche_run("<run of C2S-Scale>", "layer_13")}                         # or the sweep winner
ref = pd.read_csv("<any of the runs>/01_ingest/cells.csv", index_col="cell")["label"]          # or the stage's label set
table = cross_model_concepts(runs, ref)
```

`read_niche_run` returns the top-concept table and the cell-indexed niche labels of a run's `niches` stage (the sweep winner, or `niche_at_<config>` / `top_concepts_at_<config>.csv` of a fixed configuration). The reference labels must be the ones the niche compositions should be matched on (for the Xenium paper runs the marker labels of `stages.niches.labels`). The two-run building blocks remain available (`niche_composition`, `match_niches`, `cross_model_jaccard`, `summarize_jaccard`); without labels pass an explicit `matching` to `cross_model_jaccard` or none (equal niche ids). The reproduction tests run the recipe on the three CosMx and the three Xenium paper runs (`tests/reproduction/test_other_layers.py`).

Python: `interpretfm.niches` (`NicheGrid`, `niche_sweep`, `select_config`, `cluster_niches`, `niche_validation`, `expression_svd`, `expression_variant`, `label_composition`, `seed_stability`) and `interpretfm.niche_concepts` (`niche_feature_tests`, `top_concepts`, `niche_composition`, `match_niches`, `cross_model_jaccard`, `read_niche_run`, `cross_model_concepts`).
