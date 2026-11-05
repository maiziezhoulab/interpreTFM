# Stage `tis`: transcriptomic interpretability score

Directory `03_tis/`. Input: the activation store (`02_extract/store.json`, or `store_root`). Core environment, CPU.

## What it computes

For every configured layer the stage builds the cell-level activation matrix from the store (the classification-token row for `model.cell_pooling.kind: token`, the mean over a cell's gene tokens, special tokens excluded, for `kind: mean`), aligns it to the cells of the dataset and scores every activation dimension:

- **TIS** of dimension *d*: in each of `trials` trials one query cell is drawn; `k_ref` reference cells are drawn without replacement from the cells at or above the upper quantile of *d* and from those at or below the lower quantile (the query excluded); the query's expression profile (library-size normalized to 10⁴, log1p, L2-normalized) is compared with the two reference centroids by cosine similarity, and the trial is correct when the more similar pool matches the query's side of the median of *d*. TIS is the accuracy over trials.
- **Shuffled control**: each dimension permuted independently across cells, then the same procedure. The layer's **TIS gain** is the mean TIS over dimensions minus the mean shuffled TIS.
- **PCA control**: the same activations rotated to their principal components (`pca_of: activations`, the default; this dimension-matched change of basis is the control reported in the manuscript) or a PCA of the expression matrix with as many components (`pca_of: expression`), scored in the same way and reported as a descriptive reference.

The computations are those of `interp_pipeline.tis.core` (0.9); the stage adds the store reader, the configuration and the per-layer records.

## Parameters (`stages.tis`)

| key | default | meaning |
|---|---|---|
| `trials` | 800 | trials per dimension |
| `k_ref` | 32 | reference cells drawn from each pool |
| `quantiles` | `[0.25, 0.75]` | lower/upper pool quantiles |
| `judge` | `log1p_cp10k` | expression transform (`raw` uses `X` as is) |
| `controls` | `[shuffle, pca]` | controls to run |
| `pca_of` | `activations` | `activations` (rotate the activations) or `expression` (PCA of the expression matrix) |
| `seed` | 42 | random generator of the trials and the permutation |
| `subsample_eval` | null | restrict query cells to a random subset |
| `max_shards` | null | read only the first *n* shards (smoke runs) |
| `keep_cell_matrix` | true | save the aligned cell matrix for later stages |
| `store_root` | null | a cached activation store instead of the `extract` output |

The manuscript's values are the defaults (T = 800, m = 32, quartile pools, seed 42).

## Outputs

```
03_tis/
  tis.json                      parameters, pooling, per-layer summaries (mean/median/p90/p95 of TIS,
                                shuffled and PCA; gain; extraction counts)
  layers.csv                    one row per layer: tis_mean, shuffled_mean, tis_gain, pca_mean, ...
  <layer>/tis.json
  <layer>/tis_per_dim.csv       dim, tis, tis_shuffled, tis_pca
  <layer>/cell_activations.npy  cells x hidden (aligned to the dataset; zero rows where has_row is false)
  <layer>/has_row.npy
```

Python: `interpretfm.tis.tis_for_matrix(A, adata, TISParams(...))` scores any aligned cell × dimension matrix; `interpretfm.stores.cell_matrix` builds one from a store.
