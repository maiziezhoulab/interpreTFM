# Stage `cells`: feature retention, projection to cells, population specificity

Directory `08_cells/`. Inputs: `ingest` (cell table), the activation store, `sae` (selected checkpoint), `concepts` (concept sizes and names) and `concept_eval` (the row-split tables of the SAE). Core environment; encoding uses the GPU when available.

## What it computes

For every layer with a selected SAE:

1. **Retention.** For each concept with 5–100 panel genes, the feature with the highest validation F1 at threshold 0.15 in the row split (ties: lowest index) is kept when that F1 is at least 0.40. Features are then de-duplicated; a feature selected by several concepts is labelled by its best one (highest validation F1, then fewer panel genes, then concept id) and `n_concepts_sharing_feature` records the others. Selection uses validation rows only; the test F1 of the kept pairs is reported. A layer needs `min_features` (10) retained features to be projected; below that, `enough_features` is false and the layer is not projected (its specificity is still computed).
2. **Projection.** The retained features are averaged over each cell's gene-token rows (special tokens excluded; cells recorded more than once in a store are reduced to their first copy) into `<layer>/cells.h5ad`: cells × features, `obs` with the labels, the QC flag, `has_row` and the number of gene tokens, `obsm['spatial_um']` with the coordinates in µm, `var` with the feature table. Feature names are `<concept name>|<concept id>|feat<index>` (or `<concept id>|feat<index>`).
3. **Population specificity**, following scPISCES, on a sample of cells read from evenly spaced shards (about `n_cells`; cells at shard edges are excluded because they may be split across shards). Three representations are scored on the same cells, label shuffles and half split:
   - `sae`: every SAE feature (not only the retained ones; `gsi_features.csv` marks the retained ones and `excess_mean_retained` summarizes them);
   - `raw`: every hidden dimension split into its positive and negative part, rectified per token before the cell mean;
   - `genes`: the panel genes themselves, log-normalized expression (`log1p` of counts scaled to 10,000 per cell).

   **GSI excess**: the Gini coefficient of a feature's mean activation across populations with at least `min_pop` (30) cells, minus the mean Gini of the same feature over `n_shuffles` (5) label shuffles, averaged over active features. **MHR@k**: cells are split 50/50 within each population; in half 1 the markers of a population are the `mhr_k` (100) genes with the largest mean-expression difference against all other cells, and the `n_leading` (3) features most correlated with population membership are selected; in half 2, MHR is the share of the `n_assoc` (50) genes most correlated with each selected feature that are markers, averaged over populations. A random-feature null (`n_null` draws of random leading features) and the chance level `mhr_k / n_genes` are reported next to it. Requires `dataset.labels`.

## Parameters (`stages.cells`)

| key | default | meaning |
|---|---|---|
| `retain.tau` | 0.15 | threshold of the F1 tables used for selection |
| `retain.min_valid_f1` | 0.40 | minimum validation F1 of the best feature per concept |
| `retain.concept_size` | `[5, 100]` | eligible concepts (panel genes) |
| `retain.min_features` | 10 | retained features a layer needs to be projected |
| `retain.unit` | `row` | held-out unit of the tables used (`row` or `gene`) |
| `projection.dedup_cells` | true | first copy of each cell only |
| `projection.batch_size` | 8192 | rows per encoding batch |
| `specificity.min_pop` | 30 | minimum cells per population |
| `specificity.n_shuffles` | 5 | label shuffles for GSI excess |
| `specificity.mhr_k` | 100 | markers per population (must not exceed the number of genes) |
| `specificity.n_leading` | 3 | leading features per population |
| `specificity.n_assoc` | 50 | correlated genes per leading feature |
| `specificity.n_null` | 50 | random-feature draws for the MHR null |
| `specificity.n_cells` | 40000 | cells to sample (evenly spaced shards; all cells when larger than the dataset) |
| `specificity.representations` | `[sae, raw, genes]` | |
| `specificity.exclude_qc_flagged` | false | drop QC-flagged cells from the specificity sample (the number used is always reported) |
| `specificity.seed` | `run.seed` | shuffles, half split and null draws |
| `device` | `auto` | |
| `checkpoints` | `{}` | `{<layer>: path}`: SAE checkpoints instead of the selected ones |
| `store_root` | null | a cached activation store |

The manuscript's CosMx numbers (T6: 194 features; T7: GSI excess and MHR@100) are reproduced with `specificity.n_cells` large enough to read every shard (the published run read 75,745 cells) and `specificity.seed: 0`.

## Outputs

```
08_cells/
  selection.json                 rule and per-layer retention summary (n_concepts_kept, n_features, enough_features, test F1 of the kept set)
  specificity.json               parameters and per-layer GSI excess / MHR per representation, with null and chance
  <layer>/features.csv           feature, concept, valid_f1, n_panel_genes, test_f1, n_concepts_sharing_feature, var_name[, term_name]
  <layer>/kept_concepts.csv      every concept passing the rule (before de-duplication)
  <layer>/per_concept_best.csv   the best feature per eligible concept
  <layer>/selection.json
  <layer>/cells.h5ad             cells x retained features (projected layers only)
  <layer>/projection.json        row counts (used, special, duplicates dropped, unmatched), cells with a row, feature activity
  <layer>/specificity.json
  <layer>/gsi_features.csv       per SAE feature: gini, shuffled, excess, retained
  <layer>/mhr_by_population.csv  representation, population, n_cells, mhr, leading_features
```

Python: `interpretfm.cells` (`select_features`, `accumulate_cell_profiles`, `gsi_excess`, `mhr_split_half`, `specificity`).
