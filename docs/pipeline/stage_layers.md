# Stage `layers`: per-criterion layer summary

Directory `09_layers/`. Inputs: `tis`, `sae`, `concept_eval`, `gene_readout`, `cells`. Core environment; seconds.

## What it reports

One row per layer with every criterion and its baselines, **reported separately; no composite score is computed**. The manuscript's representative layers were fixed in a preliminary screen and kept; here the representative layer is a configuration choice (`model.representative_layer`), and `recommendations.json` only names the best layer per criterion so that a reader can weigh them.

| criterion | column | baselines (same row) |
|---|---|---|
| TIS gain | `tis_gain` | `tis_shuffled_mean`, `tis_pca_mean` |
| Concept F1, row split | `concept_f1_row_sae` | `concept_f1_row_raw`, `concept_f1_row_untrained` |
| Concept F1, whole-gene split | `concept_f1_gene_sae` | `concept_f1_gene_raw`, `concept_f1_gene_untrained` |
| Gene-level readout AUROC (hidden state) | `readout_auroc_hidden` | `readout_auroc_sae`, `readout_auroc_expression_control` (best of the expression controls, named in `readout_expression_control`), `readout_auroc_random` |
| GSI excess | `gsi_excess_sae` | `gsi_excess_raw`, `gsi_excess_genes` |
| MHR | `mhr_sae` | `mhr_raw`, `mhr_genes`, `mhr_chance` (also `mhr_null_*`) |
| Retained features | `n_features_retained` | `min_features` |

Further columns: `hidden_dim`, `tis_mean`, `sae_lambda`, `sae_fvu`, `sae_l0_frac`, `sae_dead_frac`, `n_concepts_row`, `n_concepts_gene`, `readout_auroc_hidden_sd`, `readout_auroc_untrained`, `n_concepts_kept`, `enough_features`, `retained_test_f1_mean`, `gsi_excess_sae_retained`, `mhr_k`, `specificity_n_cells`, `specificity_n_populations`. Missing upstream values are empty.

## Parameters (`stages.layers`)

| key | default | meaning |
|---|---|---|
| `readout_scaling` | `standardize` | which gene-readout probe column to report (`standardize` or `center`) |

## Outputs

```
09_layers/
  layer_table.csv         one row per layer
  recommendations.json    per criterion: best_layer, value, baselines at that layer, per_layer values,
                          representative_value / representative_is_best; best_layer_counts; agreement
  notes.txt               one line per criterion ("TIS gain: best at layer_1.norm2 (0.34; baselines: ...)")
```

Python: `interpretfm.layers` (`layer_table`, `recommendations`, `notes_text`, `CRITERIA`).
