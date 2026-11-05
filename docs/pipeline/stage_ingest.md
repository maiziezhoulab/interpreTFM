# Stage `ingest`: input checks, QC flags, vocabulary coverage, units

Directory `01_ingest/`. No inputs; core environment; seconds.

## What it checks

The stage validates the AnnData file against the `dataset` section and stops with an error naming the configuration key when a required piece is missing or wrong:

- **raw counts**: the matrix declared as counts (`X`, or `adata.layers[counts_layer]`) must be finite, non-negative and integer-valued (checked on the first 5,000 cells). Normalized or log-transformed data are rejected with a message pointing at `dataset.counts_layer`.
- **gene names**: the names from `dataset.gene_id` must be non-empty and unique; a warning is recorded when they do not look like the declared `kind` (`symbol` / `ensembl`).
- **labels**: `dataset.labels.column` must exist in `obs` when declared. Without labels the population-specificity metrics and the label-based summaries are skipped (recorded as a warning).
- **coordinates**: `dataset.spatial.obsm` must exist and be `n_cells × 2` (or more) when `dataset.spatial` is declared, and `um_per_unit` is required by the schema. Coordinates are converted to micrometres once ([units](configuration.md#sections)) and written to `coords_um.csv` and `cells.csv`.

## QC flags

Cells with fewer than `dataset.qc.min_transcripts` transcripts (default 20) or fewer than `dataset.qc.min_genes` detected genes (default 10) are **flagged**, not removed: `cells.csv` carries `qc_pass`, the `cells` stage keeps the flag in `cells.h5ad` and reports how many flagged cells entered the specificity metrics (`stages.cells.specificity.exclude_qc_flagged` excludes them). The counts are in `dataset_qc.json["qc"]` and in the report.

## Vocabulary coverage

The share of panel genes present in the model's gene vocabulary, matched by key, symbol or Ensembl id (case-insensitive). The vocabulary is looked up in this order: `model.vocab` (a JSON object `{name: id}`, `{id: name}` or a list of names, for example scGPT's `vocab.json`), `<model.checkpoint>/vocab.json`, `stages.extract.token_vocab`, `stages.concept_eval.token_vocab`, `stages.gene_readout.token_vocab`, `<stages.extract.store_root>/token_vocab.json`. When none exists the coverage is reported as not computed, with the reason. Per-gene results are in `vocab_coverage.csv`; the summary (with the first missing genes) in `dataset_qc.json["vocab_coverage"]` and the stage manifest.

## Outputs

```
01_ingest/
  dataset_qc.json       n_cells, n_genes, counts check, transcripts/genes per cell, qc (thresholds, n_pass, n_flagged),
                        genes (kind, symbols/Ensembl available, never-detected genes), vocab_coverage, labels, spatial, warnings
  cells.csv             cell, n_transcripts, n_genes, qc_pass, label, x_um, y_um
  vocab_coverage.csv    key, symbol, ensembl, in_vocab, matched_name       (when a vocabulary was found)
  coords_um.csv         cell_index, x_um, y_um                             (spatial data)
```

Python: `interpretfm.ingest` (`check_raw_counts`, `qc_flags`, `find_vocab`, `vocab_coverage`).
