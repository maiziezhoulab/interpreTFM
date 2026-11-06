# Stage `report`: the run report

Directory `12_report/`, written by the `report` stage (inputs: `ingest`, `layers`) or rebuilt at any time with `interpretfm report <run_dir>` from the resolved configuration and the finished stages.

## Files

- `report.json`: everything the page shows, in one JSON-serialisable mapping: `run`, `model`, `dataset` (checks, QC flags, vocabulary coverage, labels, coordinates), `stages` (status and duration from the manifests), `spatial` (status and note of the spatial stages), `niches` (per layer: the niche numbers beside the expression-only, label-composition and tile-null baselines; [niches](stage_niches.md)), `layer_table` (the rows of `09_layers/layer_table.csv`), `recommendations`, `metrics` (one entry per layer × criterion with `value` and `baselines`), `sae` (λ and QC per layer), `cells` (retention and specificity per layer) and `how_to_read`.
- `report.html` (with `output.html: true`): one self-contained page, inline CSS, figures embedded as PNG (matplotlib, Agg backend), no external resources, readable offline.

## What the page contains

1. **How to read this report**: the two held-out units (gene detection on held-out rows versus concept generalization on held-out genes), the cell-level comparison with single genes (same cells, shuffles and split; MHR chance level), that features carry the measured panel's information (concept labels describe what is observable on the panel), and that layers are compared per criterion without a composite.
2. **Dataset and input checks**: cells, genes, mode, QC-flagged cells (kept and flagged), vocabulary coverage, labels, coordinate unit, and the ingest warnings.
3. **Layers**: the per-criterion notes and one table per criterion with the baselines in grey next to the value, followed by the figures (TIS with its controls; concept F1 under both splits for SAE, raw dimensions and the untrained SAE; gene-level readout with the expression and random controls; GSI excess and MHR for SAE features, raw dimensions and single genes with the chance line; retained features with the minimum).
4. **Sparse autoencoders**: λ, rule, width, FVU, L0 and dead fraction per layer.
5. **Cell-level features**: retention counts and test F1 per layer; GSI excess and MHR per representation, with the chance level, the random-feature null and the number of QC-flagged cells used.
6. **Spatial analyses**: `niches: run`, `skipped` (scRNA-seq mode) or `not run`, and the niche table (selected configuration, ARI/NMI, seed ARI, purity with its tile null, the expression-only winner and the expression at the model's configuration, label composition, seed stability, top concepts).
7. **Stages**: status, duration and the first error line per stage.

## Rebuilding

```bash
interpretfm report runs/cosmx_scgpt      # exit 0: report.json (+ report.html); 1: a stage is missing; 2: not a run directory
```

Python: `interpretfm.report.write_report(cfg, run_dir, out_dir, html_page=True)`; `collect`, `make_figures`, `render_html` for the pieces.
