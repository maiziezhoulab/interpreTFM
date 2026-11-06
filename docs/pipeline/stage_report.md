# Stage `report`: the run report

Directory `13_report/`, written by the `report` stage (inputs: `ingest`, `layers`) or rebuilt at any time with `interpretfm report <run_dir>` from the resolved configuration and the finished stages. Every finished stage is read; a stage that did not run, or was skipped (scRNA-seq mode), renders as "not run" or "skipped" without an error. `stages.report.spatial_only: true` drops the requirement on `layers`, for a run of the spatial stages alone on published projected features (`cells_h5ad` of `niches` and `interfaces`; the [tutorial](../tutorials/cosmx_crop.md) does this): the layer, SAE and cell sections then render as "not run".

## Files

- `report.json`: everything the page shows, in one JSON-serialisable mapping: `run`, `model`, `dataset` (checks, QC flags, vocabulary coverage, labels, coordinates), `stages` (status and duration from the manifests), `spatial` (status and note of the spatial stages), `layer_table` (the rows of `09_layers/layer_table.csv`), `recommendations`, `metrics` (one entry per layer × criterion with `value` and `baselines`), `sae` (λ and QC per layer), `cells` (retention and specificity per layer), `niches` (per layer: the niche numbers beside the expression-only, label-composition and tile-null baselines, seed stability, top concepts; [niches](stage_niches.md)), `interfaces` (per layer: the shares of |z| > 2 under the legacy tile null and the calibrated null, z*, false-positive shares, calls, effect sizes; [interfaces](stage_interfaces.md)), `lr_screen` (radius with its neighbourhood size, pairs tested and hits per threshold, robust and depleted pairs, the hits' statistics, the per-pair fold / z / q rows and the niche placement per layer; [lr_screen](stage_lr_screen.md)), `how_to_read` and `section_notes`.
- `report.html` (with `output.html: true`): one self-contained page rendered from the Jinja2 template `interpretfm/report/templates/report.html.j2`, inline CSS, figures embedded as PNG (matplotlib, Agg backend), no external resources, readable offline.

## What the page contains

Every section opens with a short note on what its numbers measure and what they do not (`section_notes` in `report.json`), and every number sits next to its baseline (baselines in grey). The wording is neutral: no verdicts, no composite score.

1. **How to read this report**: the two held-out units (gene detection on held-out rows versus concept generalization on held-out genes), the cell-level comparison with single genes (same cells, shuffles and split; MHR chance level), that features carry the measured panel's information, that layers are compared per criterion without a composite, and how the spatial sections relate to their baselines and to the model-free screen.
2. **Dataset and input checks**: cells, genes, mode, QC-flagged cells (kept and flagged), vocabulary coverage, labels, coordinate unit, and the ingest warnings.
3. **Layers**: the per-criterion notes and one table per criterion with the baselines next to the value, then the figures: TIS with its shuffled and PCA controls; concept F1 under both splits for the sparse SAE, raw dimensions and the untrained SAE; the gene-level readout with its expression and random controls (chance line); GSI excess and MHR for SAE features, raw dimensions and single genes (chance line); retained features with the minimum.
4. **Sparse autoencoders**: λ, rule, width, FVU, L0 and dead fraction per layer.
5. **Cell-level features** (cell specificity): retention counts and test F1 per layer; GSI excess and MHR per representation with the chance level, the random-feature null and the number of QC-flagged cells used.
6. **Spatial analyses**, each with its baselines beside it:
   - **Niches**: selected configuration, ARI/NMI against the labels and seed ARI, purity with its tile null and z, the expression-only winner and the expression at the model's configuration, the label-composition partition, seed stability (model versus expression-only), top concepts; figures of ARI/NMI against the three baselines and of the seed stability.
   - **Interfaces**: pairs (niche / niche-free), the share of |z| > 2 under the calibrated null versus the legacy tile null, z*, the false-positive shares (smooth fields under both nulls, i.i.d. noise), enriched / depleted calls (niche and niche-free), median / maximum effect, neighbour correlations; a figure of the shares and the false-positive rates with the 5 % line.
   - **Ligand–receptor screen**: radius in both units with the mean number of other cells within it (and the matching target when `match_neighbors_to` was used), pairs tested and hits at every threshold, robust and depleted pairs, a table of the hits (positive cells, observed count, fold, z, q at each threshold, robust flag, top cell-type pairs), the niche placement ratios per layer; a fold-versus-z figure of all tested pairs with hits and robust hits marked, and a placement figure per layer.
7. **Stages**: status, duration and the first error line per stage.

## Rebuilding

```bash
interpretfm report runs/cosmx_scgpt      # exit 0: report.json (+ report.html); 1: a stage is missing; 2: not a run directory
```

Python: `interpretfm.report.write_report(cfg, run_dir, out_dir, html_page=True)`; `collect` (`interpretfm.report.metrics`) for the JSON, `make_figures` and `render_html` (`interpretfm.report.html`) for the page.
