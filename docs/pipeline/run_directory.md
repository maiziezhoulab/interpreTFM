# Run directory and manifests

One directory per configuration (`run.out_dir`):

```
runs/<name>/
  config.resolved.yaml   the validated configuration: absolute paths, defaults filled in
  manifest.json          run-level manifest (below)
  log.txt                run log (appended at every invocation)
  01_ingest/             one directory per stage, numbered in pipeline order
    manifest.json
    log.txt
    dataset_qc.json      the stage's outputs
    cells.csv            cell, n_transcripts, n_genes, qc_pass, label, x_um, y_um
    vocab_coverage.csv   when a vocabulary was found
    coords_um.csv        spatial data only: cell_index, x_um, y_um
  02_extract/<layer>/shard_*/   the activation store (unchanged 0.9 format), store.json
  03_tis/ 04_sae/ 05_concepts/ 06_concept_eval/ 07_gene_readout/ 08_cells/ 09_layers/ 10_niches/ 11_interfaces/ 12_lr_screen/ 13_report/
```

Stage directories are created when the stage first runs; `interpretfm status` lists every registered stage, with `-` for those not run yet. The contents of each stage directory are listed on the stage pages ([ingest](stage_ingest.md), [tis](stage_tis.md), [sae](stage_sae.md), [concepts](stage_concepts.md), [concept_eval](stage_concept_eval.md), [gene_readout](stage_gene_readout.md), [cells](stage_cells.md), [layers](stage_layers.md), [niches](stage_niches.md), [interfaces](stage_interfaces.md), [lr_screen](stage_lr_screen.md), [report](stage_report.md)).

## Stage manifest

`<stage dir>/manifest.json`, written when the stage starts (status `running`) and rewritten when it ends:

```json
{
  "stage": "ingest",
  "status": "done",
  "config_hash": "3f0c2c1a9b7e4d21",
  "package_version": "1.0.0",
  "inputs": {},
  "outputs": {"dataset_qc.json": "01_ingest/dataset_qc.json", "cells.csv": "01_ingest/cells.csv"},
  "environment": {"python": "3.11.9", "executable": "...", "platform": "...", "hostname": "...",
                  "versions": {"interpretfm": "1.0.0", "numpy": "...", "torch": "...", "anndata": "..."}},
  "model_env": null,
  "started": "2026-11-05T19:58:38+00:00",
  "ended": "2026-11-05T19:58:38+00:00",
  "duration_s": 0.4,
  "error": null,
  "extra": {"n_cells": 300, "n_genes": 50, "n_flagged": 0, "vocab_coverage": 0.9, "spatial": true}
}
```

- `status`: `running`, `done`, `failed` (then `error` holds the message and traceback) or `skipped` (not applicable to the configuration).
- `config_hash`: `Config.stage_hash(stage)`, the cache key ([configuration](configuration.md#caching-and-the-stage-hash)).
- `inputs` / `outputs`: upstream stage directories and the declared output files, relative to the run directory.
- `environment`: the interpreter that ran the stage. For a stage run in the model environment, `environment` describes the child interpreter and `model_env` how the runner resolved `model.env` (kind, description, command).
- `extra`: a short, stage-specific summary returned by the stage function.

Timestamps are UTC, ISO 8601.

## Run manifest

`runs/<name>/manifest.json`: `name`, `config_hash` (of the whole configuration), `config_file` (the source YAML), `package_version`, `environment`, `created`, `updated`, `stages` (directory names in order) and `command` (the command line of the last invocation).

## Reading manifests from Python

```python
from interpretfm.manifest import StageManifest, RunManifest
from interpretfm.runner import status, stage_dir

m = StageManifest.read(stage_dir("runs/toy", "ingest"))   # None when the stage has not run
rows = status("runs/toy")                                  # what `interpretfm status` prints
```
