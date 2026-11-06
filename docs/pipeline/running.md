# Running a pipeline

:::{note}
In development. The command, the configuration, the runner and the stages `ingest`, `tis`, `sae`, `concepts`, `concept_eval`, `gene_readout`, `cells`, `layers` and `report` work; `extract` runs from a cached activation store (`stages.extract.store_root`) or extracts scGPT and Geneformer activations in the model's own environment ([stage extract](stage_extract.md); C2S-Scale follows), and the spatial stages (`niches`, the calibrated interface test `interfaces` and the model-free ligand–receptor screen `lr_screen`) are skipped for a dataset without coordinates. `interpretfm.testing.write_toy_run(root, store=True)` writes a toy dataset, a synthetic activation store and a configuration that runs every implemented stage on a CPU in seconds. The 0.9 module entry points (`python -m interp_pipeline....`, [usage](../usage.md)) are unchanged.
:::

## The command

```bash
interpretfm validate config.yaml                 # schema, then referenced files
interpretfm run config.yaml --dry-run            # what would run, without writing anything
interpretfm run config.yaml                      # all stages, cached and resumable
interpretfm run config.yaml --stages ingest,extract,tis
interpretfm run config.yaml --stages tis --force # re-run even when cached
interpretfm status runs/cosmx_scgpt              # one line per stage from the manifests
interpretfm report runs/cosmx_scgpt              # (re)build report.json and report.html from the finished stages
interpretfm version
```

`python -m interpretfm.cli ...` is equivalent when the console script is not on your `PATH`.

Exit codes: 0 success; 1 a stage failed (its `manifest.json` holds the error); 2 invalid configuration or not a run directory; 3 valid configuration but missing files.

## Stages

| # | Stage | Inputs | Runs in | Declared output |
|---|---|---|---|---|
| 1 | `ingest` | – | core | `dataset_qc.json`, `cells.csv` (+ `vocab_coverage.csv`, `coords_um.csv`) |
| 2 | `extract` | ingest | **model environment** | `store.json` + the activation store |
| 3 | `tis` | extract | core | `tis.json`, `layers.csv` |
| 4 | `sae` | extract | core | `selected.json` |
| 5 | `concepts` | ingest | core | `universe.json`, `genes.csv` |
| 6 | `concept_eval` | extract, sae, concepts | core | `concept_f1.json`, `concept_f1.csv` |
| 7 | `gene_readout` | extract, sae, concepts | core | `gene_readout.json`, `gene_readout.csv`, `per_concept.csv.gz` |
| 8 | `cells` | ingest, extract, sae, concepts, concept_eval | core | `selection.json`, `specificity.json` |
| 9 | `layers` | tis, sae, concept_eval, gene_readout, cells | core | `layer_table.csv`, `recommendations.json` |
| 10 | `niches` | ingest, cells | core | `selected.json`; skipped without coordinates |
| 11 | `interfaces` | ingest, extract, cells, niches | core | `summary.json`; skipped without coordinates |
| 12 | `lr_screen` | ingest, niches | core | `summary.json`, `screen.csv`; skipped without coordinates |
| 13 | `report` | ingest, layers | core | `report.json` (+ `report.html`) |

Stages run in this order. With `--stages`, every selected stage still needs its inputs to have completed in the run directory (in this invocation or an earlier one); otherwise the run stops with a message naming the missing stage.

A stage is **skipped as cached** when its directory holds a `manifest.json` with status `done`, the same configuration hash and all declared outputs present. It is **skipped as not applicable** (status `skipped`) when its predicate is false for the configuration, for example `niches` without `dataset.spatial`. A failing stage writes a `failed` manifest with the traceback and stops the run; fix the cause and run again, and the finished stages are reused.

## The model environment

`extract` imports the model's code, which needs that model's software stack. The runner therefore starts it as a child process in the environment named by `model.env`: `conda run -n <name> python -m interpretfm.cli extract --config <run_dir>/config.resolved.yaml --run-dir <run_dir> --model <name> --layers <a,b>` (or `conda run -p <prefix>`, the interpreter in `env.python`, or `apptainer exec --nv` / `docker run --gpus all` with `env.container`). The child's output goes to `<stage dir>/log.txt`; the child writes the stage's manifest and, per layer, a manifest with the interpreter, the torch/CUDA versions and the installed packages it ran with. The package must be importable in that environment (`python -m pip install --no-deps -e .` inside it, or `PYTHONPATH=<repo>/src`, which the child inherits). Two cases stay in the current interpreter: `model.env: {}` (one environment for everything) and a cached store (`stages.extract.store_root`), which needs no model code. A missing environment, interpreter, conda or container runtime is reported before anything runs: for example `model.env.conda: conda environment 'scgpt' does not exist (known: [...])`. `conda` is found through `CONDA_EXE` (set by `conda activate`) or `PATH`; set `INTERPRETFM_CONDA` to choose a binary. Details, the adapters and the options: [stage extract](stage_extract.md).

## Toy example

```python
from interpretfm.testing import write_toy_run
cfg_path = write_toy_run("example")     # example/data/toy.h5ad + example/toy.yaml (300 cells, 50 genes, coordinates in 0.5 µm units)
```

```bash
interpretfm validate example/toy.yaml
interpretfm run example/toy.yaml --dry-run
interpretfm run example/toy.yaml --stages ingest
interpretfm status example/runs/toy
```

With `write_toy_run("example", store=True)` (a synthetic activation store and concept table) the whole pipeline runs on a CPU in well under a minute:

```bash
interpretfm run example/toy.yaml
xdg-open example/runs/toy/13_report/report.html
```

`configs/toy.yaml` is the same configuration, for a dataset written to `data/toy.h5ad` at the repository root.

## Python API

```python
from interpretfm import load_config, run
cfg = load_config("config.yaml")
rows = run(cfg, stages=["ingest", "extract"], force=False, dry_run=False)   # one dict per stage with its status
```

Implementing a stage (for contributors): register a function that receives a `StageContext` and writes its declared outputs into `ctx.stage_dir`.

```python
from interpretfm.runner import register_stage, StageContext

@register_stage("tis", inputs=["extract"], outputs=["tis.json"], replace=True)
def run_tis(ctx: StageContext):
    params = ctx.params                       # config.stages.tis
    store_dir = ctx.input_dir("extract")      # <run_dir>/02_extract
    ...
    ctx.write_json("tis.json", result)
    return {"n_layers": len(ctx.config.model.layers)}   # recorded in manifest["extra"]
```

`env="model"` on the decorator makes the runner execute the stage in the model environment; `applies=lambda cfg: ...` makes it skippable.
