# Stage `extract`: activations into the sharded store, across the model-environment boundary

Directory `02_extract/`. Input: `ingest`. Model environment (`model.env`) for a fresh extraction, core environment for a cached store. Minutes per layer on a GPU for a targeted spatial panel (about 1,000 cells per second for scGPT, 70 for Geneformer, 15 for C2S-Scale); the tokenization of the dataset is included.

## Two modes

| | `stages.extract.store_root` set | not set |
|---|---|---|
| what runs | nothing model-related: the existing store (the sharded 0.9 format, `<root>/activations/<layer>/shard_*/`) is checked for every configured layer and registered in `store.json` | the adapter of `model.name` loads `model.checkpoint`, tokenizes the dataset and writes `02_extract/store/activations/<layer>/shard_<i>/{activations.pt,index.pt,metadata.json}` |
| where | the current interpreter | the environment of `model.env`, as a child process; in the current interpreter when `model.env` is `{}` |
| used for | the paper configurations (`configs/paper/*_layer*.yaml`), reusing an earlier extraction | new data, new layers, the fresh-extraction checks (`configs/paper/extract_*.yaml`) |

## The boundary

scGPT, Geneformer and C2S-Scale need conflicting library versions ([environments](../installation.md)), so the runner starts the stage in the model's environment:

```
conda run -n <env> python -m interpretfm.cli extract --config <run_dir>/config.resolved.yaml --run-dir <run_dir> --model <name> --layers <a,b>
```

`model.env` names that environment: `{conda: <name>}`, `{prefix: <env directory>}`, `{python: <interpreter>}`, `{container: <image>}` (see below) or `{}` for the current interpreter (one environment holds everything: the stage then runs in-process, no subprocess). The child inherits the environment variables (`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `PYTHONPATH`), checks `--model` and `--layers` against the resolved configuration, runs the same stage function as an in-process stage, and writes:

- the store, as above;
- `02_extract/<layer>/manifest.json`: rows, cells, shards, hidden size, dtype, unique tokens, special-token rows, how many cells carry the pooling token, the adapter's summary (vocabulary coverage, model version, tokenization settings, timing), the cell subset, and the child's `environment`: interpreter, platform, `torch` version with the CUDA / cuDNN versions and the visible GPUs, and every installed distribution (`name==version`, what `pip freeze` reports);
- `store.json`, which the later stages read.

The parent never imports model code; it checks the stage manifest the child wrote and records how the environment was resolved (`model_env` in the stage manifest). The child's output is in `02_extract/log.txt`. The package must be importable in the model environment: install it there (`python -m pip install --no-deps -e .` inside the environment) or export `PYTHONPATH=<repo>/src` before running.

Errors are reported before anything heavy loads, naming the configuration key: a missing environment or interpreter (`model.env.conda: conda environment 'x' does not exist (known: [...])`), missing weights (`model.checkpoint is required to extract scgpt activations`, `model.checkpoint: <dir> must contain best_model.pt`), a layer the model does not have (`model.layers: 'layer_12.norm2' does not exist; scgpt has layers 0..11`), no panel gene in the vocabulary, and a missing GPU (`stages.extract.device: cuda requested but no GPU is visible to this process (CUDA_VISIBLE_DEVICES=...)`; set `device: cpu` to run on the CPU, slowly).

### Containers

`{container: <image>}` runs the child in a container: `apptainer exec --nv --bind <paths> <image> python -m interpretfm.cli extract ...` (also `singularity`), or `docker run --rm --gpus all -v <path>:<path> ... <image> python ...` (`podman` with `--device nvidia.com/gpu=all`). The runtime is the first of `apptainer`, `singularity`, `docker`, `podman` on `PATH`, or the one named by `INTERPRETFM_CONTAINER_RUNTIME`. The run directory, the dataset, the labels files, the weights, the vocabulary, the paths named in `stages.extract` and the configuration's directory are bound into the container at their host paths, so the resolved configuration works unchanged inside. The images are built from `containers/` (one per model environment); **neither the image builds nor the container path have been exercised on the development machine, which has no container runtime** ([containers/README.md](https://github.com/OliiverHu/interpreTFM/blob/HEAD/containers/README.md)).

## Models

| `model.name` | adapter | gene key | tokens stored per cell | layer names | environment |
|---|---|---|---|---|---|
| `scgpt` | vendored scGPT whole-human model (`checkpoint/` with `best_model.pt` and `vocab.json`) | symbols (`dataset.gene_id.kind: symbol`, or an Ensembl-keyed panel with a symbol column: `gene_keys.symbol_column`, default `feature_name`/`gene_symbol`/...) | `<cls>` + the expressed genes in panel order, `log1p` of raw counts, at most `max_length` (512) tokens; padding is not stored | `layer_<i>.norm2` (= `layer_<i>`): output of encoder block *i* | `envs/interpretfm.yml` |
| `geneformer` | Hugging Face `BertForMaskedLM` (a model folder with `config.json` + weights, e.g. the top level of a [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) clone) and the installed `geneformer` package's tokenizer and dictionaries | Ensembl ids (`kind: ensembl`, or symbols mapped through the package's gene-name dictionary) | the package's rank-value encoding (median-normalized expression, descending), with `<cls>` and `<eos>` for V2 models; genes outside the token dictionary are left out and listed in the manifest | `layer_<i>`: output of transformer block *i* | `envs/geneformer.yml` |
| `toy` | a synthetic model for tests and the tutorial (no weights, no GPU) | symbols | one classification row + one row per detected gene, 16 dimensions | any `layer_<i>[.norm2]` | any |
| `c2s_scale` | Hugging Face Gemma-2 causal language model ([vandijklab/C2S-Scale-Gemma-2-2B](https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-2B), a local model folder with `config.json`) through the vendored cell2sentence helpers | symbols, upper-cased (`kind: symbol`, or `gene_keys.symbol_column`) | one row per gene of the cell sentence (the detected genes by descending count, at most `max_genes` = 256), the hidden state of the gene's last sub-token; see below | `layer_<i>`: output of decoder block *i* | `envs/c2s.yml` |

The three real adapters run the model in evaluation mode under `torch.no_grad` in float32 (C2S-Scale: `model_dtype`, float32 by default), in dataset order, `batch_size` cells per forward pass; only the storage dtype changes with `dtype`. The vocabulary coverage of the panel is in the manifest (`check.coverage`) and, before extraction, in the `ingest` stage's `vocab_coverage.csv`.

### C2S-Scale: cell sentences

C2S-Scale reads text. The adapter builds, for every cell, the cell2sentence *cell sentence*: the detected genes (count > 0) ordered by descending count, upper-cased symbols separated by spaces, the first `max_genes` of them. Ties (equal counts, frequent on a targeted panel) are broken as the vendored cell2sentence code does, by a shuffle seeded with `tie_seed` (42) plus the cell's position within its extraction shard; so that a cell's sentence does not change with `cell_subset`, the shard and position of every cell are taken from the whole dataset before the subset is applied. Per-cell normalisation is monotonic within a cell and would not change the order, so counts are ranked as they are. The sentence is embedded in the cell2sentence cell-type-prediction prompt (the first template of the vendored `prompts/single_cell_cell_type_prediction_prompts.json`, with `num_genes` = `max_genes` and the organism from `obs["organism"]`, else `human`; the manifest records the template and the organisms); the model attends to the prompt before the genes, so the template is part of the representation. Sequences are left-padded to the batch; for every gene the hidden state at its last sub-token (`token_pooling: last`; `mean`/`max` over the sub-tokens are available) is pooled in float32 and stored in `dtype` (float16 for the manuscript's stores), one row per gene, token id = the symbol; there is no cell-level token (`model.cell_pooling: {kind: mean}`). Cells without a detected gene get no rows.

Shards: `shard_column` names an `obs` column that assigns every cell to a shard (`shard_<k>` or any label; the manuscript's stores used the dataset's random `shards` column, 60 shards), and cells keep their dataset order within a shard; without it, consecutive blocks of `cells_per_shard` cells (derived from `rows_per_shard` and the median number of detected genes). With `shard_column` and no subset, the batches are the manuscript's and the rows come out bit-identical; a subset changes the batch composition (the padded length), which moves a few float16 steps (below).

## Cell-level pooling

How a cell-level vector is read from the token rows is declared once, in `model.cell_pooling`: `{kind: token, token: "<id>"}` takes the row whose token id is `<id>` (scGPT's `<cls>` is vocabulary id `60695`, Geneformer's is `<cls>`), `{kind: mean}` averages a cell's rows outside `model.special_tokens`. Every later stage reads it from there (`interpretfm.stores.cell_matrix`); the extract stage records it in `store.json` and warns when a cell lacks the pooling token.

## Parameters (`stages.extract`)

| key | default | meaning |
|---|---|---|
| `store_root` | – | an existing store; nothing is extracted |
| `batch_size` | 8 | cells per forward pass |
| `max_length` | 512 | maximum tokens per cell (scGPT; the panel order is truncated beyond it) |
| `dtype` | `float32` | storage dtype of the rows: `float32` or `float16` (the manuscript's Geneformer and C2S-Scale stores) |
| `rows_per_shard` | 200000 | approximate token rows per shard (Geneformer: whole cells, `cells_per_shard` derived from it) |
| `device` | `cuda` | `cuda`, `cuda:<n>` or `cpu` |
| `cell_subset` | – | a number of cells (a seeded draw with `run.seed`, kept in dataset order) or a file of cell ids (one per line, or a CSV with a `cell` column); the ids used go to `cell_subset.txt`. Later stages work on the subset (cells without rows are reported as such) |
| `gene_keys` | the `concepts` stage's keys | `{symbol_column, ensembl_column}` of `var` for the model's vocabulary kind |
| `scgpt` | `{}` | (no options) |
| `geneformer` | `{model_version: V2, nproc: 1}` | tokenizer version (`V1` for the 30M-cell models and their dictionaries), tokenizer processes, `cells_per_shard` |
| `toy` | `{hidden_dim: 16, concepts: true}` | |
| `c2s_scale` | `{max_genes: 256, token_pooling: last, model_dtype: float32, tie_seed: 42}` | genes per sentence, sub-token pooling per gene (`last`, `mean`, `max`), the model's parameter dtype (`float32`, `bfloat16`, `float16`), the tie-breaking seed; `shard_column` (an `obs` column of shard labels), `cells_per_shard` (without a column), `organism` (overrides `obs["organism"]`) |

## Checking a fresh extraction against an existing store

`interpretfm.extract.compare_stores(new_root, ref_root, layer, cells, ref_exclude_tokens=[...])` aligns the two stores cell by cell and reports whether the token sequences are identical, the per-token cosine (min, 1st percentile, median), the maximum absolute difference (also relative to the reference's scale) and the per-row relative L2 difference; `compare_projections(..., checkpoint, cells, special_tokens, cells_h5ad=...)` encodes both stores' rows with an SAE, averages the latents per cell and compares the cell-level features (and the `cells` stage's `cells.h5ad` when given). The reproduction tests `tests/reproduction/test_extraction.py` and `test_extraction_c2s.py` (`INTERPRETFM_REPRO_GPU=1`) run `configs/paper/extract_cosmx_{scgpt,geneformer,c2s}.yaml` (the same 2,000 seeded CosMx cells each, in the model environments on a GPU) and require identical token sequences and a cosine of at least 0.999 for every token and every projected cell. On the manuscript's stores the three models give identical token sequences for all 2,000 cells, a minimum per-token cosine of 0.999999 and projected features with a per-cell cosine above 0.9999999; the remaining differences (scGPT 7e-6 absolute in float32, Geneformer one float16 step, 0.0078 at magnitude 8–16, C2S-Scale one float16 step, 0.125 at magnitude 128–256 on a scale of 182, with a float32 model) come from batch composition, because the padded length of a batch changes the floating-point summation order; C2S-Scale cells re-extracted in their original batches are bit-identical. The manuscript's scGPT store also holds the padding rows of each batch (token id `60694`), which this stage no longer writes; `ref_exclude_tokens=["60694"]` skips them in the comparison and every later stage ignores them through `model.special_tokens`.

## Outputs

```
02_extract/
  store.json                 root, format, layers, source (extracted | cached), model, n_shards, n_cells, cell_pooling, special_tokens, dtype, cell_subset
  store/activations/<layer>/shard_<i>/{activations.pt, index.pt, metadata.json}
  <layer>/manifest.json      counts, adapter summary, check (coverage), extraction parameters, cell subset, environment (python, torch/CUDA, packages)
  cell_subset.txt            the cell ids extracted (subset runs)
  work/                      scratch of the adapter (Geneformer: prepared h5ad and tokenized dataset; C2S-Scale: hf_cache/)
  log.txt                    the child's output
```

Python: `interpretfm.extract` (`spec_from_config`, `select_cells`, `layer_manifest`, `compare_stores`, `compare_projections`), `interpretfm.models` (`get_extractor`, `ADAPTERS`), `interpretfm.envs` (`resolve_env`, `run_in_env`, `container_command`, `child_environment`).
