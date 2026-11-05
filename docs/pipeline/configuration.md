# Configuration reference

:::{note}
The v1 pipeline is in development. The configuration schema below is implemented and validated; most stages are stubs until v1.0 (see [running a pipeline](running.md)).
:::

A run is described by one YAML file with five sections. Relative paths are resolved against the directory of the file. `interpretfm validate config.yaml` checks the schema and reports every problem at once, then checks that the referenced files exist.

```yaml
run:
  name: cosmx_scgpt
  out_dir: runs/cosmx_scgpt       # the run directory (created)
  seed: 0

dataset:
  h5ad: data/cosmx_lung.h5ad
  counts_layer: null              # null: X holds raw counts; otherwise a key of adata.layers
  gene_id: {column: index, kind: symbol}     # column: index or a var column; kind: symbol | ensembl
  labels: {column: cell_type}                # optional; groups: a CSV mapping labels to coarser groups
  spatial: {obsm: spatial, um_per_unit: 0.18}   # optional; omit for scRNA-seq data
  qc: {min_transcripts: 20, min_genes: 10}

model:
  name: scgpt                     # scgpt | geneformer | c2s_scale
  checkpoint: /path/to/weights
  env: {conda: interpretfm}       # where model code runs; see below
  layers: [layer_0.norm2, layer_1.norm2]
  cell_pooling: {kind: token, token: "60695"}   # kind: token | mean
  special_tokens: ["60694", "60695"]
  representative_layer: layer_1.norm2          # optional; must be one of layers

stages:                           # parameters per stage; any stage may be omitted
  extract: {batch_size: 64, rows_per_shard: 8192, dtype: float16}   # or store_root: <cached store>
  tis: {trials: 800, k_ref: 32, controls: [shuffle, pca]}
  sae: {width_multiplier: 8, lambda: {grid: [0.1, 0.3, 1, 3, 10], rule: {l0_max: 0.02, dead_max: 0.25}}}
  concepts: {sources: [GO:BP, GO:MF, GO:CC, REAC, KEGG], fdr: 0.05, min_overlap: 2}
  concept_eval: {thresholds: [0, 0.15, 0.3, 0.6], units: [row, gene], representations: [sae, raw, untrained]}
  gene_readout: {n_splits: 5, C: 1.0, scaling: [standardize, center]}
  cells: {}
  layers: {}
  spatial: {}
  report: {}

output:
  html: true
  json: true
  keep_store: true                # keep the activation store after the run
```

## Sections

`run`
: `name` (string, required), `out_dir` (path, required), `seed` (integer ≥ 0, default 0).

`dataset`
: `h5ad` (required). `gene_id.kind` is `symbol` or `ensembl`. `labels` is optional; without it the label-dependent analyses are skipped. `spatial` is optional: when it is present, `um_per_unit` is **required** and must be positive. It is the number of micrometres in one unit of the coordinates stored in `adata.obsm[<obsm>]` (for example `0.18` for CosMx pixels, `1.0` when the coordinates are already in µm). Every length in the pipeline is expressed in µm; the conversion happens once, in `interpretfm.units`, and all radii in stage parameters are given in µm.

`model`
: `layers` (non-empty list of layer names as used by the activation store). `env` selects the environment in which the model's code runs: `{conda: <name>}`, `{prefix: <env directory>}`, `{python: <interpreter>}`, or `{}` for the current interpreter. Exactly one key may be set. `{container: <image>}` is reserved for the container release and is rejected for now. One model per configuration: write one file per model and share the `dataset` section between them.

`stages`
: A mapping per stage name (`extract`, `tis`, `sae`, `concepts`, `concept_eval`, `gene_readout`, `cells`, `layers`, `spatial`, `report`); `ingest` takes its options from `dataset`. Unknown stage names are errors; the keys inside a stage section are defined by that stage and are not checked by the schema. The keys of the model-level stages are listed on their pages: [tis](stage_tis.md), [sae](stage_sae.md), [concepts](stage_concepts.md), [concept_eval](stage_concept_eval.md), [gene_readout](stage_gene_readout.md). Paths inside a stage section (`store_root`, `offline_matrix`, `checkpoints`, `token_vocab`) are taken as absolute or relative to the configuration file; a stage that runs in the model environment reads the resolved copy in the run directory, so prefer absolute paths there.

  Every stage that reads activations accepts `store_root`, an existing activation store (the sharded 0.9 format); `stages.extract.store_root` registers such a store for the whole run without running a model.

`output`
: `html`, `json`, `keep_store` (booleans).

## Placeholders

Any string may contain `${NAME}`. When the file is loaded the placeholder is replaced from (highest priority first) a mapping passed to `load_config(env=...)`, the process environment, and a `KEY=VALUE` file given with `--env-file` (or `load_config(env_file=...)`). A placeholder without a value stops with an error naming the variable and the keys that use it. The paper configurations in `configs/paper/` refer to the dataset, the activation stores, the model vocabularies and the cached intermediates only through such placeholders ([reproducing the manuscript](../reproducing.md)).

```yaml
dataset: {h5ad: "${INTERPRETFM_DATA}/cosmx_human_lung_sec8.h5ad", ...}
stages: {extract: {store_root: "${INTERPRETFM_STORES}/acts_extraction_scgpt_cosmx"}}
```

## Caching and the stage hash

The runner hashes, for each stage, the stage's own section together with `dataset`, `model` and `run.seed`. A cached stage result is reused only when this hash is unchanged; editing another stage's section, `run.name` or `out_dir` does not invalidate it. The resolved configuration (absolute paths, defaults filled in) is written to `<out_dir>/config.resolved.yaml` at every run.

## Python API

```python
from interpretfm.config import load_config, parse_config, ConfigError

cfg = load_config("config.yaml")      # raises ConfigError listing every problem
cfg.dataset.spatial.um_per_unit        # typed access
cfg.stage_params("tis")                # {'trials': 800, 'k_ref': 32}
cfg.stage_hash("tis")                  # the cache key of the stage
cfg.to_dict()                          # plain mapping (paths as strings)
```
