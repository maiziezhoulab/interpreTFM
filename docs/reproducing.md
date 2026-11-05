# Reproducing the manuscript

The configurations in `configs/paper/` hold, for each of the six representative layers, every parameter used for the manuscript (Methods; Supplementary Table S1). They start from the cached activation stores and the cached SAE checkpoints, so the model-level results are recomputed on a CPU without any model weights or GPU. The tests in `tests/reproduction/` compare the results with the published numbers and with the manuscript's cached intermediates.

| Configuration | Dataset | Model | Stored layer (display) |
|---|---|---|---|
| `cosmx_scgpt_layer2.yaml` (commented reference) | CosMx lung cancer | scGPT | `layer_1.norm2` (2) |
| `cosmx_geneformer_layer2.yaml` | CosMx lung cancer | Geneformer | `layer_1` (2) |
| `cosmx_c2s_layer14.yaml` | CosMx lung cancer | C2S-Scale | `layer_13` (14) |
| `xenium_scgpt_layer3.yaml` | Xenium RCC | scGPT | `layer_2.norm2` (3) |
| `xenium_geneformer_layer2.yaml` | Xenium RCC | Geneformer | `layer_1` (2) |
| `xenium_c2s_layer14.yaml` | Xenium RCC | C2S-Scale | `layer_13` (14) |

:::{note}
The reproduction data (datasets, activation stores of 24-46 GB per layer, SAE checkpoints and held-out tables) are not part of the repository. They will be deposited with the release; until then the configurations run on the machine that holds them. The cached intermediates let the tests also check the pure summary rules (λ selection, concept-F1 summary) in seconds, without a pipeline run.
:::

## Data layout

The configurations name data only through `${NAME}` placeholders ([configuration reference](pipeline/configuration.md#placeholders)). Copy `configs/paper/paths.env.example` to `configs/paper/paths.env` (ignored by git) and fill in the five locations, or export the variables in your shell:

| Variable | Expected content |
|---|---|
| `INTERPRETFM_DATA` | the datasets: `cosmx_human_lung_sec8.h5ad` (81,236 cells × 960 genes, raw counts in `X`, Ensembl `var_names`, symbols in `var["feature_name"]`, labels in `obs["author_cell_type"]`, pixel coordinates in `obsm["spatial"]`, 0.18 µm per pixel) and `xenium_rcc_union_filtered_nonzero.h5ad` (465,220 cells × 399 genes, symbol `var_names`, Ensembl ids in `var["gene_ids"]`, labels in `obs["cell_type"]`, coordinates in µm) |
| `INTERPRETFM_STORES` | the activation stores in the sharded 0.9 format, one directory per model and dataset: `acts_extraction_scgpt_cosmx`, `acts_extraction_geneformer_cosmx`, `acts_extraction_c2s_cosmx_store`, `acts_extraction_scgpt_xenium_kidney`, `acts_extraction_geneformer_xenium`, `acts_extraction_c2s_xenium_raw`, each with `activations/<layer>/shard_<i>/{activations.pt,index.pt,metadata.json}` (scGPT stores hold vocabulary ids as token ids; Geneformer stores Ensembl ids; C2S-Scale stores gene symbols) |
| `INTERPRETFM_MODELS` | model directories; only `scGPT/whole-human-pretrain/vocab.json` (the public scGPT whole-human checkpoint) is read, to translate scGPT token ids to gene symbols |
| `INTERPRETFM_PAPER_CACHE` | the manuscript's cached intermediates: `sae_v2/<dataset>/<model>/<layer>/sae/v2biased__steps_8000__lam_1/sae_<layer>_best.pt` (the selected SAE), `.../sae/v2randinit__steps_0/` (the untrained control), `.../sae_eval/v2biased__steps_8000__lam_<λ>/sae_eval.json` (QC of the five λ), `.../f1_heldout/heldout_<sparse|raw|randinit>__<gene_token_row|gene_identity>/` (held-out tables), `sae_v2/sparse_selection.json`, the concept tables `f1heldout_gene_token_row_cosmx_all_layers_l1_1e-3/gprofiler/` and `xenium_geneformer_c2sscale_strict_from_extract_to_immune/01_f1_heldout/gprofiler/` (`gprofiler_binary_gene_by_term.csv`, `gprofiler_terms.tsv`), `tis_layer_screen_3models_cosmx/<model>/<layer>/summary_row.json`, `revision_metrics/concept_f1_v2/summary_long.csv` and `revision_metrics/gene_split_ceiling/summary{,_center}.csv` |
| `INTERPRETFM_RUNS` | where the run directories are written (`${INTERPRETFM_RUNS}/<configuration name>/`) |

Symbolic links are fine when the stores live on several file systems. `<dataset>` is `cosmx` or `xenium`, `<model>` is `scgpt`, `geneformer` or `c2sscale`, and `<layer>` in the cache uses underscores (`layer_1_norm2`).

## Commands

```bash
make repro-validate                       # every paper configuration against the env file
make repro CONFIG=cosmx_scgpt_layer2      # ingest, extract (cached store), tis, sae, concepts, concept_eval, gene_readout
make repro-status CONFIG=cosmx_scgpt_layer2
make repro-test                           # pytest tests/reproduction -m reproduction -rA -s
```

`make repro` is `interpretfm run configs/paper/<name>.yaml --env-file configs/paper/paths.env --stages ...`; the run is cached and resumable, so it can be restarted after an interruption and re-run for a subset of stages. On a CPU the stages take minutes (`tis`, `sae`, `concepts`) to hours (`concept_eval`: six passes over the store; `gene_readout`: one pass plus the probes), so start it detached (for example in `tmux`) and run the tests afterwards. Setting `INTERPRETFM_REPRO_RUN=1` lets a test run the missing stages itself.

What each test checks, and the tolerance (AUDIT numbering of the acceptance tests):

| Test | Compares | Tolerance |
|---|---|---|
| T1 `test_t1_tis_gain` | TIS gain (Table S1), mean and shuffled TIS vs the cached screen; the PCA-rotated control | ±0.0005 (seeded, deterministic); PCA control ±0.005 (component signs depend on the scikit-learn version) |
| T2 `test_t2_lambda_rule_from_cache` | the λ rule on the cached QC of the five λ → λ = 1 | exact |
| T2 `test_t2_lambda_rule_and_untrained_baseline` | the `sae` stage: λ, FVU / L0 / dead-feature rate at λ = 1 vs the cached evaluation; the untrained SAE vs the cached control | ±0.005 / ±0.001; bit-identical |
| T3/T4 `test_t3_t4_concept_f1_summary_rule_from_cache` | the concept-F1 summary rule on the cached held-out tables → Table S1 (row and whole-gene split; sparse, raw, untrained) | ±0.0005 |
| T3/T4 `test_t3_t4_concept_f1` | concept F1 recomputed from the store, plus identical partitions (row counts, test genes) | ±0.005; counts ±2 |
| `test_concept_counts` | universe size, concepts scored per split, concepts marked (test F1 ≥ 0.5) | exact / ±2 |
| T5 `test_t5_gene_readout` | probe AUROC (standardized and center-only) and best single dimension vs Table S3, with the published ordering | ±0.03 (five whole-gene splits drawn with the stage's seeds; same rule, different draws) |
| T6-T13 | features kept, GSI/MHR, niches, interfaces, LR screen | placeholders until the `cells` and `spatial` stages land |

Every comparison prints one line `observed | manuscript | cached source | tolerance` (`-s`).

## Known differences from the manuscript runs

- Only the selected λ = 1 checkpoint of each layer is cached, so the `sae` stage of the paper configurations evaluates that checkpoint; the rule itself is tested on the cached QC of all five λ.
- The gene-level readout draws its five whole-gene splits with the stage's seed rule (`seed + s`); the manuscript run took split 0 from the C2S-Scale held-out run and splits 1-4 from the same rule. Means over splits agree within their SD, not exactly.
- The manuscript's TIS runs rotated the activations to their principal components as the PCA control; the stage default `pca_of: activations` reproduces them (`pca_of: expression` is the alternative).
