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
The reproduction data (datasets, activation stores of 24-46 GB per layer, SAE checkpoints and held-out tables) are not part of the repository; they are deposited with the release (see the data statement of the manuscript). The cached intermediates let the tests also check the pure summary rules (λ selection, concept-F1 summary) in seconds, without a pipeline run. The trained SAEs of the six layers are also published on the Hugging Face Hub (`interpretfm.hub.load_sae`), and the [tutorial](tutorials/cosmx_crop.md) runs the spatial stages from one of them on a crop of the CosMx section without any of the stores.
:::

What runs where: every stage of the paper configurations runs in the core environment on a CPU (the stores are cached), so no model environment or GPU is needed; only the fresh-extraction configurations below start the model environments. See [which environment do I need?](installation.md#which-environment-do-i-need).

## Data layout

The configurations name data only through `${NAME}` placeholders ([configuration reference](pipeline/configuration.md#placeholders)). Copy `configs/paper/paths.env.example` to `configs/paper/paths.env` (ignored by git) and fill in the five locations, or export the variables in your shell:

| Variable | Expected content |
|---|---|
| `INTERPRETFM_DATA` | the datasets: `cosmx_human_lung_sec8.h5ad` (81,236 cells × 960 genes, raw counts in `X`, Ensembl `var_names`, symbols in `var["feature_name"]`, labels in `obs["author_cell_type"]`, pixel coordinates in `obsm["spatial"]`, 0.18 µm per pixel) and `xenium_rcc_union_filtered_nonzero.h5ad` (465,220 cells × 399 genes, symbol `var_names`, Ensembl ids in `var["gene_ids"]`, transferred labels in `obs["cell_type"]`, cell area in `obs["cell_area"]`, coordinates in µm), plus `xenium_marker_labels.csv.gz` (one row per Xenium cell, index `cell_id`, the marker-derived labels in `label`: 14 labels including `low_quality`; the label set of the manuscript's Xenium niches, interfaces and screen, read by `stages.<niches|interfaces|lr_screen>.labels` of the Xenium configurations) |
| `INTERPRETFM_STORES` | the activation stores in the sharded 0.9 format, one directory per model and dataset: `acts_extraction_scgpt_cosmx`, `acts_extraction_geneformer_cosmx`, `acts_extraction_c2s_cosmx_store`, `acts_extraction_scgpt_xenium_kidney`, `acts_extraction_geneformer_xenium`, `acts_extraction_c2s_xenium_raw`, each with `activations/<layer>/shard_<i>/{activations.pt,index.pt,metadata.json}` (scGPT stores hold vocabulary ids as token ids; Geneformer stores Ensembl ids; C2S-Scale stores gene symbols) |
| `INTERPRETFM_MODELS` | model directories; only `scGPT/whole-human-pretrain/vocab.json` (the public scGPT whole-human checkpoint) is read, to translate scGPT token ids to gene symbols |
| `INTERPRETFM_PAPER_CACHE` | the manuscript's cached intermediates: `sae_v2/<dataset>/<model>/<layer>/sae/v2biased__steps_8000__lam_1/sae_<layer>_best.pt` (the selected SAE), `.../sae/v2randinit__steps_0/` (the untrained control), `.../sae_eval/v2biased__steps_8000__lam_<λ>/sae_eval.json` (QC of the five λ), `.../f1_heldout/heldout_<sparse|raw|randinit>__<gene_token_row|gene_identity>/` (held-out tables), `sae_v2/sparse_selection.json`, the concept tables `f1heldout_gene_token_row_cosmx_all_layers_l1_1e-3/gprofiler/` and `xenium_geneformer_c2sscale_strict_from_extract_to_immune/01_f1_heldout/gprofiler/` (`gprofiler_binary_gene_by_term.csv`, `gprofiler_terms.tsv`), `tis_layer_screen_3models_cosmx/<model>/<layer>/summary_row.json`, `revision_metrics/concept_f1_v2/summary_long.csv`, `revision_metrics/gene_split_ceiling/summary{,_center}.csv`, `revision_metrics/gsi_mhr_full/*.json`; for the spatial numbers `reconstruction/spatial_sparse/` (`tables/stageB_summary.csv`, `tables/strict_v2_*`, `runs/<dataset>/<model>/<layer>/S/{04_niche_sweep,05_niche_validation,06_niche_terms}/`, `runs/<dataset>/<model>/<layer>/selection_summary.json`), `reconstruction/readthrough_20261104/P1_xenium_k5/` (the Xenium five-niche partitions: `tables/p1_*.csv`, `tables/strict_v2_xenium_marker_k5_*`, `runs/<model>/<layer>/S/05_niche_validation_k5/`), `reconstruction/readthrough_20261104/P10_matched_radius/`, `reconstruction/readthrough_20261104/P6_marker_excluded/`, `reconstruction/spatial_paperA/tables/` and `revision_metrics/hypotheses_fable_spatial_explore/tables/C3_lr_screen_{cosmx,xenium}.csv` |
| `INTERPRETFM_RUNS` | where the run directories are written (`${INTERPRETFM_RUNS}/<configuration name>/`) |

Symbolic links are fine when the stores live on several file systems. `<dataset>` is `cosmx` or `xenium`, `<model>` is `scgpt`, `geneformer` or `c2sscale`, and `<layer>` in the cache uses underscores (`layer_1_norm2`).

## Commands

```bash
make repro-validate                       # every paper configuration against the env file
make repro CONFIG=cosmx_scgpt_layer2      # ingest, extract (cached store), tis, sae, concepts, concept_eval, gene_readout, cells, layers, niches
make repro CONFIG=cosmx_scgpt_layer2 STAGES=interfaces,lr_screen,report   # the remaining spatial stages and the report
make repro-status CONFIG=cosmx_scgpt_layer2
make repro-test                           # pytest tests/reproduction -m reproduction -rA -s
```

The same commands with `CONFIG=` set to any of the six configurations reproduce that layer; `make repro-test` runs every test whose run directory is finished and skips the others with the missing stages named.

`make repro` is `interpretfm run configs/paper/<name>.yaml --env-file configs/paper/paths.env --stages ...`; the run is cached and resumable, so it can be restarted after an interruption and re-run for a subset of stages. On a CPU the stages take minutes (`tis`, `sae`, `concepts`) to hours (`concept_eval`: six passes over the store, run six at a time with `n_jobs: 6` in the paper configurations, about 1–3 h per layer with four threads per process; `gene_readout`: one pass plus the probes; the Xenium spatial stages on 465,220 cells: `cells` about 10 min, `niches` about 1 h, `interfaces` about 15 min per layer), so start it detached (for example in `tmux`) and run the tests afterwards. Setting `INTERPRETFM_REPRO_RUN=1` lets a test run the missing stages itself.

### Fresh extraction (GPU)

The paper configurations start from the cached stores. `configs/paper/extract_cosmx_scgpt.yaml`, `extract_cosmx_geneformer.yaml` and `extract_cosmx_c2s.yaml` instead extract 2,000 seeded CosMx cells per model (the same seed-0 subset for all three) in the model's own environment on a GPU, through the runner's subprocess path, so that the adapters can be checked against the manuscript's stores. They need five more variables in the env file: `INTERPRETFM_SCGPT_ENV`, `INTERPRETFM_GENEFORMER_ENV` and `INTERPRETFM_C2S_ENV` (the conda environments of `envs/interpretfm.yml`, `envs/geneformer.yml` and `envs/c2s.yml`, with the package importable in them), `INTERPRETFM_GENEFORMER_MODEL` (the Geneformer model folder) and `INTERPRETFM_C2S_MODEL` (a local folder of `vandijklab/C2S-Scale-Gemma-2-2B`).

```bash
CUDA_VISIBLE_DEVICES=0 interpretfm run configs/paper/extract_cosmx_scgpt.yaml --env-file configs/paper/paths.env --stages ingest,extract
INTERPRETFM_REPRO_GPU=1 make repro-test     # adds test_extraction.py and test_extraction_c2s.py: runs the three configurations (cached when done) and compares
```

The tests (`tests/reproduction/test_extraction.py`, `test_extraction_c2s.py`) require, for every cell, the same token sequence as the cached store and a per-token cosine of at least 0.999, and a per-cell cosine of at least 0.999 for the features projected through the cached SAE (and against the paper run's `cells.h5ad` when it exists). On the manuscript's stores the three models give identical token sequences for all 2,000 cells, a minimum cosine of 0.999999 per token and above 0.9999999 per projected cell; the reports are written to `02_extract/check_*.json` of each run ([stage extract](pipeline/stage_extract.md)).

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
| T6 `test_t6_features_kept_from_cache` | the retention rule on the cached row-split tables → Table S1 "Features kept" (all six layers) | ±1 |
| T6 `test_t6_features_kept` | the `cells` stage: features and concepts kept | ±1 |
| T7 `test_t7_gsi_excess_and_mhr` | GSI excess and MHR@100 of SAE features, raw dimensions and single genes (every stored cell, seed 0) vs Table S1 and the cached evaluation | ±0.005 |
| T8 `test_t8_niche_selection_rule_from_cache` | the score and selection rule on the cached sweep table → r100/xm/kmeans/k3 with its seed ARI, purity, tile null and z; the cached labels' ARI/NMI | ±0.005 (purity ±0.01, z ±5 %) |
| T8 `test_t8_niche_sweep_winner_and_validation` | the `niches` stage: winner and validation (Table S5) | same |
| T9 `test_t9_label_composition_from_dataset` | the label-composition baseline on the dataset alone (author labels, 120 px, k = 3) | ±0.005 |
| T9 `test_t9_expression_and_label_composition_baselines` | the `niches` stage: expression-only winner, expression at the scGPT configuration, label composition | ±0.005 |
| T13 `test_t13_niche_concept_labels_from_cache` | niche concept labels of the cached scGPT niches (identical top-25 lists) and the labels shared with the cached Geneformer niches (2 / 8 / 8, Jaccard 0.042 / 0.190 / 0.190) | counts ±1, Jaccard ±0.005 |
| T13 `test_t13_niche_concept_labels` | the same from the `niches` stage | same |
| T10 `test_t10_interface_test_from_cache` | the interface test on the cached niche-labelled validation h5ad of each CosMx model (scGPT, Geneformer, C2S-Scale; the reference's input): niche pairs, shares of \|z\| > 2 under the tile and the calibrated null, enriched / depleted, z*, effect sizes, false-positive shares, neighbour correlations, niche-free calls; the cached pair tables are reproduced (z within 1e-5, identical calls) | counts ±1, z* ±0.1, shares ±1 pp, effect ±0.005 |
| T10 `test_t10_interface_test` | the `interfaces` stage of the paper run on the pipeline's own r100 niches; the niche-free numbers, the calibration and the fired-on gene sets are exact | niche-stratified counts ±3 and shares ±2 pp (see below); the rest as above |
| `test_partner_marker_tercile_and_marker_excluded_from_cache` | partner markers (identical sets), the partner-marker tercile over the three models (39 rows / 13 pairs, median g 0.071, CI [-0.770, 0.581]) and the marker-excluded re-test (3 spillover-robust pairs; reduced-score median g -1.067, CI [-2.056, 0.079]; the full score on the same permutations equals the tercile row) | g and CI ±0.005, counts exact |
| T11 `test_t11_lr_screen_from_dataset` | the ligand–receptor screen (`interpretfm.lr_screen`) on the dataset alone at 100 px = 18 µm, seed 7: 397 pairs on the panel (97 testable at ≥ 3 transcripts), 7 / 33 hits, 3 robust (CEACAM6, ESAM, PECAM1), none depleted; every tested pair's observed count, fold, z and q against the manuscript's screen table; the niche placement of the robust hits on the cached scGPT niches | counts exact; fold, z, q and placement ratios ±0.005 (observed counts exact) |
| T11 `test_t11_lr_screen_matched_radius_from_dataset` | the same screen with `match_neighbors_to` = the Xenium section's mean count at 100 µm (286.67): matched radius 652.8 px = 117.5 µm (286.70 other cells), 2 / 52 hits, 1 robust (CEACAM6), per pair against the manuscript's matched-radius table; about five minutes | radius ±0.1 px, counts exact, fold / z / q ±0.005 |
| T12 `test_t12_matched_neighbour_radii` | coordinates only, both sections (the Xenium coordinates through the Xenium paper configuration): CosMx 7.20 other cells at 100 px and 286.70 at 652.8 px, Xenium 286.67 at 100 µm and 7.24 at 15.4 µm, the matched radii 652.8 px and 15.4 µm; against the cached `p10_matched_radii.json` | counts ±0.01, radii ±0.1 unit (cached values to 1e-9) |
| T11/T12 `test_t11_t12_lr_screen_stage` | the `lr_screen` stage of the paper run: the T11 counts and per-pair values, 7.20 other cells within 100 px, the placement of the robust hits on the pipeline's own niches (matched to the cached niches by their ratio pattern) | as T11; placement ratios ±0.05 (see below) |
| `test_fresh_extraction_matches_cached_store`, `test_fresh_c2s_extraction_matches_cached_store` (GPU, `INTERPRETFM_REPRO_GPU=1`) | 2,000 cells extracted in the model environment (scGPT `layer_1.norm2`, Geneformer `layer_1`, C2S-Scale `layer_13`) vs the cached store: token sequences, per-token cosine, projected features through the cached SAE | identical token sequences; cosine ≥ 0.999 per token and per projected cell |

Every comparison prints one line `observed | manuscript | cached source | tolerance` (`-s`).

### All six layers

`test_model_stages.py` (T1–T5) is parametrised over the six configurations; `test_cells_and_spatial.py` holds the CosMx scGPT spatial tests (T6–T13) and `test_other_layers.py` the five other layers. What is tested per layer and from which cache (✓ cache test, R run-dir test, – not available):

| | CosMx scGPT L2 | CosMx GF L2 | CosMx C2S L14 | Xenium scGPT L3 | Xenium GF L2 | Xenium C2S L14 |
|---|---|---|---|---|---|---|
| T1 TIS gain (Table S1; CosMx also vs `tis_layer_screen_3models_cosmx`) | R | R | R | R (manuscript only: no cached Xenium screen row) | R | R |
| T2 λ rule on the cached QC of five λ; the `sae` stage at λ = 1; untrained SAE vs the cached control | ✓ R | ✓ R | ✓ R | ✓ R | ✓ R | ✓ R |
| T3/T4 concept F1 (row, whole-gene) rule on the cached tables; recomputed from the store | ✓ R | ✓ R | ✓ R | ✓ R | ✓ R (the cached row-split tables of the de-duplicated rerun) | ✓ R |
| T5 gene-level readout (Table S3, ±0.03) | R | R | R | R | R | R |
| T6 features kept (rule on the cached row-split tables; the `cells` stage) | ✓ R | ✓ R | ✓ R | ✓ R | ✓ R (125 after `dedup_cells: auto`) | ✓ R |
| T7 GSI excess / MHR@100 vs `gsi_mhr_full` | R | R | R | R (±0.02: cell subset, see below) | R (±0.02) | R (±0.02) |
| T8 niche selection rule on the cached sweep; agreement of the cached niche labels; the `niches` stage at the manuscript configuration | ✓ R (r100/xm/kmeans/k3) | ✓ R (r120/m/kmeans/k3) | ✓ R (r100/m/kmeans/k3) | ✓ R (five niches r100/xm/gmm/k5; three niches r100/m/gmm/k3) | ✓ R (r120/m/kmeans/k5; r150/m/gmm/k3) | ✓ R (r100/xm/kmeans/k5; r150/m/gmm/k3) |
| T9 expression-only and label-composition baselines | ✓ R | R | R | R (k = 5) | R (k = 5) | R (k = 5) |
| T10 interface test on the cached niche-labelled h5ad; the `interfaces` stage | ✓ R (author labels) | ✓ R | ✓ R | ✓ R (marker labels, five niches, `strict_v2_xenium_marker_k5_*`) | ✓ R | ✓ R |
| partner-marker tercile and marker-excluded QC (three CosMx models) | ✓ | ✓ | ✓ | – | – | – |
| T11 ligand–receptor screen on the dataset and the `lr_screen` stage; placement on the cached and the pipeline's niches | ✓ R (`C3_lr_screen_cosmx`) | R (placement) | R (placement) | ✓ R (`C3_lr_screen_xenium`, marker labels; `p1_lr_screen_niche_context`) | ✓ R | ✓ R |
| T12 matched-neighbour radii (both sections) | ✓ R | – | – | ✓ (coordinates) | – | – |
| T13 niche concept labels: cross-model recipe on the cached top-25 lists and on the paper runs | ✓ R (vs GF 2/8/8) | ✓ R (vs C2S 6/10/11) | ✓ R | ✓ R (five niches: matched 0.308 / 0.262 vs unmatched 0.073 / 0.095) | ✓ R (vs C2S 0.287 / 0.099) | ✓ R |

Label sets: the CosMx tests use the author labels (`dataset.labels`); the Xenium cell-level tests (T5 cell-type control, T7 populations) use the transferred `cell_type` column as the manuscript did, and the Xenium spatial tests the marker labels of `xenium_marker_labels.csv.gz` through the stages' `labels` override.

## Known differences from the manuscript runs

- Only the selected λ = 1 checkpoint of each layer is cached, so the `sae` stage of the paper configurations evaluates that checkpoint; the rule itself is tested on the cached QC of all five λ.
- The gene-level readout draws its five whole-gene splits with the stage's seed rule (`seed + s`); the manuscript run took split 0 from the C2S-Scale held-out run and splits 1-4 from the same rule. Means over splits agree within their SD, not exactly.
- The manuscript's TIS runs rotated the activations to their principal components as the PCA control; the stage default `pca_of: activations` reproduces them (`pca_of: expression` is the alternative).
- The niche sweep's top scores are close (the CosMx scGPT winner r100/xm/kmeans/k3 leads r120/xm/kmeans/k3 by 0.002 in the manuscript's run). The clustering depends on the scikit-learn version through the PCA solver and k-means numerics, so the `niches` stage run in a newer environment can rank these two configurations the other way while every number of the published configuration stays within tolerance; the reproduction test reports both. The manuscript's environment (scikit-learn 1.3.2) reproduces the published winner.
- The interface test's niche-stratified numbers depend on the niche partition: the paper run tests the pipeline's own r100 niches (ARI 0.4175 against the author labels; the cached labels of the manuscript give 0.4187), so pairs near the 50-edge threshold appear or disappear (127 vs 129 pairs) and the shares of |z| > 2 move by about 1 pp. The reproduction test prints both values and accepts ±3 pairs, ±2 pp and ±0.05 on the median and maximum effect there; the niche-free test, the calibration (z*, false-positive shares, neighbour correlations) and the fired-on gene sets do not depend on the niches and are reproduced exactly, as is the whole test on the cached niche labels (`test_t10_interface_test_from_cache`).
- The ligand–receptor screen itself is model-free and reproduces the manuscript's tables to the digit (the permutation draw order of the reference is kept). Its niche placement depends on the niche partition like the interface test: on the pipeline's own r100 niches the robust hits' ratios differ from the cached ones by up to about 0.02 (the pipeline numbers niches by size, so the test matches niche ids by the ratio pattern and accepts ±0.05); on the cached niche labels (`test_t11_lr_screen_from_dataset`) they are exact. The cell-type pairs carrying a hit (`top_celltype_pairs`) list the top three by share; two pairs with the same share can appear in either order.
- CosMx Geneformer and C2S-Scale niches: as for scGPT, the sweep's top scores lie within a few hundredths and the seed-to-seed ARI of k-means moves by up to 0.015 between library versions (Geneformer r120/m/kmeans/k3: 0.9867 here, 0.9985 in the manuscript's sweep; rank 5 of 36 instead of 1), while the ARI / NMI of the manuscript configuration against the author labels are reproduced to 1e-4 (0.3422 / 0.2924). The run-dir tests of these layers assert the manuscript configuration's ARI / NMI (±0.01) and seed ARI (±0.02) and report the sweep winner and its score gap; the niche-stratified interface shares take ±3 pp (the legacy tile share moved by 2.3 pp with the partition).
- Xenium GSI excess / MHR@100 (T7): the `cells` stage reads the evenly spaced shards holding about `n_cells` cells (100,000 in the Xenium configurations) and drops cells at shard edges; the manuscript's evaluation applied the same rule to the stores as they were indexed then and read 63,068 (scGPT) / 84,895 (Geneformer, C2S-Scale) cells, while the stage reads about 80,000 (scGPT). The populations are the same 14 transferred labels and the values agree within 0.01 (sparse GSI excess 0.266 vs 0.256, MHR@100 0.821 vs 0.826 for scGPT); the test accepts ±0.02 when the cell counts differ and prints both counts.
- Xenium niches: the manuscript's Xenium partitions are the five-niche configurations (readthrough P1; Supplementary Note S1), listed as the first fixed configuration of each Xenium configuration with the earlier three-niche configuration second; the sweep winner of the stage is reported beside them. Five-cluster GMM / k-means on 465,220 cells moves slightly between library versions, so the run-dir tests accept ±0.01 on the ARI / NMI against the marker labels and 3 % of the cells on the niche sizes, report the purity and its tile null without asserting them (scGPT's five-niche GMM gives purity 0.471 against the manuscript run's 0.531 while its ARI / NMI against the marker labels are 0.080 / 0.149 against 0.081 / 0.144; the three-niche configurations reproduce the cached purity exactly), and the niche-stratified interface numbers ±10 pairs / ±5 pp (scGPT: 375 pairs, 15 enriched / 26 depleted, z* 2.059 against 376, 15 / 27, 2.097); for the screen's placement on the pipeline's five niches each robust hit must lie in the matched core niche with its largest ratio within ±0.3 of the cached one, the other ratios being reported. The Xenium expression-only sweep's top configurations (r100/m/gmm/k3 in the manuscript's run, r100/xm/gmm/k3 here) swap because the GMM's seed stability moves between library versions; the winner is reported and the expression ARI at the published winner is checked where that configuration is among the fixed ones. The niche-free interface test, the calibration and the screen are reproduced exactly, and the cached partitions themselves are tested exactly (`*_from_cache`).
- Parallel concept evaluation: `concept_eval` with `n_jobs: 6` runs the six evaluations in separate processes through the same function as the sequential stage; the tables are identical (toy test) and the Table S1 values of the layers evaluated both ways agree to the digit.
- Stage hashes: `dataset.labels` no longer enters the hash of the label-free stages (`extract`, `tis`, `sae`, `concepts`, `concept_eval`), so run directories written before this change show a different hash for those stages; `make repro-test` checks stage status, not hashes, and a cached result is kept as long as it is not re-run.
- Xenium Geneformer row split: the stage drops duplicate cells of the store before drawing the split (`dedup_cells: auto`, 17,164,805 rows), so each split holds half the rows of the cached tables of this layer (to within 0.1 %); the test checks that ratio.
- Gene-level readout, random-vector control: the printed value (±0.03) and the 0.47–0.53 band are asserted; the cached rows are another draw and are reported.

### Printed values not reproduced within tolerance

Pipeline values from the test environment (Python 3.11, scikit-learn 1.9.1). These comparisons are listed in `tests/reproduction/expected.py` (`KNOWN_DIFFERENCES`); the tests report them and assert every other comparison of the same test.

| Comparison | Manuscript | Manuscript value | Pipeline |
|---|---|---|---|
| Xenium Geneformer TIS gain | Table S1 | 0.195 | 0.201 |
| Xenium C2S-Scale five niches (r100/xm/kmeans/k5), seed-to-seed ARI | Table S4 | 0.996 | 0.616 |
| Xenium expression-only niches at the Geneformer configuration (r120/m/kmeans/k5), ARI / NMI | Table S4 | 0.052 / 0.105 | 0.059 / 0.126 |
| Xenium expression-only niches at the C2S-Scale configuration (r100/xm/kmeans/k5), ARI / NMI | Table S4 | 0.034 / 0.095 | 0.046 / 0.098 |
| Xenium C2S-Scale niche-stratified interface calls on the pipeline's niches, enriched / depleted | Table S9 | 18 / 24 | 15 / 22 |
| Xenium C2S-Scale false-positive rate on i.i.d. noise, niche-stratified pairs (%) | manuscript run (not printed) | 4.5 | 5.6 |
| Xenium Geneformer concepts marked at the row split (test F1 ≥ 0.5) | manuscript run (range 262–593 printed) | 572 | 540 |

The interface test on the cached partitions reproduces the printed calls (`test_t10_interface_test_from_cache_xenium`).

Set `INTERPRETFM_REPRO_REPORT_ALL=1` to list every comparison outside its tolerance in one run (each is printed as `OUT OF TOLERANCE`) instead of stopping at the first.
