# Stage `concept_eval`: held-out concept alignment

Directory `06_concept_eval/`. Inputs: the activation store, `sae` (the selected and the untrained checkpoints) and `concepts` (the universe). Core environment; encoding uses the GPU when available.

## What it computes

Every gene-token row of the store (one measured gene in one cell) carries the concept memberships of its gene as labels. A *unit* of a representation predicts membership of concept *c* at threshold τ when its activation exceeds τ; the prediction is scored by F1 against the gene-level label. Three representations are scored with the same rows, partitions, labels, thresholds and metric:

- `sae`: the features of the SAE selected by the λ rule;
- `raw`: every hidden dimension split into its positive and negative part (`[I; -I]`), so that all units are non-negative;
- `untrained`: the untrained SAE of the same width.

Two held-out units:

- **row split** (`row`): each mapped row goes to train (70 %), validation (20 %) or test (10 %) by a deterministic hash of the seed, shard, row index and token id, so rows of the same gene (and of the same cell) occur in every partition; this measures generalization to held-out observations of known genes;
- **whole-gene split** (`gene`): the measured genes are split 70/20/10 and all rows of a gene follow it, so test genes are never seen during feature selection.

Per concept and threshold the `top_m` units by validation F1 are kept and scored on the test partition. Stores that recorded a cell more than once are reduced to the first copy of each cell before evaluation (`dedup_cells`).

**Concept F1** (the summary reported per layer, unit and representation): at τ = 0.15, for every concept with 5–100 panel genes and positive rows in both held-out partitions, the unit with the highest validation F1 (ties: lowest index) scored by its test F1; the mean over concepts, so each concept counts once and a broadly active unit cannot dominate.

## Parameters (`stages.concept_eval`)

| key | default | meaning |
|---|---|---|
| `thresholds` | `[0, 0.15, 0.3, 0.6]` | activation thresholds (latent scale) |
| `split` | `{valid: 0.2, test: 0.1}` | held-out fractions |
| `units` | `[row, gene]` | held-out units to run |
| `representations` | `[sae, raw, untrained]` | |
| `summary` | `{tau: 0.15, concept_size: [5, 100]}` | the concept-F1 rule |
| `dedup_cells` | `auto` | `auto`/`true` keep the first copy of every cell; `false` keeps every row |
| `seed` | `run.seed` | split seed |
| `top_m` | 200 | units kept per concept and threshold on validation |
| `batch_size` | 8192 | rows per encoding batch |
| `max_shards` | null | read only the first *n* shards (smoke runs) |
| `device` | `auto` | |
| `token_vocab` | null | JSON vocabulary when the store holds token numbers |
| `checkpoints` | `{}` | `{<layer>: path}`: SAE checkpoints to score instead of the selected ones |
| `store_root` | null | a cached activation store |
| `n_jobs` | 1 | evaluations (unit × representation) run at a time; above 1 each runs in its own spawned process through the same function as the sequential path, so the tables are identical and only the wall time changes (the paper configurations use 6) |
| `threads_per_job` | null | torch threads of every evaluation process (default: torch's own default, which follows `OMP_NUM_THREADS`) |

`n_jobs` and `threads_per_job` are left out of the stage's configuration hash (`hash_ignore` at registration): they cannot change a result, so a cached stage stays valid when they change. Six evaluations of a 952-shard store take about 17 h one after the other and about 3 h with `n_jobs: 6` on a CPU node; memory is about 3–7 GB per process.

## Outputs

```
06_concept_eval/
  concept_f1.json               parameters and per-layer summaries
  concept_f1.csv                layer, unit, representation, n_units, n_concepts, concept_f1, share_test_f1_ge_0.5
  <layer>/concept_f1.json
  <layer>/f1_<unit>_<representation>/
      valid_concept_f1.csv      concept, feature, threshold, precision, recall, f1, tp, fp, fn, pred_pos, true_pos
      test_concept_f1.csv       the validation-selected units scored on test
      per_concept.csv           concept, feature, valid_f1, test_f1, n_genes, supported, in_size_range
      support.csv               positive rows per concept and partition
      split.json                partition sizes; the gene lists for the whole-gene split
      counts.json
```

Python: `interpretfm.concept_eval.assign_rows` → `heldout_concept_f1` → `summarize_concept_f1`; the F1 counting reuses `interp_pipeline.get_annotation.f1_alignment` (0.9).
