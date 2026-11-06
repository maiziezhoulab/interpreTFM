# Stage `lr_screen`: model-free ligand–receptor co-localization screen

Directory `12_lr_screen/`. Inputs: `ingest` and `niches` (the niche partition used to place the hits). Core environment, CPU; about a minute for 80,000 cells and 400 pairs. Applies only when `dataset.spatial` is set; a configuration without coordinates records the stage as `skipped`. Needs `dataset.labels`.

**The screen is model-free.** It reads the dataset only: the raw counts, the coordinates in their native unit and the labels of `dataset.labels`. No activation, feature or other foreground-model output enters it, and it is computed once per run regardless of the model and layers. The manuscript presents it as the model-free reference: a reader compares the feature-based spatial results (niches, interfaces) with what the measured transcripts and the coordinates already show. Only the last step, the niche placement of the hits, uses a model-derived object, the niche partition of each projected layer, and it is descriptive (where in the tissue a hit lives).

## What it computes

1. **Pairs.** The ligand–receptor database (the packaged CellChatDB human export, see below, or `database` / `complexes` overrides) is restricted to the pairs whose ligand and receptor genes are all on the panel; a ligand or receptor that is a complex is resolved to its subunits, and pairs with the same (ligand gene set, receptor gene set) are kept once, in database order. Genes are matched by symbol (`dataset.gene_id`, with `stages.concepts.gene_keys.symbol_column` for an Ensembl-indexed panel). The pairs are written to `pairs.csv`.
2. **Radius.** `radius_um`, converted to the coordinate unit with `dataset.spatial.um_per_unit` (CosMx: 18 µm = 100 px; Xenium: 100 µm), or `match_neighbors_to: <count>`: the radius at which the section's mean number of other cells within the radius equals `count` (see [matched-neighbour radius](#matched-neighbour-radius)). The mean number of other cells within the radius actually used, `2 · |cell pairs within r| / n`, is always recorded; it is the neighbourhood size that makes two sections comparable.
3. **Statistic.** For a pair, a cell is ligand-positive when every ligand gene has at least `thr` transcripts (receptor-positive likewise). The statistic is the number of ordered (ligand-positive, receptor-positive) pairs of distinct cells within the radius. A pair is tested at a threshold when both sides have at least `min_positive` cells. Thresholds: `thresholds[0]` is the primary one (1 transcript in the manuscript), the others sensitivity thresholds (3 transcripts).
4. **Null.** Positivity is permuted within cell type (`dataset.labels`), ligand and receptor independently: `2 · n_perm` within-type permutations of the cell indices are drawn once from one generator seeded with `seed` (the first `n_perm` for the ligand side) and shared by every pair, in the order of the reference implementation, so that the manuscript's tables are reproduced to the digit. fold = (observed + 1) / (null mean + 1); z = (observed − null mean) / null sd; two-sided normal p; an empirical upper p; Benjamini–Hochberg q over the pairs tested at a threshold.
5. **Calls.** Hit: q < `q_max` and fold ≥ `fold_min`; robust hit: hit at every threshold; depleted (primary threshold): q < `q_max` and fold ≤ 1 / `fold_min`.
6. **Placement.** For every hit at the primary threshold or robust hit (and the pairs of `placement_pairs`), the close pairs at the primary threshold are attributed to the ligand-positive cell; for each niche of the partition `niche.column` / `niche.config` of `10_niches/<layer>/labels.csv` (or `niche_labels` overrides), the share of close pairs whose ligand cell lies in the niche is divided by the niche's share of all cells (1 = as expected from the niche's size; cells without a niche label belong to no niche but stay in both denominators). The cell-type pairs carrying the close pairs (ligand cell → receptor cell, top three by share) are added to `screen.csv`.

## Matched-neighbour radius

A radius in µm holds very different numbers of cells on different platforms (segmentation, cell density) and the same number of native units differs in µm (CosMx 100 px = 18 µm, Xenium 100 µm). `interpretfm.neighbours` gives the curve of the mean number of other cells within a radius and inverts it:

```python
from interpretfm.neighbours import mean_other_cells, radius_curve, match_radius

n_ref = mean_other_cells(coords_ref, 100.0)                  # the reference section's neighbourhood size at its radius
m = match_radius(coords_other, n_ref, step=0.1, round_decimals=1)   # the other section's radius with the same mean count
m["radius"], m["mean_other_cells"]                            # rounded radius (native units) and the exact count there
curve = radius_curve(coords_other, radii)                     # the whole curve on a grid of radii
```

`match_radius` evaluates the curve on multiples of `step` (a geometric bracket, refined until at most 64 grid steps remain, so the two grid points used by the linear interpolation are those of the full `step` grid), interpolates linearly, rounds to `round_decimals` decimals and recomputes the count at the rounded radius. The manuscript's rule is this function with `step` 0.5 px for CosMx and 0.1 µm for Xenium, rounded to 0.1 unit: the CosMx section has 7.20 other cells within 100 px and reaches the Xenium section's 286.67 (at 100 µm) at 652.8 px = 117.5 µm; the Xenium section reaches 7.20 at 15.4 µm.

Cross-section matching is a recipe, not a command: compute the reference count with `mean_other_cells` and set `stages.lr_screen.match_neighbors_to` to it in the other section's configuration (with `match_grid_step` in that section's native unit); the stage records the target, the matched radius in both units and the count at it in `summary.json` and the manifest (`matched`). The hit set is radius-specific (a larger radius dilutes near-contact co-localization: folds shrink while z grows), so a matched radius makes neighbourhood sizes comparable, not effect sizes.

## CellChatDB tables

`src/interpretfm/data/cellchatdb/` ships the human interaction table (3,233 rows: `interaction_name`, `pathway_name`, `ligand`, `receptor`, `annotation`) and the complex table (338 complexes, `complex_name`, `subunit_1` … `subunit_5`) exported unchanged from `CellChatDB.human` of CellChat 2.2.0, with their source, export date and licence in `NOTICE.md` (CellChat and its database are GPL-3, the licence of this package; see also `THIRD_PARTY_NOTICES.md`). CellChat itself (the R package and its communication-probability model) is not used. `database: <csv>` replaces the interaction table by any CSV with the columns `interaction_name`, `ligand`, `receptor` (`pathway_name` and `annotation` optional; a ligand or receptor may also be a `;`-separated gene list); `complexes: <csv>` replaces the complex table, `complexes: false` uses none.

## Parameters (`stages.lr_screen`)

| key | default | meaning |
|---|---|---|
| `radius_um` | – | radius in µm (one of `radius_um` / `match_neighbors_to` is required) |
| `match_neighbors_to` | – | target mean number of other cells within the radius; the stage finds the radius |
| `match_grid_step` | 0.1 | grid step of the matching, in native coordinate units (manuscript: 0.5 px CosMx, 0.1 µm Xenium) |
| `match_round_decimals` | 1 | decimals the matched radius is rounded to (native units) |
| `match_r_max` | extent diagonal | upper bound of the radius search (native units) |
| `thresholds` | `[1, 3]` | positivity thresholds (transcripts); the first is the primary one |
| `n_perm` | 100 | within-type permutations |
| `seed` | 0 | seed of the permutations (manuscript: CosMx 7, Xenium 11) |
| `min_positive` | 100 | minimum positive cells per side for a pair to be tested |
| `q_max`, `fold_min` | 0.05, 1.2 | hit rule |
| `database`, `complexes` | packaged | CSV overrides (`complexes: false` for none) |
| `niche` | `{column: niche}` | niche partition column of `10_niches/<layer>/labels.csv`, or `{config: <label>}` for `niche_at_<label>` |
| `niche_labels` | – | `{<layer>: csv}` overrides (a `cell` column and the niche column) |
| `placement` | true | compute the niche placement |
| `placement_pairs` | `[]` | interaction names placed even when they are not hits |

## Outputs

```
12_lr_screen/
  summary.json                  radius (both units), matched-radius record, mean other cells, pairs, tested / hits per
                                threshold, robust and depleted pairs, the hits' statistics, placement ratios per layer
  pairs.csv                     the database pairs on the panel (ligand_genes / receptor_genes ;-joined)
  screen.csv                    one row per pair: n_lig_pos / n_rec_pos / tested / obs / null_mean / null_sd / fold / z / p /
                                p_emp_upper / q / hit per threshold (suffix _thr<t>), robust_hit, depleted_thr<primary>,
                                top_celltype_pairs; sorted by z at the primary threshold
  <layer>/placement.csv         interaction_name, niche, share_pairs, share_cells, ratio, n_close_pairs
```

`report.json` carries the counts, the radius with its neighbourhood size, the hits' table and the placement ratios (`lr_screen`), and the HTML page shows them with a fold-versus-z figure and the placement per layer ([report](stage_report.md)).

## Reproduction

`configs/paper/cosmx_scgpt_layer2.yaml` (radius 18 µm = 100 px, seed 7) reproduces the manuscript's CosMx screen: 397 pairs on the panel (97 testable at ≥ 3 transcripts), 7 hits at ≥ 1, 33 at ≥ 3, 3 robust (CEACAM6, ESAM, PECAM1), none depleted, every pair's observed count, fold, z and q equal to the manuscript's table; with `match_neighbors_to: 286.67` (and `match_grid_step: 0.5`) the matched radius 652.8 px = 117.5 µm, where 2 / 52 / 1 pairs are hits ([reproducing](../reproducing.md), T11 and T12). The Xenium configurations carry the manuscript's parameters (100 µm, seed 11); the manuscript's Xenium screen permuted within marker-derived labels, which P3 configures through `dataset.labels.file`.

Python: `interpretfm.lr_screen` (`load_cellchatdb`, `panel_pairs`, `neighbour_graph`, `screen_pairs`, `screen_summary`, `niche_placement`, `celltype_pairs`), `interpretfm.neighbours` (`mean_other_cells`, `radius_curve`, `match_radius`) and `interpretfm.stages.lr_screen` (`screen_params`, `resolve_radius`, `read_lr_summary`).
