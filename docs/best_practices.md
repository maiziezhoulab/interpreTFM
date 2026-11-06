# Evaluation best practices

These recommendations come from the design of the evaluations in the manuscript. Most of them guard against a feature-level result that looks informative but is already explained by a simpler representation, by gene identity, or by the spatial layout of the tissue.

## Report every feature metric next to its baselines

A concept-alignment or specificity score for SAE features means little on its own. Score, with the same rows, partitions, labels, thresholds and metric:

- **raw hidden dimensions**, each split into its positive and negative parts so that all units are non-negative like SAE features;
- **an untrained SAE** of the same width and architecture as the trained one.

Both baselines can be written as checkpoints and passed to the same evaluator:

```python
import torch
from interp_pipeline.sae.sae_base import AutoEncoder

d = 512                 # hidden size of the layer (scGPT: 512)
n_latents = 8 * d       # same width as the trained SAE

# raw hidden dimensions: ReLU([x, -x]) gives the positive and negative parts
eye = torch.eye(d)
torch.save({"state_dict": {"encoder.weight": torch.cat([eye, -eye])}, "d_in": d, "n_latents": 2 * d},
           "raw_dims.pt")

# untrained SAE: same architecture and initialization as the trained SAE, no training
# (the bias-free AutoEncoder is shown as an example)
torch.manual_seed(0)
torch.save({"state_dict": AutoEncoder(d, n_latents).state_dict(), "d_in": d, "n_latents": n_latents},
           "untrained_sae.pt")
```

Then call `heldout_report_for_layer` once per checkpoint ([usage](usage.md#4-held-out-concept-alignment)). Summarize per concept, so that each concept counts once and a broadly active unit cannot dominate the summary.

## Hold out whole genes to test concept generalization

All occurrences of a gene share its concept label. When gene-token rows are split at random, the test partition contains new observations of genes already seen during feature selection, so a feature that simply detects a gene scores well. This row split answers "does the feature recognize known member genes in new cells?".

To ask whether a representation generalizes to genes it has not seen, hold out whole genes (`heldout_unit="gene_identity"`): every row of a gene stays in one partition. Report both splits and name the held-out unit wherever a number is quoted.

## Compare cell-level features with single genes

A feature that marks a population is only as useful as the best marker gene it replaces. Score cell-level features with the same specificity metrics (GSI, MHR@100) as raw dimensions and as single genes (log-normalized expression), on the same cells and populations. Report how close the features come to single genes, not only how far they exceed raw dimensions.

## Give spatial radii in µm per platform

Coordinate units differ between platforms (for example 0.18 µm pixels in CosMx, micrometers in Xenium), so the same radius in coordinate units covers very different tissue areas and neighbor counts.

- State every radius, tile and block size in both coordinate units and µm. The pipeline takes radii in µm and converts them once through `dataset.spatial.um_per_unit`; the stage summaries and the report print both units.
- When comparing platforms or datasets, also report the mean number of neighbors within the radius, and consider repeating an analysis at matched neighbor counts (`stages.lr_screen.match_neighbors_to`; `interpretfm.neighbours.match_radius`).

## Say which labels a spatial result uses

Niche validation, the interface test and the screen all condition on cell labels (ARI/NMI against them, permutations within them, the label-composition baseline). Name the label set with every number: author annotations, transferred labels and marker-derived labels give different partitions of the same cells, and a label set that was itself derived from expression makes the expression-only baseline look weaker than it is. `dataset.labels` sets the default, and `stages.<niches|interfaces|lr_screen>.labels` lets one stage use another set; the report records which one each section used.

## Calibrate spatial nulls on contact-free fields, and check spillover

Permutation nulls that shuffle values within large tiles can call interfaces "enriched" more often than their nominal rate in fields that contain no cell–cell contact effect, because spatial autocorrelation survives the shuffle.

- **Calibrate the call threshold** on contact-free negative controls: smooth random fields whose neighbor correlation matches the real score, passed through the same null. Choose the threshold that keeps the per-pair false-positive rate on these fields near the nominal level, and report that rate.
- **Adjust for transcript depth.** Program scores that average many features tend to track transcript counts; adjust within cell type for counts, detected genes and cell area before testing.
- **Check partner-marker spillover.** In imaging-based platforms, transcripts of a neighboring cell can be assigned to the wrong cell. Look at interface z-scores across terciles of the partner cell type's marker counts, and recompute calls after removing features that fire on either cell type's top markers. A call that follows partner-marker counts is not evidence of an interface program.
- **Know the power of the calibrated test.** Calibration trades sensitivity for a controlled false-positive rate. On the CosMx section, a planted interface effect of 5 % of the program score was detected in 38 % of the tests (positive-control check of the manuscript), so an absence of calibrated calls on one section does not rule out effects of that size; report the calibrated and the uncalibrated shares side by side and state the planted-effect sensitivity when it is available.
- A negative z-score means depletion relative to the null, not avoidance.

## Choose the layer per criterion, not by a composite score

The per-criterion layer table (`layers` stage) lists, for every evaluated layer, TIS with its controls, concept F1 on both held-out units with the raw and untrained baselines, the gene-level readout, the retained features and the cell-level specificity. The layers that lead differ between criteria, and the manuscript's representative layers were chosen per model for one stated criterion. Pick the layer for the question at hand (concept alignment for feature interpretation, specificity for cell-level markers, niche and interface validation for spatial use), say which criterion chose it, and do not average criteria into one score.

## Label post hoc analyses

- Write down analysis rules before computing results: the SAE ℓ1-weight rule, the feature-retention rule, the summary metric, replication criteria and the inclusion rule for exploratory analyses. Time-stamp them (a commit in the analysis repository works).
- Report every pre-specified analysis, whether or not its rule was met.
- Label anything added after seeing results as post hoc.

## Further habits

- Select features, thresholds and configurations on validation data only; use test data once.
- Build the concept universe from the measured panel, and describe results as panel-observable concepts, not genome-wide pathways.
- Fix and report random seeds; report seed-to-seed variation for clustering.
- Treat cells and edges in one section as technical units nested in one specimen; replicate across sections or donors before generalizing.
- Read co-localization and interface results as spatial association of expression, not ligand binding, signaling direction or receptor activation.
