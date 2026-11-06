# Concepts

This page explains each measure in plain language, following the manuscript's Methods. Each section ends with where the measure is implemented (the pipeline stage and the analysis module).

## Layer-resolved activations

The foundation model is frozen; nothing is fine-tuned. Each model receives its native input: scGPT gene tokens with normalized expression values and a leading classification token, Geneformer expression-ranked gene tokens, and C2S-Scale a ranked list of gene names ("cell sentence"). For every evaluated layer, the hidden state of every token is written to a sharded activation store together with its cell and token identifiers.

Two kinds of rows are kept apart:

- **Gene-token rows**: the context-dependent representation of one measured gene in one cell. Concept alignment and the projection of features to cells use these rows; special tokens are excluded.
- **Cell-level rows**: the classification-token state for scGPT and Geneformer, and the mean over gene-token rows for C2S-Scale. TIS uses these.

Stored layer identifiers are zero-based (`layer_1.norm2` is display Layer 2).

**Implementation:** the `extract` stage ([stage extract](pipeline/stage_extract.md); adapters in `interpretfm.models`, comparison of stores in `interpretfm.extract`), built on `interp_pipeline.adapters.models`, `interp_pipeline.extraction` and the activation store `interp_pipeline.io.activation_store`.

## Transcriptomic interpretability score (TIS)

TIS asks whether variation along one representation dimension shows up in the measured transcriptome. For a dimension, cells in its upper and lower activation quartiles form a high pool and a low pool. In each trial a query cell is drawn, a fixed number of reference cells is sampled from each pool (excluding the query), and the query is assigned to the pool whose mean expression profile is closer by cosine similarity. Similarity is computed on measured expression (library-size normalized, log-transformed, L2-normalized), not on model activations. TIS is the fraction of trials in which this assignment matches whether the query's own activation is above the dimension's median.

- A high TIS means the dimension's variation is reflected in expression. It does not assign a concept to the dimension.
- **Controls:** each dimension permuted across cells (the layer-level TIS gain is the mean TIS minus the mean after permutation), and the same activations rotated to their principal components (a dimension-matched change of basis) as a descriptive reference.

**Implementation:** the `tis` stage ([stage tis](pipeline/stage_tis.md); `interpretfm.tis`, with the shuffled and PCA controls), built on `interp_pipeline.tis.core` (scores and controls) and `interp_pipeline.tis.io` (reading cell-level rows, building the expression matrix used for similarity).

## Sparse autoencoders

Each evaluated layer gets its own SAE, trained on that layer's gene-token activations. The SAE maps an activation vector to a wider, non-negative, mostly-zero feature vector and reconstructs the input from it; training minimizes reconstruction error plus an ℓ1 penalty on the features. The dictionary has eight times as many features as the hidden state has dimensions.

In the manuscript the SAE has an encoder bias and an input bias, decoder columns of unit norm, and an ℓ1 weight chosen by a rule fixed in advance: among candidate weights with at most 2% of features active per token and a dead-feature rate of at most 0.25, the one with the lowest fraction of variance unexplained.

**Implementation:** the `sae` stage ([stage sae](pipeline/stage_sae.md); `interpretfm.sae`: the ReLU SAE with biases and unit-norm decoder, dead-feature resampling, the fixed ℓ1-weight rule and the untrained control). `interp_pipeline.sae.sae_base.AutoEncoder` and `interp_pipeline.sae.trainers.fit_sae_for_layer` fit a bias-free ReLU SAE; `interp_pipeline.sae.normalize` rescales checkpoints. The concept-alignment evaluator reads checkpoints with or without biases.

## Concept universe

Biological concepts come from g:Profiler (human; Gene Ontology biological process, molecular function and cellular component, Reactome and KEGG; false-discovery rate 0.05), queried with the measured gene panel. Terms must overlap at least two measured genes. The result is a binary gene × concept matrix.

Because the universe is restricted to the panel, alignment tests concepts that are observable on that panel. It is not genome-wide pathway discovery.

**Implementation:** the `concepts` stage ([stage concepts](pipeline/stage_concepts.md); `interpretfm.concepts`: the g:Profiler universe with a cache and an offline matrix mode), built on `interp_pipeline.get_annotation.gprofiler_client` and `panel`; `membership_gt` builds an alternative GO membership matrix from QuickGO annotations. The [format](data.md#concept-matrix) of the binary gene × concept matrix is documented so that it can be supplied directly.

## Held-out concept alignment

A feature predicts that a gene-token row belongs to a concept when its activation exceeds a threshold (thresholds are on the activation scale, not probabilities). Predictions are compared with the gene's concept membership by F1. The best feature for each concept is chosen on a validation partition and scored on a test partition. All occurrences of a gene share the gene's label, so alignment measures association with annotated gene-set membership, not pathway activity in a particular cell.

Two held-out units answer different questions:

- **Row split:** gene-token rows are assigned to training, validation and test partitions, so other rows of the same gene or cell can be in other partitions. This tests held-out observations of known genes.
- **Whole-gene split:** genes are assigned to partitions and all rows of a gene follow it, so test genes are never seen during selection. This tests generalization to unseen genes.

**Baselines:** the same analysis, on the same rows, partitions, labels, thresholds and metric, for raw hidden dimensions (each split into its positive and negative parts so that all units are non-negative) and for an untrained SAE of the same width. A per-concept summary counts each concept once: for each concept with 5–100 panel genes and positive rows in both validation and test, the unit with the highest validation F1 is scored by its test F1, and concept F1 is the mean over concepts.

**Implementation:** the `concept_eval` stage ([stage concept_eval](pipeline/stage_concept_eval.md); `interpretfm.concept_eval`: both held-out units for SAE features, raw dimensions and the untrained SAE, with the per-concept summary), built on `interp_pipeline.get_annotation.f1_alignment.heldout_report_for_layer` (`heldout_unit="gene_token_row"` or `"gene_identity"`). Raw dimensions and untrained SAEs can also be evaluated by passing them as checkpoints ([best practices](best_practices.md)).

## Population specificity

Cell-level profiles are the mean activation of each feature over a cell's gene tokens; raw dimensions (positive and negative parts) and single genes (log-normalized expression) are scored the same way. Following scPISCES:

- **Gini specificity index (GSI):** the Gini coefficient of a feature's mean activation across cell populations. GSI excess subtracts the mean GSI of the same feature after shuffling population labels.
- **Marker-gene hit rate (MHR@100):** cells are split in half within each population. In one half, a population's markers are the 100 genes with the largest mean expression difference between it and all other cells, and the features most correlated with population membership are selected; in the other half, MHR is the share of the genes most correlated with each selected feature that are markers, averaged over populations.

**Implementation:** the `gene_readout` stage ([stage gene_readout](pipeline/stage_gene_readout.md); `interpretfm.gene_readout`).

## Cell-level projection

Concept-associated features are retained by a fixed rule (for each concept, the feature with the highest validation F1 if it passes a minimum F1, then deduplicated) and projected to cells by averaging over each cell's gene-token rows. Selection uses validation rows only.

**Implementation:** the `cells` stage ([stage cells](pipeline/stage_cells.md); `interpretfm.cells`: feature retention, gene-token projection to cells, GSI and MHR@k against raw dimensions and single genes). `interp_pipeline.downstream.f1` (feature tables and GO-based concept reduction) and `build_interpretable_from_sae_latents` (`python -m interp_pipeline.downstream.f1.build_interpretable_from_sae_latents --help`) build a cell × concept-feature AnnData from classification-token activations.

## Spatial niches

For each cell, a microenvironment vector averages the feature profiles of all cells within a radius. Niches are clustered from the microenvironment alone (`m`) or from the cell's own profile together with its microenvironment (`xm`), by k-means or a Gaussian mixture, over a grid of radii, cluster numbers and random seeds. Configurations are ranked by seed-to-seed agreement (ARI), spatial neighborhood purity against a tile-shuffle null, and separation between niche centroids; configurations with very small clusters are rejected.

**Baselines:** the same sweep on components of log-normalized expression (expression-only niches), and neighborhood cell-type composition (label-composition niches), which by construction favors agreement with the same labels.

**Implementation:** the `niches` stage ([stage niches](pipeline/stage_niches.md); `interpretfm.niches`, `interpretfm.niche_concepts`), built on `interp_pipeline.downstream.context` (neighborhoods, microenvironment vectors, clustering, the sweep and its scores, validation summaries). The sweep takes any cell × feature matrix, so the expression-only baseline is run by passing expression components instead of features.

## Interface statistic and spillover checks

A cell's program load is the mean of its clipped, standardized, positive feature values. Because this load can track transcript counts, it is adjusted within each cell type for transcript count, detected genes and cell area. Edges connect cells within a fixed radius; for each pair of cell groups within a niche, the observed score is the mean load over its edges.

- **Null:** the adjusted load is permuted within strata defined by spatial blocks, niche and cell type, keeping coordinates, graph and labels fixed; the result is a z-score and an empirical upper-tail value.
- **Calibration:** the call threshold is set on contact-free negative controls, smooth random fields with matched neighbor correlation passed through the same null, so that the per-pair false-positive rate on such fields is near 5%. This is not a false-discovery-rate control across pairs.
- **Spillover checks:** z-scores within terciles of partner-marker transcript counts, and calls recomputed after removing features that fire on either cell type's top markers.

A negative z-score means depletion relative to the null, not biological avoidance.

**Implementation:** the `interfaces` stage ([stage interfaces](pipeline/stage_interfaces.md); `interpretfm.interfaces`: the count adjustment, the stratified null, the contact-free calibration and the spillover checks), built on `interp_pipeline.downstream.interaction` (program scores, radius edges, group-pair tables, the within-tile permutation null).

## Ligand–receptor co-localization screen

This screen uses no model features. For each ligand–receptor pair whose genes are all on the panel, a cell is positive for a gene at one or more transcripts (a complex needs all subunits). The statistic is the number of ligand-positive and receptor-positive cell pairs within a radius; the null permutes positivity within cell type, independently for ligand and receptor. Hits need a multiple-testing-adjusted q-value and a minimum fold change; for each hit, the share of close pairs inside each model niche, relative to the niche's share of cells, places the pair in tissue context. Because neighbor counts differ between platforms, the screen can be repeated at matched neighbor counts.

It reports co-localization of expression, not ligand binding, signaling direction or receptor activation.

**Implementation:** the `lr_screen` stage ([stage lr_screen](pipeline/stage_lr_screen.md); `interpretfm.lr_screen`, `interpretfm.neighbours`).
