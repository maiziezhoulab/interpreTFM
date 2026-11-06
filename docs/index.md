# interpreTFM

interpreTFM is a layer-resolved framework for evaluating what transcriptomic foundation models encode.
It extracts hidden states from frozen scGPT, Geneformer and C2S-Scale models, decomposes each evaluated layer with a sparse autoencoder (SAE), scores the features against baselines, and projects concept-associated features back to cells for analyses of single-cell and targeted spatial transcriptomic data.

:::{figure} _static/figure1.png
:alt: The interpreTFM workflow (panels a–f) and the downstream interpretability analyses (panels g–i).
:width: 100%

**Figure 1.** The interpreTFM workflow: expression profiles are tokenized for a frozen foundation model (a, b), layer activations are scored with the transcriptomic interpretability score (c) and decomposed by a sparse autoencoder (d), features are evaluated by held-out concept alignment and cell-level specificity (e) and projected to cells (f), and the resulting profiles support layer-wise comparison, niche discovery and interface analysis (g–i).
:::

## What interpreTFM measures

- **Transcriptomic interpretability score (TIS):** whether variation along a representation dimension is reflected in measured expression.
- **Held-out concept alignment:** whether SAE features mark panel-observable biological concepts on held-out gene-token rows or held-out whole genes, compared with raw hidden dimensions and an untrained SAE of the same width.
- **Population specificity (GSI, MHR@100):** how specific cell-level features are to cell populations, compared with single genes.
- **Spatial niches:** niches built from the features, compared with expression-only and label-composition baselines.
- **Interfaces:** program enrichment on cell-group interfaces under a calibrated statistic, with checks for transcript spillover.
- **Ligand–receptor co-localization:** a model-free screen that places candidate pairs in tissue context.

Each measure is meant to be read next to its baseline. The [concepts page](concepts.md) explains them in plain language, and the [best practices](best_practices.md) list how to report them.

Start with the [installation guide](installation.md) (which environment you need), the [tutorial](tutorials/cosmx_crop.md) (the spatial stages on a crop of the public CosMx section, on a CPU) and [running a pipeline](pipeline/running.md) (one command, one configuration file). The manuscript's layers are reproduced from `configs/paper/` ([reproducing the manuscript](reproducing.md)), and the trained SAEs are exported for the Hugging Face Hub ([stage `sae`](pipeline/stage_sae.md), `interpretfm.hub`).

```{toctree}
:maxdepth: 2
:caption: User guide

installation
tutorials/cosmx_crop
concepts
data
usage
best_practices
```

```{toctree}
:maxdepth: 2
:caption: Pipeline

pipeline/configuration
pipeline/running
pipeline/run_directory
pipeline/stage_ingest
pipeline/stage_extract
pipeline/stage_tis
pipeline/stage_sae
pipeline/stage_concepts
pipeline/stage_concept_eval
pipeline/stage_gene_readout
pipeline/stage_cells
pipeline/stage_layers
pipeline/stage_niches
pipeline/stage_interfaces
pipeline/stage_lr_screen
pipeline/stage_report
reproducing
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
faq
roadmap
changelog
citation
```
