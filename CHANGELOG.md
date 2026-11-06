# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- v1 pipeline core (in development): the `interpretfm` command (`validate`, `run`, `status`, `report`, `version`), a typed YAML configuration (`run`, `dataset`, `model`, `stages`, `output`) with the spatial unit declared once as `dataset.spatial.um_per_unit`, a cached and resumable stage runner with `--stages`, `--force` and `--dry-run`, run and stage `manifest.json`, a subprocess boundary to per-model conda environments, the `ingest` stage, stubs for the remaining stages, toy fixtures (`interpretfm.testing`) and `configs/toy.yaml`.
- Documentation pages for the configuration, running a pipeline and the run directory; API page for the pipeline core.
- Model-level stages of the v1 pipeline: `tis` (TIS with shuffled and PCA controls), `sae` (the biased ReLU SAE trainer with the fixed resampler, QC, the λ rule and the untrained baseline), `concepts` (g:Profiler concept universe with cache, an offline matrix mode and configurable gene keys), `concept_eval` (held-out concept F1 under the row and whole-gene splits for SAE, raw and untrained units, with cell de-duplication) and `gene_readout` (gene-level linear readout with expression and random controls); the stage order gains `concept_eval` and `gene_readout`. Shared helpers `interpretfm.stores` (activation-store reading, cached stores through `store_root`), `interpretfm.genes` (gene keys, token vocabularies) and `interpretfm.params`; `write_toy_run(store=True)` writes a synthetic activation store. One documentation page per stage.
- Cell-level stages and the report of the v1 pipeline: `ingest` now checks raw counts, gene names, labels and coordinates, flags low-quality cells (`dataset.qc`, defaults 20 transcripts / 10 genes; kept, never dropped), reports the panel's coverage of the model vocabulary (`model.vocab`) and writes a cell table in µm; `cells` (feature retention by the fixed rule, projection of features to cells, GSI excess and MHR@k for SAE features, raw dimensions and single genes on the same cells); `layers` (per-criterion layer table with baselines and best-layer notes, no composite score); `report` (`report.json` with every metric beside its baseline and one self-contained HTML page), wired to `interpretfm report`. Helpers `interpretfm.ingest`, `interpretfm.cells`, `interpretfm.layers`, `interpretfm.report`.
- Manuscript configurations for the six representative layers (`configs/paper/`), starting from the cached activation stores and SAE checkpoints and naming data only through `${INTERPRETFM_*}` placeholders (`configs/paper/paths.env.example`); reproduction tests (`tests/reproduction/`, marker `reproduction`, skipped unless the data are configured) for TIS gain, the λ rule and untrained baseline, concept F1 under both held-out units, concept counts and the gene-level readout, with placeholders for the cell-level and spatial numbers; `make repro`/`repro-test` and the documentation page "Reproducing the manuscript".
- Spatial niches (`niches` stage, replacing the `spatial` stub; skipped for data without coordinates): the manuscript's niche sweep (radii in µm converted through `dataset.spatial.um_per_unit`, k-means / GMM, `m` / `xm`, seeds, minimum cluster size, the score 1.5 ARI + 0.1 clip(z) + separability) with validation (purity against the tile-shuffle null, ARI/NMI against the labels), the expression-only (truncated SVD) and label-composition (scikit-learn, no squidpy) baselines at their own winner and at the model's configuration, optional expression preprocessing variants, seed stability, and niche concept labels (Wilcoxon inside vs outside, BH, top 25 by effect) with a cross-model Jaccard recipe; `interpretfm.niches`, `interpretfm.niche_concepts`; the niche numbers beside their baselines in `report.json` and the HTML page; the `cells` and `niches` sections of the paper configurations; reproduction tests T6-T9 and T13.
- `${NAME}` placeholders in configuration strings, filled from the environment or a `KEY=VALUE` file (`--env-file`, `load_config(env_file=...)`, `interpretfm.config.load_env_file`).

### Changed

- Version `1.0.0.dev0`; `pyyaml` is a dependency.
- The TIS PCA control defaults to `pca_of: activations` (the activations rotated to their principal components, the control the manuscript reports); `pca_of: expression` remains available.

## [0.9.0] - 2026-11-05

First public pre-release of the analysis package.

### Added

- Documentation site (Sphinx, Read the Docs): installation, concepts, data formats, usage, evaluation best practices, roadmap, FAQ and API reference.
- Package metadata (`interpretfm` distribution, version 0.9.0), optional extras `models`, `annotation`, `docs` and `dev`.
- Held-out concept alignment with row and whole-gene splits, including optional de-duplication of cells that appear more than once in an activation store (`dedup_cells`, `--f1-dedup-cells`).
- Example model specification for the all-layer runner (`configs/model_specs.example.json`).
- CPU tests that need no data: imports, command-line help, the quick-start examples on toy data, and held-out F1 regressions.
- Continuous integration (tests and documentation build), pre-commit configuration, contribution guide, code of conduct, security policy, citation file and third-party notices.

[Unreleased]: https://github.com/OliiverHu/interpreTFM/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/OliiverHu/interpreTFM/releases/tag/v0.9.0
