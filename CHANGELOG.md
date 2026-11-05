# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- v1 pipeline core (in development): the `interpretfm` command (`validate`, `run`, `status`, `report`, `version`), a typed YAML configuration (`run`, `dataset`, `model`, `stages`, `output`) with the spatial unit declared once as `dataset.spatial.um_per_unit`, a cached and resumable stage runner with `--stages`, `--force` and `--dry-run`, run and stage `manifest.json`, a subprocess boundary to per-model conda environments, the `ingest` stage, stubs for the remaining stages, toy fixtures (`interpretfm.testing`) and `configs/toy.yaml`.
- Documentation pages for the configuration, running a pipeline and the run directory; API page for the pipeline core.
- Model-level stages of the v1 pipeline: `tis` (TIS with shuffled and PCA controls), `sae` (the biased ReLU SAE trainer with the fixed resampler, QC, the λ rule and the untrained baseline), `concepts` (g:Profiler concept universe with cache, an offline matrix mode and configurable gene keys), `concept_eval` (held-out concept F1 under the row and whole-gene splits for SAE, raw and untrained units, with cell de-duplication) and `gene_readout` (gene-level linear readout with expression and random controls); the stage order gains `concept_eval` and `gene_readout`. Shared helpers `interpretfm.stores` (activation-store reading, cached stores through `store_root`), `interpretfm.genes` (gene keys, token vocabularies) and `interpretfm.params`; `write_toy_run(store=True)` writes a synthetic activation store. One documentation page per stage.

### Changed

- Version `1.0.0.dev0`; `pyyaml` is a dependency.

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
