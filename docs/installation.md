# Installation

## Requirements

- Linux x86-64. The environment files were exported from working Linux installations with CUDA 12.
- Python 3.8 to 3.11. Continuous integration runs the tests on each of these versions; the main environment uses Python 3.8, and the Geneformer and C2S-Scale environments use 3.10 and 3.9.
- An NVIDIA GPU for activation extraction and SAE training. TIS, concept alignment and the spatial analyses run on CPU from existing activation stores, though large stores take time.

## Which environment do I need?

| You want to | Environment | GPU |
|---|---|---|
| run the stages after extraction on an existing activation store, a published SAE or projected features (TIS, SAE training and evaluation, concept alignment, cell-level and spatial stages, the report, the tutorial) | `interpretfm` (`envs/interpretfm.yml`), or any Python 3.8–3.11 with `pip install -e .` | no (SAE training is faster on one) |
| extract activations from scGPT | `interpretfm` plus `nnsight` and the scGPT checkpoint | yes |
| extract activations from Geneformer | `geneformer` (`envs/geneformer.yml`) plus the Geneformer package and weights | yes |
| extract activations from C2S-Scale | `c2s` (`envs/c2s.yml`) plus the model folder | yes |
| build the documentation | `interpretfm-docs` (`envs/docs.yml`) or `docs/requirements.txt` | no |

The model environments are needed for the `extract` stage only. `interpretfm run` starts that stage as a child process in the environment named by `model.env` of the configuration and runs everything else in the interpreter that runs the command:

```yaml
model:
  env: {conda: geneformer}            # a named conda environment
  # env: {prefix: /opt/envs/geneformer}   # a conda environment by path
  # env: {python: /opt/envs/geneformer/bin/python}   # any interpreter with the package importable
  # env: {container: geneformer.sif}  # an Apptainer/Singularity image, or a Docker/Podman image name
  # env: {}                           # run in-process (the current interpreter holds the model's stack)
```

A configuration that starts from a cached store (`stages.extract.store_root`) never starts the model environment, and `model.env: {}` keeps a run in-process. Install the package in each model environment with `python -m pip install --no-deps -e .` (or export `PYTHONPATH=<repo>/src`), so that the child process can import it.

## Conda environments (recommended)

scGPT, Geneformer and C2S-Scale depend on conflicting library versions, so each model has its own environment. Activation stores share one format, so all later stages run in the main environment.

| Environment | Use | File |
|---|---|---|
| `interpretfm` | analysis, SAE training and evaluation, scGPT extraction | `envs/interpretfm.yml` |
| `geneformer` | Geneformer extraction | `envs/geneformer.yml` |
| `c2s` | C2S-Scale extraction | `envs/c2s.yml` |

```bash
git clone https://github.com/OliiverHu/interpreTFM.git
cd interpreTFM
conda env create -f envs/interpretfm.yml
conda activate interpretfm
python -m pip install -e .
python -m pip install --no-deps nnsight==0.4.5    # scGPT extraction only
```

Repeat with `envs/geneformer.yml` or `envs/c2s.yml` for the other models, and run `python -m pip install --no-deps -e .` inside each of them: the pipeline's `extract` stage runs inside the model's environment as a child process (`model.env: {conda: geneformer}` in the configuration) and imports the package there; everything after extraction runs in `interpretfm`. Without an install, exporting `PYTHONPATH=<repo>/src` also works, since the child inherits it. The environment files were exported from the installations that produced the manuscript's activation stores; `geneformer.yml` and `c2s.yml` match them version for version, and `interpretfm.yml` holds that installation's versions of every package interpreTFM uses (Python 3.8, PyTorch 2.1.2 with CUDA 12.1 wheels, scikit-learn 1.3.2, scanpy 1.9.8; the fresh-extraction tests reproduce the stores from them). Container images of the three environments can be built from `containers/` (not tested on the development machine; see `containers/README.md`) and used with `model.env: {container: <image>}`.

Notes:

- Install in editable mode (`-e`). Some modules import plotting helpers from `plot/` at the repository root, which a regular (non-editable) install does not contain.
- The Geneformer package is not on PyPI. In the `geneformer` environment, install it from its [Hugging Face repository](https://huggingface.co/ctheodoris/Geneformer) (`git lfs install`, clone the repository, then `python -m pip install .` inside it).
- `nnsight` is imported by the scGPT and C2S-Scale model loaders (extraction only). It is not in `envs/interpretfm.yml`: its package metadata requires torch >= 2.4, while the manuscript environment runs it with torch 2.1.2, so install it after the environment with `--no-deps` as above (`envs/c2s.yml` lists it; the C2S-Scale environment has a newer torch).
- `python -m pip install -e .` only adds packages that are missing; it does not change versions that the environment already provides. A plain `python -m pip install .` also works for the pipeline (`interpretfm`); only some 0.9 analysis modules import plotting helpers from `plot/`, which is why the editable install is recommended.

## pip only

Install PyTorch for your platform first (see [pytorch.org](https://pytorch.org)), then:

```bash
python -m pip install -e ".[models]"      # adds transformers, nnsight and datasets
```

Optional extras: `hub` (`safetensors` and `huggingface_hub`, to export trained SAEs or load the published ones with `interpretfm.hub`, and to download the tutorial data), `annotation` (GO hierarchy tools used by `interp_pipeline.downstream.f1.go_reduce`), `docs` (documentation build) and `dev` (tests and linting).

## Model weights and data

Weights, datasets and activation stores are not distributed with the code.

| Model | Source | Notes |
|---|---|---|
| scGPT | [scGPT repository](https://github.com/bowang-lab/scGPT), whole-human checkpoint | the checkpoint folder must contain `best_model.pt` and `vocab.json` |
| Geneformer | [Hugging Face: ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) | a model folder with `config.json` and the weights (the top level of a full clone is the V2 104M-parameter model used for the manuscript; the sub-folders hold the other sizes); the tokenizer and the token dictionaries come from the installed Geneformer package (`stages.extract.geneformer.model_version` selects V1/V2 dictionaries) |
| C2S-Scale | [Hugging Face: vandijklab/C2S-Scale-Gemma-2-2B](https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-2B) | a local model folder (`config.json`, `tokenizer.model`, safetensors weights; `huggingface-cli download vandijklab/C2S-Scale-Gemma-2-2B --local-dir <folder>`). The 2B model loads in float32 (about 10 GB of GPU memory) unless `stages.extract.c2s_scale.model_dtype` says otherwise |

Check each model's license before use. See [data and configuration formats](data.md) for the expected AnnData fields.

## Check the installation

```bash
python -c "import interp_pipeline; print(interp_pipeline.__version__)"
python -m pip install -e ".[dev]"
python -m pytest
```

The tests run on CPU without data or model weights. Tests that need a real activation store are skipped with a message that names the environment variable to set.

## Build the documentation

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The API reference is generated from the source tree with heavy dependencies mocked, so the package does not need to be installed for this step. `envs/docs.yml` is the environment used to build these pages.
