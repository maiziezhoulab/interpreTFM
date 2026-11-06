# Conda environments

scGPT, Geneformer and C2S-Scale need conflicting library versions, so each model has its own environment. Activation stores share one format, so all stages after extraction run in the main environment.

| Environment | Use | File |
|---|---|---|
| `interpretfm` | analysis, SAE training and evaluation, scGPT extraction | `interpretfm.yml` |
| `geneformer` | Geneformer extraction | `geneformer.yml` |
| `c2s` | C2S-Scale extraction | `c2s.yml` |
| `interpretfm-docs` | building the documentation | `docs.yml` |

```bash
conda env create -f envs/interpretfm.yml
conda activate interpretfm
python -m pip install -e .
python -m pip install --no-deps nnsight==0.4.5    # scGPT and C2S-Scale extraction only
```

Notes:

- The model files are exports of working Linux x86-64 installations with CUDA 12 (`conda env export`, machine-specific lines removed). Pip-only packages are listed under `pip:`. `interpretfm.yml` keeps, at the exported versions, the packages that interpreTFM imports and their dependencies (PyTorch 2.1.2 with the CUDA 12.1 wheels, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.3.2, scanpy 1.9.8, anndata 0.9.2, transformers 4.46.3, ...); the research-only tools of the original installation (scvi-tools and its JAX stack, wandb, R) are left out, because some of their pinned wheels are no longer on PyPI.
- `nnsight` (imported by the scGPT and C2S-Scale model loaders, so needed for extraction only) is installed after the environment with `python -m pip install --no-deps nnsight==0.4.5`: its package metadata asks for torch >= 2.4 and torchvision, which the manuscript environment does not hold, while the package itself runs with torch 2.1.2 (the extraction tests reproduce the manuscript's stores with this combination).
- Install this repository with `python -m pip install --no-deps -e .` inside each environment after creating it: the pipeline runs the `extract` stage inside the model's environment (`model.env` in the configuration) and imports the package there; `PYTHONPATH=<repo>/src` also works. `geneformer.yml` and `c2s.yml` are version-for-version the environments that wrote the manuscript's activation stores, and `interpretfm.yml` holds that environment's versions of every package interpreTFM uses (checked against them; the fresh-extraction tests reproduce the stores from them). `c2s.yml` lists `typing_extensions`, `packaging`, `matplotlib-inline`, `prompt-toolkit` and `wcwidth` twice, once under conda and once under `pip:`, because the exported environment holds both installs; the pip versions are the active ones.
- `geneformer.yml`: the Geneformer package is not on PyPI; install it from <https://huggingface.co/ctheodoris/Geneformer> (the manuscript's stores were tokenized with the repository at commit `873a2fd` of 2025-08-18, package version 0.1.0; its tokenizer and gene dictionaries are what the `geneformer` adapter uses). PyTorch comes from the PyTorch CUDA 12.1 index (`--extra-index-url` in the file).
- Container images of the three model environments can be built from `containers/` (one Dockerfile per environment, an Apptainer definition, a GitHub Actions workflow started by hand); they have not been built on the development machine.
- `interpretfm.yml` lists `flash-attn` as a commented-out line, because it needs a CUDA toolchain to build; interpreTFM does not import it, and the line can be uncommented where a CUDA toolchain is available.
- Model weights are not part of any environment; see the installation guide.
