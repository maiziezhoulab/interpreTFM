# Container images for the model environments

One image per model environment, built from the conda files in `envs/` with the repository installed, on a CUDA runtime base image. They serve `model.env: {container: <image>}` of the pipeline ([running a pipeline](../docs/pipeline/running.md), [stage extract](../docs/pipeline/stage_extract.md)): the runner starts `python -m interpretfm.cli extract ...` inside the image with the run directory and the data paths bound in.

| File | Environment | Base image | Contents |
|---|---|---|---|
| `Dockerfile.interpretfm` | `envs/interpretfm.yml` (scGPT extraction, SAE training, every CPU stage) | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | conda env + the package |
| `Dockerfile.geneformer` | `envs/geneformer.yml` | same | conda env + the Geneformer package (pinned commit of the Hugging Face repository, dictionaries only, no weights) + the package |
| `Dockerfile.c2s` | `envs/c2s.yml` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` | conda env + the package |

**Build status: not tested.** The machine the pipeline was developed on has no Docker, Podman or Apptainer, so these files have been written against the environment files but never built or run. The GitHub Actions workflow `.github/workflows/containers.yml` builds them on demand (`workflow_dispatch` only, never on push) and can push them to GHCR; run it once before relying on an image, and expect to adjust the base image or the conda solve.

## Building locally

From the repository root (the build context is the whole repository, so that `pip install -e .` sees `src/`):

```bash
docker build -f containers/Dockerfile.interpretfm -t interpretfm:1.0 .
docker build -f containers/Dockerfile.geneformer  -t interpretfm-geneformer:1.0 .
docker build -f containers/Dockerfile.c2s         -t interpretfm-c2s:1.0 .
```

Model weights are not part of any image. Keep them on the host and reference them through the configuration (`model.checkpoint`, `${INTERPRETFM_MODELS}`); the runner binds those paths into the container.

## Apptainer / Singularity

HPC systems usually run Apptainer. The simplest route is to convert a built Docker image:

```bash
apptainer build interpretfm-geneformer.sif docker-daemon://interpretfm-geneformer:1.0      # from the local Docker daemon
apptainer build interpretfm-geneformer.sif docker://ghcr.io/<owner>/interpretfm-geneformer:1.0   # from a registry
```

To build without Docker, write a definition file that bootstraps from the same base image and repeats the Dockerfile's steps; `interpretfm.def` is that translation for `Dockerfile.interpretfm` (the `%post` section is the Dockerfile's `RUN` lines, `%files` its `COPY`, `%environment` its `ENV`). The other two follow the same pattern; the Geneformer one adds the `git lfs` steps of `Dockerfile.geneformer`. Build with `apptainer build --fakeroot interpretfm.sif containers/interpretfm.def` from the repository root.

Use the image in a configuration:

```yaml
model:
  env: {container: /path/to/interpretfm-geneformer.sif}     # or ghcr.io/<owner>/interpretfm-geneformer:1.0 for Docker
```

The runtime is found on `PATH` (`apptainer`, `singularity`, `docker`, `podman`, in that order) or named by `INTERPRETFM_CONTAINER_RUNTIME`. Apptainer runs with `--nv`; Docker with `--gpus all --user <uid>:<gid>`; Podman with `--device nvidia.com/gpu=all`.
