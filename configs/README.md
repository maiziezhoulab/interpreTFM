# Configuration examples

| File | Used by | Content |
|---|---|---|
| `paper/<dataset>_<model>_layer<N>.yaml` | `interpretfm run ... --env-file paper/paths.env` | The manuscript's parameters for the six representative layers, starting from the cached stores and checkpoints; data locations come from `${INTERPRETFM_*}` placeholders (`paper/paths.env.example`; the filled-in `paper/paths.env` is git-ignored). See the documentation page "Reproducing the manuscript". |
| `paper/extract_cosmx_<model>.yaml` | `interpretfm run ... --env-file paper/paths.env --stages ingest,extract` | Fresh extraction of 2,000 seeded CosMx cells (scGPT, Geneformer, C2S-Scale) in the model's own environment on a GPU, the input of the fresh-extraction reproduction tests (`INTERPRETFM_REPRO_GPU=1`); needs the `INTERPRETFM_*_ENV`, `INTERPRETFM_GENEFORMER_MODEL` and `INTERPRETFM_C2S_MODEL` variables of `paper/paths.env.example`. |
| `model_specs.example.json` | `python -m interp_pipeline.layer_experiments.all_layer_sae_f1 --model-spec-json ...` | One entry per model: activation-store root, layers to process, and how cell-level rows are read (`activation_pooling`, `token_value`). The layer lists are the 29 model-layer combinations evaluated in the manuscript. |

Replace every `/path/to/...` with your own location before running a stage.
Stored layer identifiers are zero-based (`layer_1.norm2` is display Layer 2).
See the [usage guide](https://interpretfm.readthedocs.io/en/latest/usage.html) (placeholder URL) for details.
