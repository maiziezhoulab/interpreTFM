# Configuration examples

| File | Used by | Content |
|---|---|---|
| `paper/<dataset>_<model>_layer<N>.yaml` | `interpretfm run ... --env-file paper/paths.env` | The manuscript's parameters for the six representative layers, starting from the cached stores and checkpoints; data locations come from `${INTERPRETFM_*}` placeholders (`paper/paths.env.example`; the filled-in `paper/paths.env` is git-ignored). See the documentation page "Reproducing the manuscript". |
| `paper/extract_cosmx_<model>.yaml` | `interpretfm run ... --env-file paper/paths.env --stages ingest,extract` | Fresh extraction of 2,000 seeded CosMx cells (scGPT, Geneformer, C2S-Scale) in the model's own environment on a GPU, the input of the fresh-extraction reproduction tests (`INTERPRETFM_REPRO_GPU=1`); needs the `INTERPRETFM_*_ENV`, `INTERPRETFM_GENEFORMER_MODEL` and `INTERPRETFM_C2S_MODEL` variables of `paper/paths.env.example`. |
| `paper/reinit_cosmx_<model>_<weights>.yaml` | `interpretfm run ... --env-file paper/paths.env --stages ingest,extract,sae,concepts,concept_eval` | Randomly re-initialized scGPT and Geneformer (`stages.extract.weights`: `trained`, `random_table`, `random_all`) at CosMx Layer 2 on the GSI/MHR cell sample (`paper/cosmx_<model>_sample_cells.txt.gz`): the five SAEs, the λ rule and concept F1, from the cached stores and checkpoints (`INTERPRETFM_REINIT_CACHE`); without `store_root` the stores are re-extracted on a GPU. |
| `paper/readout_controls_<dataset>_<model>_layer<N>.yaml` | `interpretfm run ... --env-file paper/paths.env --stages ingest,extract,sae,concepts,gene_readout` | The post hoc controls of the gene-level readout for the six representative layers: the gene-symbol controls, the representations with their symbol component removed and (CosMx scGPT, Geneformer) the re-initialized models, on the genes observed by all three models (`paper/<dataset>_readout_genes.txt`) and the manuscript's splits. |
| `model_specs.example.json` | `python -m interp_pipeline.layer_experiments.all_layer_sae_f1 --model-spec-json ...` | One entry per model: activation-store root, layers to process, and how cell-level rows are read (`activation_pooling`, `token_value`). The layer lists are the 29 model-layer combinations evaluated in the manuscript. |

Replace every `/path/to/...` with your own location before running a stage.
Stored layer identifiers are zero-based (`layer_1.norm2` is display Layer 2).
See the [usage guide](https://interpretfm.readthedocs.io/en/latest/usage.html) for details.
