# Third-party notices

interpreTFM is licensed under the GNU General Public License v3.0 or later ([`LICENSE`](LICENSE)).
It contains code adapted from the projects below. Their licenses are compatible with GPL-3.0, and their notices are kept with the code.

## scGPT

- **Location:** `src/interp_pipeline/vendor/scgpt_local/` (model definition, pretrained-weight loading and tokenizer helpers)
- **Source:** <https://github.com/bowang-lab/scGPT>
- **License:** MIT License, Copyright (c) 2022 suber. Full text: [`src/interp_pipeline/vendor/scgpt_local/LICENSE.scGPT`](src/interp_pipeline/vendor/scgpt_local/LICENSE.scGPT)
- **Changes:** reduced to the parts needed for inference and adapted to the interpreTFM extraction interface. Each file carries a notice.

## cell2sentence (C2S-Scale)

- **Location:** `src/interp_pipeline/vendor/c2s_local/` (cell-sentence generation, prompt formatting and model loading helpers; the files in `prompts/` are unmodified copies)
- **Source:** <https://github.com/vandijklab/cell2sentence>
- **License:** Apache License 2.0, Copyright 2025 Yale University. Full text: [`src/interp_pipeline/vendor/c2s_local/LICENSE.cell2sentence`](src/interp_pipeline/vendor/c2s_local/LICENSE.cell2sentence)
- **Changes:** adapted to the interpreTFM extraction interface; prompt formatting uses the first template of each task instead of a random one. Each Python file carries a notice. The upstream project has no NOTICE file.

## CellChatDB (human ligand–receptor database)

- **Location:** `src/interpretfm/data/cellchatdb/` (`cellchatdb_human_interaction.csv.gz`, `cellchatdb_human_complex.csv.gz`, `NOTICE.md`)
- **Source:** `CellChatDB.human` of CellChat 2.2.0, <https://github.com/jinworks/CellChat> (Jin et al., *Nature Communications* 2021; Jin et al., *Nature Protocols* 2025)
- **License:** GNU General Public License v3 (GPL-3), the license of this package. Copyright: the CellChat authors.
- **Changes:** none; the two data frames are exported as CSV (the interaction table restricted to five columns). The CellChat package itself is not used; only its database table is redistributed for the ligand–receptor screen (`interpretfm.lr_screen`).

## Tutorial bundle and exported SAEs (distributed outside the repository)

- **Tutorial bundle** (`tutorial/` of the Hugging Face repository): files derived from the tutorial's crop of the public CosMx human NSCLC section (NanoString CosMx SMI lung cancer dataset, section 8): the retained SAE features of its 2,822 cells (`cells_layer_1.norm2.h5ad`: feature values, QC flags and cell ids only; the stages take the cell types and coordinates from the crop), one shard of the scGPT activation store restricted to 80 of them, plus `cosmx_author_groups.csv` (the manuscript's author-type grouping). The crop itself (counts, author cell types, coordinates) is not redistributed: the CosMx data carry no explicit licence (NanoString states none, and the Hugging Face repository of the Nicheformer data release has no dataset card), so `interpretfm.hub.tutorial_crop` builds it on the user's machine from the public file `nanostring_lung_annotated.h5ad` of `theislab/SpatialCorpus-110M` at revision `5494a62f93d612cbc8a4ef773dc82f6b1e7a1218`. The bundle's `README.md` names this source and credits the data: He S. *et al.*, *Nat Biotechnol* 40, 1794–1806 (2022), doi:10.1038/s41587-022-01483-z; NanoString / Bruker Spatial Biology; the Nicheformer data release (Tejada-Lapuerta A., Schaar A.C. *et al.*, *Nat Methods* 22, 2525–2538 (2025)).
- **Trained SAE weights** (`<model>/<dataset>/<layer>/` of the Hugging Face repository): trained by this package on activations of scGPT (MIT), Geneformer (Apache-2.0) and C2S-Scale-Gemma-2-2B (CC BY 4.0, built on Gemma 2); released under GPL-3.0-or-later with a model card naming the base model and its licence. The C2S-Scale cards credit the van Dijk lab under CC BY 4.0 and carry the Gemma notice ("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"); the base models themselves are not redistributed.

## Dependencies and model weights

Dependencies installed from PyPI or conda (for example PyTorch (BSD-3), Transformers (Apache-2.0), scanpy (BSD-3), nnsight (MIT), safetensors and huggingface_hub (Apache-2.0), and the Geneformer package (Apache-2.0)) are not redistributed here and keep their own licenses. Pretrained model weights (scGPT, Geneformer, C2S-Scale) are not included; check each model's license before use.
