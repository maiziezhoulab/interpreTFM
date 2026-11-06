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

## Dependencies and model weights

Dependencies installed from PyPI or conda (for example PyTorch, Transformers, scanpy, and the Geneformer package) are not redistributed here and keep their own licenses. Pretrained model weights (scGPT, Geneformer, C2S-Scale) are not included; check each model's license before use.
