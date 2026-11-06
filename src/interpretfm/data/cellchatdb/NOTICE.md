# CellChatDB (human) ligand–receptor tables

Files: `cellchatdb_human_interaction.csv.gz` (3,233 interactions: `interaction_name`, `pathway_name`,
`ligand`, `receptor`, `annotation`) and `cellchatdb_human_complex.csv.gz` (338 complexes:
`complex_name`, `subunit_1` … `subunit_5`).

- Source: `CellChatDB.human` of the CellChat R package (Jin et al., *Nature Communications* 2021,
  <https://doi.org/10.1038/s41467-021-21246-9>; Jin et al., *Nature Protocols* 2025), CellChat version 2.2.0
  (<https://github.com/jinworks/CellChat>).
- Export: the two data frames written as-is with `write.csv` (the `interaction` table restricted to the five
  columns above, the `complex` table with its row names as `complex_name`) in November 2026; no row was
  added, removed or edited.
- Licence: CellChat, including its database, is distributed under the GNU General Public License v3
  (GPL-3), the same licence as this package. These tables are redistributed under GPL-3 with this notice;
  the copyright is the CellChat authors'.

Use in this package: `interpretfm.lr_screen.load_cellchatdb` reads the two files by default; the
`lr_screen` stage accepts `database:` / `complexes:` CSV paths to use another ligand–receptor table in the
same layout. CellChat itself (the R package and its communication-probability model) is not used.
