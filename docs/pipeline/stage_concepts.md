# Stage `concepts`: the panel concept universe

Directory `05_concepts/`. Input: `ingest` (the dataset). Core environment; needs the network unless `offline_matrix` is set.

## What it computes

A binary gene × concept table over the measured genes. Online, the panel is enriched with g:Profiler (organism `hsapiens`; Gene Ontology biological process, molecular function and cellular component, Reactome and KEGG; FDR-corrected at 0.05) and every enriched term becomes a concept whose members are the measured genes in its intersection; terms with fewer than `min_overlap` (2) measured genes are dropped. g:Profiler results are cached on disk (`cache_dir`, default `05_concepts/cache/`), so a rerun does not query the service again, and `universe/manifest.json` records the sources, thresholds, client version and retrieval time.

Offline, `offline_matrix` names a CSV with genes as rows (any name the dataset knows: the gene key, a symbol or an Ensembl id) and concepts as columns (0/1); genes not on the panel are dropped, panel genes absent from the file are non-members, and the same `min_overlap` filter applies. `offline_terms` (TSV with `term_id`, `name`, `source`) adds concept names.

The universe is restricted to the genes of the panel: it tests alignment with concepts observable on that panel, not genome-wide pathway discovery.

**Gene keys.** Every gene-level table is keyed by `dataset.gene_id` (the `var` column or index, declared as `symbol` or `ensembl`). The stage also writes `genes.csv` with the symbol and Ensembl id of every gene when the `var` table knows them (`gene_keys.symbol_column` / `gene_keys.ensembl_column`, or the usual column names), so later stages can map token ids, g:Profiler output and offline tables back to the key. Token ids that are vocabulary numbers (scGPT) are translated through a JSON vocabulary (`token_vocab` of the downstream stages, or `<store>/token_vocab.json` written by the extraction).

## Parameters (`stages.concepts`)

| key | default | meaning |
|---|---|---|
| `sources` | `[GO:BP, GO:MF, GO:CC, REAC, KEGG]` | g:Profiler sources |
| `fdr` | 0.05 | significance threshold (FDR) |
| `min_overlap` | 2 | minimum measured genes per concept |
| `organism` | `hsapiens` | |
| `query_key` | the dataset key | send `symbol` or `ensembl` names to g:Profiler |
| `cache_dir` | `05_concepts/cache` | g:Profiler cache |
| `offline_matrix` | null | gene × concept CSV (offline mode) |
| `offline_terms` | null | concept names for the offline matrix |
| `gene_keys` | `{}` | `symbol_column`, `ensembl_column` in `adata.var` |

## Outputs

```
05_concepts/
  universe.json                 n_genes, n_concepts, n_concepts_5_100, concept sizes, manifest
  genes.csv                     key, symbol, ensembl
  universe/gene_by_concept.csv  genes (key) x concepts, 0/1
  universe/terms.tsv            term_id, name, source, n_genes[, p_value]
  universe/manifest.json
  universe/enrichment.csv       the raw g:Profiler table (online mode)
```
