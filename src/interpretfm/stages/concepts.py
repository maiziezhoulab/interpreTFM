"""Stage ``concepts``: the panel concept universe (gene x concept membership) and the gene table.

Parameters (``stages.concepts``): ``sources`` (``[GO:BP, GO:MF, GO:CC, REAC, KEGG]``), ``fdr``
(0.05), ``min_overlap`` (2), ``organism`` (``hsapiens``), ``query_key`` (``symbol`` |
``ensembl``; default: the dataset's gene key), ``cache_dir`` (g:Profiler cache; default
``<stage dir>/cache``), ``offline_matrix`` (a gene x concept 0/1 CSV used instead of
g:Profiler; ``offline_terms`` adds names) and ``gene_keys``
(``{symbol_column, ensembl_column}``: extra ``var`` columns that name the genes).

Outputs: ``universe.json`` (summary and manifest), ``universe/gene_by_concept.csv`` (genes as
rows keyed by the dataset gene key, concepts as columns), ``universe/terms.tsv``,
``universe/manifest.json``, ``universe/enrichment.csv`` (online mode) and ``genes.csv``
(key, symbol, Ensembl id per measured gene).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from interpretfm.concepts import params_dict, params_from_mapping, universe_from_gprofiler, universe_from_matrix
from interpretfm.genes import GeneTable
from interpretfm.params import param_path
from interpretfm.runner import StageContext, register_stage

OUTPUTS = ["universe.json", "genes.csv"]


def load_gene_table(cfg, params: Dict[str, Any]) -> GeneTable:
    """The :class:`~interpretfm.genes.GeneTable` of the configured dataset."""
    import anndata as ad

    adata = ad.read_h5ad(cfg.dataset.h5ad, backed="r")
    gk = params.get("gene_keys") or {}
    return GeneTable.from_adata(adata, cfg.dataset.gene_id, symbol_column=gk.get("symbol_column"),
                                ensembl_column=gk.get("ensembl_column"))


@register_stage("concepts", inputs=["ingest"], outputs=OUTPUTS)
def run_concepts(ctx: StageContext) -> Optional[Dict[str, Any]]:
    """The panel concept universe (gene x concept membership) and the gene table."""
    cfg = ctx.config
    p = params_from_mapping(ctx.params)
    genes = load_gene_table(cfg, ctx.params)
    genes.to_frame().to_csv(ctx.path("genes.csv"), index=False)
    if p.offline_matrix:
        path = param_path(cfg, p.offline_matrix, "concepts", "offline_matrix")
        terms_path = ctx.params.get("offline_terms")
        if terms_path:
            terms_path = param_path(cfg, terms_path, "concepts", "offline_terms")
        ctx.log.info("concepts: loading the offline gene x concept table %s", path)
        universe = universe_from_matrix(genes, path, min_overlap=p.min_overlap, terms_path=terms_path)
    else:
        cache = param_path(cfg, p.cache_dir, "concepts", "cache_dir", must_exist=False) if p.cache_dir else ctx.path("cache", "x").parent
        ctx.log.info("concepts: g:Profiler enrichment of %d genes (sources %s, FDR %g, cache %s)", len(genes.keys),
                     list(p.sources), p.fdr, cache)
        universe = universe_from_gprofiler(genes, p, cache, query_name=cfg.run.name)
    universe.write(ctx.path("universe", "manifest.json").parent)
    sizes = universe.concept_sizes
    summary = {"params": params_dict(p), "gene_key": genes.kind, "n_genes": len(genes.keys),
               "n_concepts": int(universe.membership.shape[1]),
               "n_concepts_5_100": int(((sizes >= 5) & (sizes <= 100)).sum()),
               "concept_size": {"min": int(sizes.min()) if len(sizes) else 0, "median": float(sizes.median()) if len(sizes) else 0.0,
                                "max": int(sizes.max()) if len(sizes) else 0},
               "n_genes_in_any_concept": int((universe.membership.sum(axis=1) > 0).sum()),
               "manifest": universe.manifest}
    ctx.write_json("universe.json", summary)
    ctx.log.info("concepts: %d concepts over %d genes (%d with 5-100 genes)", summary["n_concepts"], summary["n_genes"],
                 summary["n_concepts_5_100"])
    return {k: summary[k] for k in ("n_genes", "n_concepts", "n_concepts_5_100")}
