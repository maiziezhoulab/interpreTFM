"""The panel concept universe: a binary gene x concept table over the measured genes.

Concepts come from a g:Profiler enrichment of the panel (Gene Ontology biological process,
molecular function and cellular component, Reactome and KEGG; FDR-corrected at 0.05;
every term must overlap at least ``min_overlap`` measured genes), using the cached
client :class:`interp_pipeline.get_annotation.gprofiler_client.GProfilerClient`, or from
an offline table (a CSV with genes as rows and concepts as columns, 0/1 entries) when no
network is available or a fixed universe must be reused.

The result is keyed by the dataset's gene key (:class:`interpretfm.genes.GeneTable`).
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from interpretfm.genes import GeneTable

DEFAULT_SOURCES = ("GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG")


@dataclass
class UniverseParams:
    """``stages.concepts``: how the universe is built."""

    sources: Sequence[str] = DEFAULT_SOURCES
    fdr: float = 0.05
    min_overlap: int = 2
    organism: str = "hsapiens"
    query_key: Optional[str] = None  #: ``symbol`` | ``ensembl`` | None (the dataset's key)
    cache_dir: Optional[str] = None
    offline_matrix: Optional[str] = None
    gene_keys: Dict[str, Optional[str]] = field(default_factory=dict)  #: symbol_column / ensembl_column


@dataclass
class Universe:
    """Gene x concept membership (0/1 ``DataFrame`` keyed by gene key) and the concept table."""

    membership: Any  #: pandas.DataFrame, genes x concepts
    terms: Any  #: pandas.DataFrame: term_id, name, source, n_genes[, p_value]
    manifest: Dict[str, Any] = field(default_factory=dict)
    enrichment: Any = None  #: the raw g:Profiler result (DataFrame) when built online

    @property
    def concept_sizes(self):
        return self.membership.sum(axis=0).astype(int)

    def write(self, out_dir: Union[str, Path]) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.membership.to_csv(out_dir / "gene_by_concept.csv")
        self.terms.to_csv(out_dir / "terms.tsv", sep="\t", index=False)
        with open(out_dir / "manifest.json", "w") as fh:
            json.dump(self.manifest, fh, indent=2)
        if self.enrichment is not None:
            self.enrichment.to_csv(out_dir / "enrichment.csv", index=False)
        return out_dir

    @classmethod
    def read(cls, out_dir: Union[str, Path]) -> "Universe":
        import pandas as pd

        out_dir = Path(out_dir)
        membership = pd.read_csv(out_dir / "gene_by_concept.csv", index_col=0)
        membership.index = membership.index.astype(str)
        membership.columns = membership.columns.astype(str)
        terms = pd.read_csv(out_dir / "terms.tsv", sep="\t")
        manifest = {}
        if (out_dir / "manifest.json").exists():
            with open(out_dir / "manifest.json") as fh:
                manifest = json.load(fh)
        return cls(membership=membership.astype(np.int8), terms=terms, manifest=manifest)


def _as_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in x]
    s = str(x).strip()
    if s.startswith("["):
        try:
            return [str(v) for v in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    return [v.strip() for v in s.split(",") if v.strip()]


def membership_from_pairs(genes: GeneTable, term_genes: Dict[str, Sequence[str]], min_overlap: int):
    """0/1 ``DataFrame`` (gene key x term) from ``{term: [gene names]}``; names are resolved to
    keys and terms with fewer than ``min_overlap`` panel genes are dropped."""
    import pandas as pd

    key_row = {k: i for i, k in enumerate(genes.keys)}
    cols, data = [], []
    for term, names in term_genes.items():
        col = np.zeros(len(genes.keys), dtype=np.int8)
        for k in genes.resolve(names):
            if k is not None:
                col[key_row[k]] = 1
        if int(col.sum()) >= int(min_overlap):
            cols.append(str(term))
            data.append(col)
    if not cols:
        return pd.DataFrame(index=pd.Index(genes.keys, name="gene"), dtype=np.int8)
    return pd.DataFrame(np.stack(data, axis=1), index=pd.Index(genes.keys, name="gene"), columns=cols)


def universe_from_gprofiler(genes: GeneTable, params: UniverseParams, cache_dir: Union[str, Path],
                            query_name: str = "panel") -> Universe:
    """Enrich the panel with g:Profiler (cached under ``cache_dir``) and build the universe."""
    import pandas as pd

    from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec

    query = genes.names_for_query(params.query_key)
    client = GProfilerClient(cache_dir=str(cache_dir))
    spec = GProfilerSpec(organism=params.organism, user_threshold=float(params.fdr),
                         significance_threshold_method="fdr", sources=list(params.sources), no_evidences=False,
                         return_dataframe=True)
    res = client.profile(query, spec=spec, query_name=query_name)
    res = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    if res.empty:
        raise RuntimeError("g:Profiler returned no enriched terms for the panel")
    term_col = "native" if "native" in res.columns else "term_id"
    inter_col = "intersections" if "intersections" in res.columns else "intersection"
    pcol = next((c for c in ("p_value", "p_value_fdr", "p_value_adj") if c in res.columns), None)
    if term_col not in res.columns or inter_col not in res.columns:
        raise RuntimeError(f"g:Profiler output lacks term/intersection columns: {list(res.columns)[:30]}")
    if pcol is not None:
        res = res[res[pcol] <= float(params.fdr)]
    term_genes = {str(r[term_col]): _as_list(r[inter_col]) for _, r in res.iterrows()}
    membership = membership_from_pairs(genes, term_genes, params.min_overlap)
    keep = res[res[term_col].astype(str).isin(membership.columns)]
    terms = pd.DataFrame({
        "term_id": keep[term_col].astype(str).values,
        "name": keep["name"].astype(str).values if "name" in keep.columns else keep[term_col].astype(str).values,
        "source": keep["source"].astype(str).values if "source" in keep.columns else "",
        "n_genes": membership[keep[term_col].astype(str).values].sum(axis=0).values.astype(int),
    })
    if pcol is not None:
        terms["p_value"] = keep[pcol].values
    terms = terms.drop_duplicates("term_id").reset_index(drop=True)
    membership = membership[terms["term_id"].tolist()]
    manifest = {"source": "gprofiler", "organism": params.organism, "sources": list(params.sources),
                "fdr": float(params.fdr), "min_overlap": int(params.min_overlap), "query_key": params.query_key or genes.kind,
                "n_query_genes": len(query), "n_terms_enriched": int(len(res)), "n_concepts": int(membership.shape[1]),
                "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cache_dir": str(cache_dir),
                "gprofiler_official": _version("gprofiler")}
    return Universe(membership=membership, terms=terms, manifest=manifest, enrichment=res)


def universe_from_matrix(genes: GeneTable, path: Union[str, Path], min_overlap: int = 2,
                         terms_path: Optional[Union[str, Path]] = None) -> Universe:
    """Load a gene x concept 0/1 CSV (genes as rows, any known gene name) as the universe.

    Genes not on the panel are dropped, panel genes absent from the file are non-members
    of every concept, and concepts with fewer than ``min_overlap`` panel genes are dropped.
    ``terms_path`` (TSV with ``term_id``, ``name``, ``source``) adds concept names.
    """
    import pandas as pd

    raw = pd.read_csv(path, index_col=0)
    raw.index = raw.index.astype(str)
    term_genes = {str(c): raw.index[raw[c].astype(float) > 0].tolist() for c in raw.columns}
    membership = membership_from_pairs(genes, term_genes, min_overlap)
    n_unknown = int(sum(k is None for k in genes.resolve(raw.index)))
    terms = pd.DataFrame({"term_id": membership.columns.astype(str), "name": membership.columns.astype(str),
                          "source": [c.split(":")[0] if ":" in c else "" for c in membership.columns],
                          "n_genes": membership.sum(axis=0).values.astype(int)})
    if terms_path is not None:
        meta = pd.read_csv(terms_path, sep="\t" if str(terms_path).endswith((".tsv", ".txt")) else ",")
        meta.columns = [c.lower() for c in meta.columns]
        if "term_id" in meta.columns:
            meta["term_id"] = meta["term_id"].astype(str)
            terms = terms.drop(columns=[c for c in ("name", "source") if c in meta.columns]).merge(
                meta[[c for c in ("term_id", "name", "source") if c in meta.columns]], on="term_id", how="left")
            terms = terms[["term_id", "name", "source", "n_genes"]]
    manifest = {"source": "offline_matrix", "path": str(Path(path).resolve()), "min_overlap": int(min_overlap),
                "n_file_genes": int(raw.shape[0]), "n_file_genes_not_on_panel": n_unknown,
                "n_file_concepts": int(raw.shape[1]), "n_concepts": int(membership.shape[1])}
    return Universe(membership=membership, terms=terms, manifest=manifest)


def params_from_mapping(params: Dict[str, Any]) -> UniverseParams:
    from interpretfm.params import get_list, get_mapping, get_param

    gk = get_mapping(params, "gene_keys", "concepts")
    p = UniverseParams(
        sources=get_list(params, "sources", list(DEFAULT_SOURCES), "concepts"),
        fdr=get_param(params, "fdr", 0.05, "concepts", float),
        min_overlap=get_param(params, "min_overlap", 2, "concepts", int),
        organism=get_param(params, "organism", "hsapiens", "concepts", str),
        query_key=get_param(params, "query_key", None, "concepts", str, choices=("symbol", "ensembl")),
        cache_dir=get_param(params, "cache_dir", None, "concepts", str),
        offline_matrix=get_param(params, "offline_matrix", None, "concepts", str),
        gene_keys={"symbol_column": gk.get("symbol_column"), "ensembl_column": gk.get("ensembl_column")},
    )
    if not (0 < p.fdr <= 1):
        raise ValueError("stages.concepts.fdr must be in (0, 1]")
    if p.min_overlap < 1:
        raise ValueError("stages.concepts.min_overlap must be >= 1")
    return p


def params_dict(p: UniverseParams) -> Dict[str, Any]:
    d = asdict(p)
    d["sources"] = list(p.sources)
    return d


def _version(module: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("gprofiler-official" if module == "gprofiler" else module)
    except Exception:  # noqa: BLE001
        return None


def concept_size_mask(sizes, size_range: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of concepts with ``size_range[0] <= n_genes <= size_range[1]``."""
    s = np.asarray(sizes)
    return (s >= int(size_range[0])) & (s <= int(size_range[1]))
