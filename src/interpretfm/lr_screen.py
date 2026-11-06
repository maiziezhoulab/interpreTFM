"""Model-free ligand–receptor co-localization screen (the manuscript's panel-wide screen).

The screen uses the expression matrix (raw counts), the cell coordinates and the cell labels
only; no model output enters it, so it is the model-free reference against which the
feature-based spatial analyses are read.

For every ligand–receptor pair of a database whose genes are all on the panel (complexes
resolved to their subunits, pairs de-duplicated by gene sets):

- a cell is *ligand-positive* when every ligand gene has at least ``thr`` transcripts
  (receptor-positive likewise); the primary threshold is 1 transcript, 3 is the sensitivity
  threshold; a pair is tested when both sides have at least ``min_positive`` cells;
- the statistic is the number of (ligand-positive, receptor-positive) pairs of *distinct* cells
  within the radius (ordered: cell *i* ligand-positive and cell *j* receptor-positive);
- the null permutes positivity within cell type, ligand and receptor independently
  (``n_perm`` permutations drawn once and shared by all pairs);
- fold = (observed + 1) / (null mean + 1), z = (observed − null mean) / null sd, two-sided
  normal p, Benjamini–Hochberg q over the tested pairs of a threshold;
- hit: q < ``q_max`` and fold ≥ ``fold_min``; robust: hit at both thresholds; depleted:
  q < ``q_max`` and fold ≤ 1 / ``fold_min`` at the primary threshold.

The random-number consumption follows the reference implementation (one generator seeded with
``seed``; ``2 n_perm`` within-type permutations, the first ``n_perm`` for the ligand side),
so that the manuscript's tables are reproduced to the digit.

Niche placement of a hit (:func:`niche_placement`): the close pairs are attributed to the
ligand-positive cell; the share of close pairs whose ligand cell lies in a niche, divided by the
niche's share of all cells, says where in the tissue the hit lives (1 = as expected from the
niche's size). Cells without a niche label belong to no niche but stay in the denominators.
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("interpretfm")

#: The ligand–receptor database shipped with the package (see ``data/cellchatdb/NOTICE.md``).
DATA_DIR = Path(__file__).resolve().parent / "data" / "cellchatdb"
DEFAULT_INTERACTIONS = DATA_DIR / "cellchatdb_human_interaction.csv.gz"
DEFAULT_COMPLEXES = DATA_DIR / "cellchatdb_human_complex.csv.gz"

INTERACTION_COLUMNS = ("interaction_name", "ligand", "receptor")


# --- database -----------------------------------------------------------------------------------


def _read_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh)
    return pd.read_csv(path)


def load_cellchatdb(interactions: Optional[Union[str, Path]] = None, complexes: Optional[Union[str, Path]] = None,
                    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """The interaction table and the complex → subunit mapping.

    ``interactions`` is a CSV with at least ``interaction_name``, ``ligand`` and ``receptor``
    (``pathway_name`` and ``annotation`` are carried along when present); ``complexes`` a CSV
    with ``complex_name`` and ``subunit_1`` … ``subunit_n`` columns. Both default to the packaged
    CellChatDB human export. ``complexes=False`` uses no complex table (ligand / receptor names
    are genes, or ``;``-separated gene lists).
    """
    db = _read_csv(interactions or DEFAULT_INTERACTIONS)
    missing = [c for c in INTERACTION_COLUMNS if c not in db.columns]
    if missing:
        raise ValueError(f"ligand–receptor database {interactions or DEFAULT_INTERACTIONS}: missing column(s) {missing}")
    for c in ("pathway_name", "annotation"):
        if c not in db.columns:
            db[c] = ""
    db = db.fillna({"pathway_name": "", "annotation": ""})
    if complexes is False:
        return db, {}
    cx = _read_csv(complexes or DEFAULT_COMPLEXES).fillna("")
    if "complex_name" not in cx.columns:
        raise ValueError(f"complex table {complexes or DEFAULT_COMPLEXES}: expected a 'complex_name' column")
    sub_cols = [c for c in cx.columns if str(c).startswith("subunit_")]
    cmap = {str(r["complex_name"]): [str(r[c]) for c in sub_cols if str(r[c])] for _, r in cx.iterrows()}
    return db, cmap


def resolve_genes(name: str, complexes: Dict[str, List[str]]) -> List[str]:
    """The genes of a ligand or receptor name: its complex subunits, a ``;``-separated list, or the name itself."""
    if name in complexes:
        return list(complexes[name])
    if ";" in name:
        return [g for g in name.split(";") if g]
    return [name]


def panel_pairs(db: pd.DataFrame, complexes: Dict[str, List[str]], genes: Iterable[str]) -> pd.DataFrame:
    """The database pairs whose ligand and receptor genes are all on the panel, one row per distinct
    (ligand gene set, receptor gene set) in database order; ``ligand_genes`` / ``receptor_genes`` are
    ``;``-joined."""
    on_panel = set(str(g) for g in genes)
    rows, seen = [], set()
    for r in db.itertuples(index=False):
        lg, rg = resolve_genes(str(r.ligand), complexes), resolve_genes(str(r.receptor), complexes)
        if all(g in on_panel for g in lg) and all(g in on_panel for g in rg):
            key = (tuple(sorted(lg)), tuple(sorted(rg)))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"interaction_name": str(r.interaction_name), "pathway_name": str(r.pathway_name), "annotation": str(r.annotation),
                         "ligand": str(r.ligand), "receptor": str(r.receptor), "ligand_genes": ";".join(lg), "receptor_genes": ";".join(rg)})
    cols = ["interaction_name", "pathway_name", "annotation", "ligand", "receptor", "ligand_genes", "receptor_genes"]
    return pd.DataFrame(rows, columns=cols)


# --- neighbour graph and permutations -----------------------------------------------------------


def neighbour_graph(coords: np.ndarray, radius: float) -> sp.csr_matrix:
    """Binary symmetric adjacency (float32 CSR, no diagonal) of cell pairs within ``radius`` (native units)."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    n = coords.shape[0]
    pr = cKDTree(coords).query_pairs(float(radius), output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pr), np.float32), (pr[:, 0], pr[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


def within_type_permutations(labels: Sequence, n_perm: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """``2 n_perm`` permutations of the cell indices that keep every cell within its label (the
    reference's draw order: one generator, labels in sorted order); the first ``n_perm`` are the
    ligand side, the rest the receptor side."""
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(int(seed))
    type_idx = [np.flatnonzero(labels == t) for t in sorted(set(labels.tolist()))]
    tables = np.empty((2 * n_perm, n), dtype=np.int64)
    for p in range(2 * n_perm):
        perm = np.arange(n)
        for idx in type_idx:
            perm[idx] = idx[rng.permutation(len(idx))]
        tables[p] = perm
    return tables[:n_perm], tables[n_perm:]


def positive_cells(X: sp.csr_matrix, columns: Sequence[int], thr: float) -> np.ndarray:
    """Cells with at least ``thr`` transcripts of every gene in ``columns``."""
    n = X.shape[0]
    v = np.ones(n, dtype=bool)
    for j in columns:
        v &= np.asarray(X[:, int(j)].todense()).ravel() >= thr
    return v


def pair_statistic(A: sp.csr_matrix, s: np.ndarray, t: np.ndarray, P_L: np.ndarray, P_R: np.ndarray,
                   batch: int = 25) -> Tuple[float, np.ndarray]:
    """Observed count of (ligand-positive, receptor-positive) close pairs and its within-type null draws."""
    n_perm = P_L.shape[0]
    obs = float(s @ (A @ t.astype(np.float32)))
    nulls = np.empty(n_perm)
    for b in range(0, n_perm, batch):
        ps = list(range(b, min(b + batch, n_perm)))
        T = np.stack([t[P_R[p]] for p in ps], 1).astype(np.float32)
        AT = A @ T
        for j, p in enumerate(ps):
            nulls[p] = float(s[P_L[p]] @ AT[:, j])
    return obs, nulls


# --- the screen ---------------------------------------------------------------------------------


@dataclass
class ScreenParams:
    """Parameters of the screen in native coordinate units (``radius``)."""

    radius: float
    thresholds: List[int] = field(default_factory=lambda: [1, 3])
    n_perm: int = 100
    seed: int = 0
    min_positive: int = 100
    q_max: float = 0.05
    fold_min: float = 1.2

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if not self.thresholds:
            raise ValueError("thresholds must list at least one positivity threshold")
        if self.n_perm < 2:
            raise ValueError("n_perm must be >= 2")
        if self.fold_min <= 1.0:
            raise ValueError("fold_min must be > 1")

    @property
    def primary(self) -> int:
        return int(self.thresholds[0])

    def to_dict(self) -> Dict[str, Any]:
        return {"radius": float(self.radius), "thresholds": [int(t) for t in self.thresholds], "n_perm": int(self.n_perm),
                "seed": int(self.seed), "min_positive": int(self.min_positive), "q_max": float(self.q_max), "fold_min": float(self.fold_min)}


def gene_index(genes: Sequence[str]) -> Dict[str, int]:
    """Gene symbol → column; a repeated symbol keeps its last column (as the reference did) and is logged."""
    gi: Dict[str, int] = {}
    dup = set()
    for i, g in enumerate(genes):
        g = str(g)
        if g in gi:
            dup.add(g)
        gi[g] = i
    if dup:
        log.warning("lr_screen: %d gene symbol(s) occur more than once on the panel (last column used): %s", len(dup), sorted(dup)[:10])
    return gi


def screen_pairs(X, genes: Sequence[str], labels: Sequence, coords: np.ndarray, pairs: pd.DataFrame, params: ScreenParams,
                 A: Optional[sp.csr_matrix] = None, progress: Optional[Any] = None) -> pd.DataFrame:
    """Run the screen; returns one row per pair with, per threshold ``thr``: ``n_lig_pos_thr<thr>``,
    ``n_rec_pos_thr<thr>``, ``tested_thr<thr>``, ``obs``, ``null_mean``, ``null_sd``, ``fold``, ``z``, ``p``,
    ``p_emp_upper``, ``q`` and ``hit``; plus ``robust_hit`` and ``depleted_thr<primary>``. Rows are sorted by
    ``z`` at the primary threshold, descending (untested pairs last)."""
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    X = sp.csr_matrix(X).astype(np.float32)
    n = X.shape[0]
    labels = np.asarray(labels)
    if len(labels) != n or len(coords) != n:
        raise ValueError(f"lr_screen: {n} cells in the matrix, {len(labels)} labels, {len(coords)} coordinates")
    gi = gene_index(genes)
    if A is None:
        A = neighbour_graph(coords, params.radius)
    P_L, P_R = within_type_permutations(labels, params.n_perm, params.seed)
    thresholds = [int(t) for t in params.thresholds]
    rows = []
    for k, pr in enumerate(pairs.itertuples(index=False)):
        lg, rg = [gi[g] for g in pr.ligand_genes.split(";")], [gi[g] for g in pr.receptor_genes.split(";")]
        rec: Dict[str, Any] = pr._asdict()
        for thr in thresholds:
            s, t = positive_cells(X, lg, thr), positive_cells(X, rg, thr)
            rec[f"n_lig_pos_thr{thr}"], rec[f"n_rec_pos_thr{thr}"] = int(s.sum()), int(t.sum())
            if s.sum() < params.min_positive or t.sum() < params.min_positive:
                rec[f"tested_thr{thr}"] = False
                continue
            o, nl = pair_statistic(A, s, t, P_L, P_R)
            mu, sd = nl.mean(), nl.std(ddof=1)
            z = (o - mu) / sd if sd > 0 else np.nan
            rec.update({f"tested_thr{thr}": True, f"obs_thr{thr}": o, f"null_mean_thr{thr}": mu, f"null_sd_thr{thr}": sd,
                        f"fold_thr{thr}": (o + 1) / (mu + 1), f"z_thr{thr}": z,
                        f"p_thr{thr}": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan,
                        f"p_emp_upper_thr{thr}": (1 + (nl >= o).sum()) / (params.n_perm + 1)})
        rows.append(rec)
        if progress is not None and k % 50 == 0:
            progress(k, len(pairs), rec)
    res = pd.DataFrame(rows)
    if res.empty:
        res = pd.DataFrame(columns=list(pairs.columns))
    for thr in thresholds:
        for c in (f"tested_thr{thr}", f"n_lig_pos_thr{thr}", f"n_rec_pos_thr{thr}", f"obs_thr{thr}", f"null_mean_thr{thr}", f"null_sd_thr{thr}",
                  f"fold_thr{thr}", f"z_thr{thr}", f"p_thr{thr}", f"p_emp_upper_thr{thr}"):
            if c not in res.columns:
                res[c] = False if c.startswith("tested") else np.nan
        tested = res[f"tested_thr{thr}"].fillna(False).astype(bool) & res[f"p_thr{thr}"].notna()
        q = np.full(len(res), np.nan)
        if tested.sum():
            q[tested.to_numpy()] = multipletests(res.loc[tested, f"p_thr{thr}"], method="fdr_bh")[1]
        res[f"q_thr{thr}"] = q
        res[f"hit_thr{thr}"] = (res[f"q_thr{thr}"] < params.q_max) & (res[f"fold_thr{thr}"] >= params.fold_min)
        res[f"tested_thr{thr}"] = res[f"tested_thr{thr}"].fillna(False).astype(bool)
    hit_cols = [f"hit_thr{thr}" for thr in thresholds]
    res["robust_hit"] = np.logical_and.reduce([res[c].fillna(False).astype(bool).to_numpy() for c in hit_cols]) if len(res) else []
    p0 = params.primary
    res[f"depleted_thr{p0}"] = (res[f"q_thr{p0}"] < params.q_max) & (res[f"fold_thr{p0}"] <= 1.0 / params.fold_min)
    res = res.sort_values(f"z_thr{p0}", ascending=False, na_position="last").reset_index(drop=True)
    return res


def screen_summary(res: pd.DataFrame, params: ScreenParams) -> Dict[str, Any]:
    """Counts per threshold (tested, hits), robust hits and depleted pairs."""
    thresholds = [int(t) for t in params.thresholds]
    p0 = params.primary
    out: Dict[str, Any] = {"n_pairs": int(len(res)), "thresholds": {}}
    for thr in thresholds:
        tested = res[f"tested_thr{thr}"].fillna(False).astype(bool)
        out["thresholds"][str(thr)] = {"n_tested": int(tested.sum()), "n_hits": int(res[f"hit_thr{thr}"].fillna(False).astype(bool).sum()),
                                       "hits": [str(x) for x in res.loc[res[f"hit_thr{thr}"].fillna(False).astype(bool), "interaction_name"]]}
    out["n_robust"] = int(res["robust_hit"].fillna(False).astype(bool).sum()) if len(res) else 0
    out["robust"] = [str(x) for x in res.loc[res["robust_hit"].fillna(False).astype(bool), "interaction_name"]] if len(res) else []
    out["n_depleted"] = int(res[f"depleted_thr{p0}"].fillna(False).astype(bool).sum()) if len(res) else 0
    out["depleted"] = [str(x) for x in res.loc[res[f"depleted_thr{p0}"].fillna(False).astype(bool), "interaction_name"]] if len(res) else []
    return out


# --- niche placement ----------------------------------------------------------------------------


def close_pair_weights(X, gi: Dict[str, int], A: sp.csr_matrix, ligand_genes: Sequence[str], receptor_genes: Sequence[str],
                       thr: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell: the number of close pairs attributed to it as the ligand-positive cell (``w``),
    with the ligand and receptor indicator vectors."""
    s = positive_cells(X, [gi[g] for g in ligand_genes], thr)
    t = positive_cells(X, [gi[g] for g in receptor_genes], thr)
    deg = A @ t.astype(np.float32)
    return s * deg, s, t


def niche_placement(res: pd.DataFrame, X, genes: Sequence[str], A: sp.csr_matrix, niche: Sequence, which: Sequence[int],
                    thr: float = 1.0) -> pd.DataFrame:
    """Share of close pairs (ligand side) per niche over the niche's share of cells, for the rows
    ``which`` of ``res``. ``niche`` holds one label per cell (``None``/NaN = no niche). Long table:
    ``interaction_name``, ``niche``, ``share_pairs``, ``share_cells``, ``ratio``."""
    X = sp.csr_matrix(X).astype(np.float32)
    gi = gene_index(genes)
    lab = pd.Series(list(niche), dtype=object)
    ids = sorted(set(lab.dropna().astype(str)), key=lambda v: (len(v), v))
    masks = {nn: (lab.astype(str).to_numpy() == nn) & lab.notna().to_numpy() for nn in ids}
    rows = []
    for i in which:
        w, _s, _t = close_pair_weights(X, gi, A, res.at[i, "ligand_genes"].split(";"), res.at[i, "receptor_genes"].split(";"), thr)
        tot = max(float(w.sum()), 1.0)
        for nn in ids:
            inn = masks[nn]
            share_pairs, share_cells = float(w[inn].sum() / tot), float(inn.mean())
            rows.append({"interaction_name": res.at[i, "interaction_name"], "niche": nn, "share_pairs": share_pairs, "share_cells": share_cells,
                         "ratio": share_pairs / share_cells if share_cells > 0 else np.nan, "n_close_pairs": float(w.sum())})
    return pd.DataFrame(rows, columns=["interaction_name", "niche", "share_pairs", "share_cells", "ratio", "n_close_pairs"])


def celltype_pairs(X, genes: Sequence[str], A: sp.csr_matrix, labels: Sequence, ligand_genes: Sequence[str], receptor_genes: Sequence[str],
                   thr: float = 1.0, top: int = 3) -> str:
    """The cell-type pairs (ligand cell → receptor cell) carrying a hit's close pairs, the top ``top`` by share."""
    X = sp.csr_matrix(X).astype(np.float32)
    gi = gene_index(genes)
    labels = np.asarray(labels)
    _w, s, t = close_pair_weights(X, gi, A, ligand_genes, receptor_genes, thr)
    S = (sp.diags(s.astype(np.float32)) @ A @ sp.diags(t.astype(np.float32))).tocoo()
    if S.nnz == 0:
        return ""
    counts = pd.Series([f"{labels[a]}->{labels[b]}" for a, b in zip(S.row, S.col)]).value_counts(normalize=True)
    return "; ".join(f"{k} {v:.2f}" for k, v in counts.head(top).items())


__all__ = ["DATA_DIR", "DEFAULT_COMPLEXES", "DEFAULT_INTERACTIONS", "ScreenParams", "celltype_pairs", "close_pair_weights", "gene_index",
           "load_cellchatdb", "neighbour_graph", "niche_placement", "pair_statistic", "panel_pairs", "positive_cells", "resolve_genes",
           "screen_pairs", "screen_summary", "within_type_permutations"]
