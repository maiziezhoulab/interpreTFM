"""Calibrated cell–cell interface test, partner-marker tercile check and marker-excluded QC.

The statistic is the manuscript's strict crosstalk test on a per-cell *program score*
``P`` (the mean positive z-score of the projected features, each clipped at its 0.995
quantile; :func:`interp_pipeline.downstream.interaction.core.cell_program_scores`):

- ``P_adj`` residualises ``P`` on log total counts, log detected genes and log cell area
  within each exact label (:func:`adjust_score`);
- for every unordered pair of cell groups (optionally within each niche) with at least
  ``min_edges`` edges closer than the edge radius, the observed statistic is the mean over
  those edges of the two cells' scores (:class:`PairTest`);
- the null permutes the score within strata (square block × niche × exact label) and
  reports ``z = (observed - null mean) / null sd`` plus one-sided empirical p values;
- **calibration**: contact-free smooth random fields (Gaussian bumps at three length scales,
  mixed with white noise so that the neighbour correlation matches ``P_adj``) are pushed
  through the same null; ``z* = max(2, 95th percentile of |z|)`` pooled over fields and
  pairs, and a pair is *enriched* / *depleted* when ``z >= z*`` / ``z <= -z*``
  (:func:`calibrate`). The calibration controls the per-pair false-positive rate under
  smooth contact-free fields; it is not a false-discovery rate and its power is not
  evaluated;
- the **legacy** uncalibrated test (raw ``P``, strata = large tile × niche × label, 20
  permutations) is computed beside the calibrated one; the manuscript reports both shares
  of ``|z| > 2``;
- **effect %** = ``100 · (observed - null mean) / mean edge P`` for the same edges;
- the **niche-free** variant drops the niche from the pairs and the strata (used for the
  cross-model comparisons, the tercile check and the marker-excluded QC).

The permutation nulls draw ``rng.random(n)`` once per permutation and sort within strata, so
every result is a deterministic function of the seeds in :class:`InterfaceParams`; the
reference implementation's random-number consumption order is kept so that the manuscript's
tables are reproduced to floating-point precision.

:func:`partner_markers` / :func:`markers_by_group`, :func:`partner_marker_tercile`,
:func:`fired_on_genes` and :func:`marker_excluded_calls` implement the two QC checks of the
niche-free calls over several models (see ``docs/pipeline/stage_interfaces.md``).

All lengths here are in the dataset's native coordinate unit; the stage converts from µm with
``dataset.spatial.um_per_unit`` (:meth:`InterfaceParams.from_um`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from interp_pipeline.downstream.interaction.core import build_edges, cell_program_scores
from interpretfm.niches import radius_to_native

CALLS = ("enriched", "depleted", "not supported")
COVARIATE_COLUMNS = ("log_counts", "log_genes", "log_area")
DEFAULT_SIGNAL_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
DEFAULT_EXCLUDE = ("low_quality", "unresolved")


# --- labels -------------------------------------------------------------------------------------


def norm_label(x: Any) -> str:
    """Normalised label: stripped, lower case, spaces, slashes and hyphens replaced by ``_``."""
    return str(x).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_group_map(path: Union[str, Path]) -> Dict[str, str]:
    """``{label: group}`` from a CSV with columns ``label`` (or ``cell_type``) and ``group``.

    Keys and values are normalised with :func:`norm_label`.
    """
    df = pd.read_csv(path)
    key = "label" if "label" in df.columns else ("cell_type" if "cell_type" in df.columns else None)
    if key is None or "group" not in df.columns:
        raise ValueError(f"{path}: expected columns 'label' (or 'cell_type') and 'group', got {list(df.columns)}")
    return {norm_label(a): norm_label(b) for a, b in zip(df[key], df["group"])}


def group_labels(labels: Sequence[Any], group_map: Optional[Mapping[str, str]] = None,
                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """``(exact, group)`` object arrays: the (normalised) labels and their groups.

    Labels absent from ``group_map`` form their own group. ``group_map`` keys are matched
    after the same normalisation; with ``normalize=False`` labels are used verbatim (the map
    is still looked up both verbatim and normalised).
    """
    exact = np.asarray([norm_label(x) if normalize else str(x) for x in labels], dtype=object)
    if not group_map:
        return exact, exact.copy()
    gm = {str(k): str(v) for k, v in group_map.items()}
    gm.update({norm_label(k): (norm_label(v) if normalize else str(v)) for k, v in group_map.items()})
    group = np.asarray([gm.get(x, gm.get(norm_label(x), x)) for x in exact], dtype=object)
    return exact, group


# --- scores and covariates ----------------------------------------------------------------------


def program_score(X, clip_quantile: float = 0.995) -> np.ndarray:
    """Per-cell program score: mean over features of the positive part of the clipped z-scores
    (float64; the computation is the 0.9 library's ``cell_program_scores`` in float32)."""
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X = np.nan_to_num(X.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return cell_program_scores(X, float(clip_quantile))[1].astype(np.float64)


def covariates(counts, area: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Technical covariates from a cells × genes raw-count matrix: ``log_counts`` (log1p total
    counts), ``log_genes`` (log1p detected genes) and ``log_area`` (log1p of ``area``, when
    given), plus the untransformed ``total_counts`` and ``n_genes``."""
    tot = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    ng = np.asarray((counts > 0).sum(axis=1)).ravel().astype(np.float64)
    out = pd.DataFrame({"log_counts": np.log1p(tot), "log_genes": np.log1p(ng)})
    if area is not None:
        out["log_area"] = np.log1p(np.asarray(area, dtype=np.float64))
    out["total_counts"], out["n_genes"] = tot, ng
    return out


def cell_area(obs: pd.DataFrame, spec: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    """Cell area from ``obs`` as the configuration describes it: ``{column: <name>}`` or
    ``{product: [<a>, <b>]}`` (for example a bounding box width × height); ``None`` without a spec."""
    if not spec:
        return None
    if spec.get("column"):
        col = spec["column"]
        if col not in obs.columns:
            raise KeyError(f"covariates.area.column: '{col}' not in adata.obs")
        return obs[col].astype(float).to_numpy()
    if spec.get("product"):
        cols = list(spec["product"])
        if len(cols) != 2 or any(c not in obs.columns for c in cols):
            raise KeyError(f"covariates.area.product: expected two adata.obs columns, got {cols}")
        return obs[cols[0]].astype(float).to_numpy() * obs[cols[1]].astype(float).to_numpy()
    raise ValueError("covariates.area: expected {column: <obs column>} or {product: [<a>, <b>]}")


def adjust_score(P: np.ndarray, exact: np.ndarray, cov: pd.DataFrame,
                 columns: Sequence[str] = COVARIATE_COLUMNS) -> np.ndarray:
    """Residualise ``P`` on the covariate ``columns`` (with an intercept) within each exact label."""
    cols = [c for c in columns if c in cov.columns]
    C = cov[cols].to_numpy(dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    out = np.zeros_like(P)
    for c in np.unique(exact):
        m = exact == c
        A = np.column_stack([np.ones(int(m.sum())), C[m]])
        beta, *_ = np.linalg.lstsq(A, P[m], rcond=None)
        out[m] = P[m] - A @ beta
    return out


# --- the pair test ------------------------------------------------------------------------------


@dataclass
class PermResult:
    """Observed statistic, null mean/sd, z and one-sided empirical p values per pair."""

    obs: np.ndarray
    null_mean: np.ndarray
    null_sd: np.ndarray
    z: np.ndarray
    p_upper: np.ndarray
    p_lower: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"obs": self.obs, "null_mean": self.null_mean, "null_sd": self.null_sd, "z": self.z,
                             "p_upper": self.p_upper, "p_lower": self.p_lower})


class PairTest:
    """Edge-mean statistic for unordered group pairs (within niches when ``niche`` is given) and
    its within-stratum permutation null.

    ``O = mean over same-niche cross-group edges of (s_i + s_j) / 2``. Edges between two
    cells of the same group, or touching an excluded group, are dropped; a pair is kept when
    it has at least ``min_edges`` edges. The null permutes the cell score within strata
    (square block of side ``tile`` × niche × exact label); graph, coordinates and labels stay
    fixed.
    """

    def __init__(self, coords: np.ndarray, group: np.ndarray, exact: np.ndarray, niche: Optional[np.ndarray] = None,
                 edges: Optional[np.ndarray] = None, radius: Optional[float] = None,
                 exclude: Sequence[str] = DEFAULT_EXCLUDE, min_edges: int = 50):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.exact = np.asarray(exact, dtype=object)
        self.group = np.asarray(group, dtype=object)
        gi, gnames = pd.factorize(pd.Series(self.group))
        self.gi = gi.astype(np.int64)
        self.gnames = np.asarray(gnames, dtype=object)
        G = len(gnames)
        if niche is None:
            ni = np.zeros(len(self.coords), dtype=np.int64)
            nnames = np.array(["all"], dtype=object)
        else:
            ni, nnames = pd.factorize(pd.Series(np.asarray(niche).astype(str)))
            ni, nnames = ni.astype(np.int64), np.asarray(nnames, dtype=object)
        self.ni, self.nnames = ni, nnames
        self.use_niche = niche is not None
        if edges is None:
            if radius is None:
                raise ValueError("PairTest needs edges or a radius")
            edges = build_edges(self.coords, float(radius))
        self.all_edges = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
        e0, e1 = self.all_edges[:, 0], self.all_edges[:, 1]
        bad = np.isin(self.gnames, list(exclude))
        keep = (ni[e0] == ni[e1]) & (gi[e0] != gi[e1]) & ~bad[gi[e0]] & ~bad[gi[e1]]
        e0, e1 = e0[keep], e1[keep]
        lo, hi = np.minimum(gi[e0], gi[e1]), np.maximum(gi[e0], gi[e1])
        code = ni[e0].astype(np.int64) * G * G + lo * G + hi
        ucode, cidx = np.unique(code, return_inverse=True)
        cnt = np.bincount(cidx, minlength=len(ucode))
        ok = cnt >= int(min_edges)
        remap = np.full(len(ucode), -1)
        remap[ok] = np.arange(int(ok.sum()))
        cid = remap[cidx]
        sel = cid >= 0
        self.e0, self.e1, self.cid = e0[sel], e1[sel], cid[sel]
        ucode = ucode[ok]
        self.cnt = cnt[ok]
        self.pairs = pd.DataFrame({"niche": nnames[ucode // (G * G)], "group_a": self.gnames[(ucode % (G * G)) // G],
                                   "group_b": self.gnames[ucode % G], "n_edges": self.cnt})
        self.npairs = len(self.pairs)
        self._ex = pd.factorize(pd.Series(self.exact))[0].astype(np.int64)

    def pair_keys(self) -> List[str]:
        return [f"{a}|{b}" for a, b in zip(self.pairs["group_a"], self.pairs["group_b"])]

    def stat(self, v: np.ndarray) -> np.ndarray:
        """The edge-mean statistic of every pair for cell values ``v``."""
        return np.bincount(self.cid, weights=0.5 * (v[self.e0] + v[self.e1]), minlength=self.npairs) / self.cnt

    def strata(self, tile: float, offset: Sequence[float] = (0.0, 0.0), use_niche: Optional[bool] = None) -> np.ndarray:
        """Stratum id of every cell: square block of side ``tile`` × niche (when used) × exact label."""
        use_niche = self.use_niche if use_niche is None else use_niche
        t = np.floor((self.coords + np.asarray(offset, dtype=np.float64)[None, :]) / float(tile)).astype(np.int64)
        t -= t.min(axis=0)
        ex = self._ex
        key = (t[:, 0] * (t[:, 1].max() + 1) + t[:, 1]) * (self.ni.max() + 1 if use_niche else 1)
        if use_niche:
            key = key + self.ni
        key = key * (ex.max() + 1) + ex
        return np.unique(key, return_inverse=True)[1]

    @staticmethod
    def permutation(sid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One within-stratum permutation ``perm`` (``v[perm]`` is the shuffled vector)."""
        base = np.argsort(sid, kind="stable")
        po = np.lexsort((rng.random(len(sid)), sid))
        perm = np.empty_like(po)
        perm[base] = po
        return perm

    def permutations(self, sid: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
        """``n_perm`` within-stratum permutations as an ``(n_perm, n_cells)`` index array."""
        rng = np.random.default_rng(int(seed))
        return np.stack([self.permutation(sid, rng) for _ in range(int(n_perm))]) if n_perm else np.zeros((0, len(sid)), np.int64)

    def null(self, v: np.ndarray, perms: np.ndarray, n_perm: Optional[int] = None) -> PermResult:
        """Observed statistic and null from precomputed permutations (:meth:`permutations`)."""
        v = np.asarray(v, dtype=np.float64)
        obs = self.stat(v)
        n_perm = perms.shape[0] if n_perm is None else int(n_perm)
        null = np.zeros((self.npairs, n_perm))
        for b in range(n_perm):
            null[:, b] = self.stat(v[perms[b]])
        return self._result(obs, null, n_perm)

    def run(self, v: np.ndarray, *, tile: float, n_perm: int, seed: int, use_niche: Optional[bool] = None) -> PermResult:
        """The permutation test of ``v`` with strata of block ``tile`` and ``n_perm`` draws from ``seed``."""
        v = np.asarray(v, dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        obs = self.stat(v)
        null = np.zeros((self.npairs, int(n_perm)))
        sid = self.strata(tile, use_niche=use_niche)
        for b in range(int(n_perm)):
            null[:, b] = self.stat(v[self.permutation(sid, rng)])
        return self._result(obs, null, int(n_perm))

    @staticmethod
    def _result(obs: np.ndarray, null: np.ndarray, n_perm: int) -> PermResult:
        mu, sd = null.mean(1), null.std(1, ddof=1)
        z = np.where(sd > 1e-12, (obs - mu) / np.where(sd > 1e-12, sd, 1.0), np.nan)
        p_up = (1 + (null >= obs[:, None]).sum(1)) / (n_perm + 1)
        p_lo = (1 + (null <= obs[:, None]).sum(1)) / (n_perm + 1)
        return PermResult(obs=obs, null_mean=mu, null_sd=sd, z=z, p_upper=p_up, p_lower=p_lo)


def neighbour_corr(v: np.ndarray, edges: np.ndarray) -> float:
    """Correlation between a cell's value and the mean value of its neighbours in the radius graph."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    s = np.bincount(edges[:, 0], weights=v[edges[:, 1]], minlength=n) + np.bincount(edges[:, 1], weights=v[edges[:, 0]], minlength=n)
    d = np.bincount(edges[:, 0], minlength=n) + np.bincount(edges[:, 1], minlength=n)
    m = d > 0
    return float(np.corrcoef(v[m], s[m] / d[m])[0, 1])


def smooth_field(coords: np.ndarray, L: float, rng: np.random.Generator) -> np.ndarray:
    """A standardized contact-free random field: Gaussian bumps of width ``L`` at random centres
    (about two per ``L``² of the bounding box) with normal weights, summed over the 30 nearest centres."""
    coords = np.asarray(coords, dtype=np.float64)
    span = coords.max(0) - coords.min(0)
    nb = int(np.prod(span) / (L * L) * 2) + 10
    centers = coords.min(0) + rng.random((nb, 2)) * span
    w = rng.normal(size=nb)
    d, j = cKDTree(centers).query(coords, k=min(30, nb))
    f = (np.exp(-(d ** 2) / (2 * L * L)) * w[j]).sum(1)
    return (f - f.mean()) / f.std()


def matched_field(S: np.ndarray, N: np.ndarray, edges: np.ndarray, target_corr: float,
                  fractions: Sequence[float] = DEFAULT_SIGNAL_FRACTIONS) -> Tuple[float, float, np.ndarray]:
    """``sqrt(s)·S + sqrt(1-s)·N`` for the signal fraction ``s`` whose neighbour correlation is
    closest to ``target_corr``; returns ``(s, correlation, field)``."""
    best = None
    for s in fractions:
        f = np.sqrt(s) * S + np.sqrt(1 - s) * N
        r = neighbour_corr(f, edges)
        if best is None or abs(r - target_corr) < abs(best[1] - target_corr):
            best = (float(s), r, f)
    return best


def calls_from_z(z: np.ndarray, zstar: float) -> np.ndarray:
    return np.where(z >= zstar, "enriched", np.where(z <= -zstar, "depleted", "not supported")).astype(object)


# --- parameters ---------------------------------------------------------------------------------


@dataclass
class InterfaceParams:
    """The interface test's parameters in native coordinate units (see :meth:`from_um`)."""

    edge_radius: float = 120.0
    block: float = 120.0
    legacy_tile: float = 400.0
    length_scales: Sequence[float] = (60.0, 150.0, 400.0)
    n_perm: int = 100
    calib_perm: int = 50
    calib_reps: int = 2
    legacy_perm: int = 20
    min_edges: int = 50
    clip_quantile: float = 0.995
    exclude: Sequence[str] = DEFAULT_EXCLUDE
    signal_fractions: Sequence[float] = DEFAULT_SIGNAL_FRACTIONS
    seeds: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SEEDS))
    um_per_unit: Optional[float] = None

    @classmethod
    def from_um(cls, um_per_unit: float, edge_radius_um: float = 21.6, block_um: float = 21.6, legacy_tile_um: float = 72.0,
                length_scales_um: Sequence[float] = (10.8, 27.0, 72.0), **kw) -> "InterfaceParams":
        """Parameters given in µm, converted with ``um_per_unit`` (rounded to 1e-6 units)."""
        return cls(edge_radius=radius_to_native(edge_radius_um, um_per_unit), block=radius_to_native(block_um, um_per_unit),
                   legacy_tile=radius_to_native(legacy_tile_um, um_per_unit),
                   length_scales=[radius_to_native(v, um_per_unit) for v in length_scales_um], um_per_unit=float(um_per_unit), **kw)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InterfaceParams":
        """The inverse of :meth:`to_dict` (the ``um`` block is derived and ignored)."""
        return cls(**{k: v for k, v in d.items() if k != "um"})

    def seed(self, name: str) -> int:
        return int(self.seeds.get(name, DEFAULT_SEEDS[name]))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["length_scales"] = [float(v) for v in self.length_scales]
        d["exclude"], d["signal_fractions"] = list(self.exclude), [float(v) for v in self.signal_fractions]
        if self.um_per_unit:
            upu = float(self.um_per_unit)
            d["um"] = {"edge_radius": self.edge_radius * upu, "block": self.block * upu, "legacy_tile": self.legacy_tile * upu,
                       "length_scales": [float(v) * upu for v in self.length_scales]}
        return d


#: Seeds of the reference implementation. ``calibration`` draws the smooth fields; ``noise`` the i.i.d. field;
#: ``*_rep`` seeds are offset by the field replicate; the others seed one permutation test each.
DEFAULT_SEEDS: Dict[str, int] = {
    "calibration": 777, "noise": 5, "calib_legacy_rep": 900, "calib_niche_rep": 910, "calib_free_rep": 920, "calib_noise": 930,
    "legacy": 20240626, "legacy_adjusted": 20240627, "niche": 31, "free": 41, "legacy_free": 42,
}


# --- calibration and the full test --------------------------------------------------------------


def _pooled(zs: Iterable[np.ndarray]) -> Dict[str, float]:
    allz = np.concatenate(list(zs))
    a = np.abs(allz[np.isfinite(allz)])
    return {"fpr_abs_z_gt2": float(np.mean(a > 2)), "q95_abs_z": float(np.quantile(a, 0.95)),
            "q99_abs_z": float(np.quantile(a, 0.99)), "sd_z": float(np.std(allz[np.isfinite(allz)]))}


def calibrate(Tn: PairTest, Tf: PairTest, Padj: np.ndarray, P: np.ndarray, edges: np.ndarray, params: InterfaceParams,
              ) -> Dict[str, Any]:
    """Smooth-field and i.i.d.-noise calibration of the three nulls.

    For each length scale and replicate a smooth field matched to ``P_adj``'s neighbour
    correlation is run through the legacy null (``v1``), the calibrated niche null
    (``v2n``) and the niche-free null (``v2f``); ``zstar_niche`` / ``zstar_free`` are
    ``max(2, q95 |z|)`` of ``v2n`` / ``v2f``. Returns the ``calibration`` mapping of the
    summary (false-positive shares, quantiles, per-length-scale values, the fields).
    """
    r_real = neighbour_corr(Padj, edges)
    rng = np.random.default_rng(params.seed("calibration"))
    zc: Dict[str, List[np.ndarray]] = {"v1": [], "v2n": [], "v2f": []}
    fields = []
    for L in params.length_scales:
        for rep in range(int(params.calib_reps)):
            S = smooth_field(Tn.coords, float(L), rng)
            N = rng.normal(size=len(S))
            s, r, f = matched_field(S, N, edges, r_real, params.signal_fractions)
            fields.append({"L": float(L), "rep": rep, "signal_fraction": s, "neighbour_corr": r})
            zc["v1"].append(Tn.run(f, tile=params.legacy_tile, n_perm=params.legacy_perm, seed=params.seed("calib_legacy_rep") + rep).z)
            zc["v2n"].append(Tn.run(f, tile=params.block, n_perm=params.calib_perm, seed=params.seed("calib_niche_rep") + rep).z)
            zc["v2f"].append(Tf.run(f, tile=params.block, n_perm=params.calib_perm, seed=params.seed("calib_free_rep") + rep, use_niche=False).z)
    noise = np.random.default_rng(params.seed("noise")).normal(size=len(P))
    zc_noise = Tn.run(noise, tile=params.block, n_perm=params.calib_perm, seed=params.seed("calib_noise")).z
    cal: Dict[str, Any] = {"neighbour_corr_Padj": r_real, "neighbour_corr_P": neighbour_corr(P, edges), "fields": fields}
    for k, v in zc.items():
        cal[k] = _pooled(v)
        per_L = {}
        for L in params.length_scales:
            idx = [i for i, fl in enumerate(fields) if fl["L"] == float(L)]
            aa = np.abs(np.concatenate([v[i] for i in idx]))
            aa = aa[np.isfinite(aa)]
            per_L[str(int(L)) if float(L).is_integer() else f"{float(L):g}"] = {"fpr_abs_z_gt2": float(np.mean(aa > 2)), "q95_abs_z": float(np.quantile(aa, 0.95))}
        cal[k]["by_L"] = per_L
    an = np.abs(zc_noise[np.isfinite(zc_noise)])
    cal["iid_noise_v2n"] = {"fpr_abs_z_gt2": float(np.mean(an > 2)), "q95_abs_z": float(np.quantile(an, 0.95))}
    cal["zstar_niche"] = max(2.0, cal["v2n"]["q95_abs_z"])
    cal["zstar_free"] = max(2.0, cal["v2f"]["q95_abs_z"])
    return cal


def interface_test(X, coords: np.ndarray, exact: np.ndarray, group: np.ndarray, niche: np.ndarray, cov: pd.DataFrame,
                   params: InterfaceParams, model: str = "model", edges: Optional[np.ndarray] = None,
                   calibration_only: bool = False) -> Dict[str, Any]:
    """The complete interface test of one model's projected features ``X`` (cells × features).

    Returns ``{"niche": table, "free": table, "summary": mapping, "scores": table, "edges": array}``:
    the niche-stratified pair table (``z_v1``, ``effect_pct_v1``, ``z_v1_adjusted_score``,
    ``z_v2``, ``effect_pct_v2``, ``p_upper_v2``, ``p_lower_v2``, ``zstar_v2``, ``call_v2``,
    ``mean_edge_P``), the niche-free table (``z_v1_free``, ``z_v2_free``, ``effect_pct_v2_free``,
    ``zstar_v2_free``, ``call_v2_free``), the summary (calibration, counts, ``frac_absz_gt2_v1``
    / ``_v2``, effect sizes) and the per-cell scores (``P``, ``P_adj``, ``exact``, ``group``, ``niche``).
    """
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    P = program_score(X, params.clip_quantile)
    Padj = adjust_score(P, exact, cov)
    if edges is None:
        edges = build_edges(coords, params.edge_radius)
    Tn = PairTest(coords, group, exact, niche=niche, edges=edges, exclude=params.exclude, min_edges=params.min_edges)
    Tf = PairTest(coords, group, exact, niche=None, edges=edges, exclude=params.exclude, min_edges=params.min_edges)
    info: Dict[str, Any] = {"model": model, "n_cells": int(len(P)), "n_features": int(X.shape[1]), "n_edges": int(len(edges)),
                            "n_pairs_niche": Tn.npairs, "n_pairs_free": Tf.npairs, "block": params.block,
                            "covariates": [c for c in COVARIATE_COLUMNS if c in cov.columns], "params": params.to_dict()}
    cal = calibrate(Tn, Tf, Padj, P, edges, params)
    info["calibration"] = cal
    scores = pd.DataFrame({"P": P, "P_adj": Padj, "exact": exact, "group": group, "niche": np.asarray(niche).astype(str)})
    if calibration_only:
        return {"niche": None, "free": None, "summary": info, "scores": scores, "edges": edges}
    zstar_n, zstar_f = cal["zstar_niche"], cal["zstar_free"]

    v1 = Tn.run(P, tile=params.legacy_tile, n_perm=params.legacy_perm, seed=params.seed("legacy"))
    v1adj = Tn.run(Padj, tile=params.legacy_tile, n_perm=params.legacy_perm, seed=params.seed("legacy_adjusted"))
    v2 = Tn.run(Padj, tile=params.block, n_perm=params.n_perm, seed=params.seed("niche"))
    meanP = Tn.stat(P)
    out = Tn.pairs.copy()
    out["z_v1"] = v1.z
    out["effect_pct_v1"] = 100 * (v1.obs - v1.null_mean) / meanP
    out["z_v1_adjusted_score"] = v1adj.z
    out["z_v2"] = v2.z
    out["effect_pct_v2"] = 100 * (v2.obs - v2.null_mean) / meanP
    out["p_upper_v2"], out["p_lower_v2"] = v2.p_upper, v2.p_lower
    out["zstar_v2"] = zstar_n
    out["call_v2"] = calls_from_z(out["z_v2"].to_numpy(), zstar_n)
    out["mean_edge_P"] = meanP
    out.insert(0, "model", model)

    v2f = Tf.run(Padj, tile=params.block, n_perm=params.n_perm, seed=params.seed("free"), use_niche=False)
    v1f = Tf.run(P, tile=params.legacy_tile, n_perm=params.legacy_perm, seed=params.seed("legacy_free"), use_niche=False)
    meanPf = Tf.stat(P)
    outf = Tf.pairs.drop(columns="niche").copy()
    outf["z_v1_free"] = v1f.z
    outf["z_v2_free"] = v2f.z
    outf["effect_pct_v2_free"] = 100 * (v2f.obs - v2f.null_mean) / meanPf
    outf["zstar_v2_free"] = zstar_f
    outf["call_v2_free"] = calls_from_z(outf["z_v2_free"].to_numpy(), zstar_f)
    outf.insert(0, "model", model)

    info["n_supported_niche"] = {k: int(v) for k, v in out["call_v2"].value_counts().items()}
    info["n_supported_free"] = {k: int(v) for k, v in outf["call_v2_free"].value_counts().items()}
    info["frac_absz_gt2_v1"] = float(np.mean(np.abs(out["z_v1"]) > 2))
    info["frac_absz_gt2_v2"] = float(np.mean(np.abs(out["z_v2"]) > 2))
    info["effect_pct_v2"] = {"median_abs": float(np.nanmedian(np.abs(out["effect_pct_v2"]))),
                             "max_abs": float(np.nanmax(np.abs(out["effect_pct_v2"])))}
    return {"niche": out, "free": outf, "summary": info, "scores": scores, "edges": edges}


def summary_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """One flat row of the headline numbers (the layout of the manuscript's summary table)."""
    cal = summary["calibration"]
    sn, sf = summary.get("n_supported_niche", {}), summary.get("n_supported_free", {})
    return {"n_pairs_niche": summary["n_pairs_niche"], "frac_absz_gt2_v1": summary.get("frac_absz_gt2_v1"),
            "frac_absz_gt2_v2": summary.get("frac_absz_gt2_v2"), "n_enriched_v2": sn.get("enriched", 0),
            "n_depleted_v2": sn.get("depleted", 0), "zstar_niche": cal["zstar_niche"], "zstar_free": cal["zstar_free"],
            "median_abs_effect_pct_v2": (summary.get("effect_pct_v2") or {}).get("median_abs"),
            "max_abs_effect_pct_v2": (summary.get("effect_pct_v2") or {}).get("max_abs"),
            "neighbour_corr_P": cal["neighbour_corr_P"], "neighbour_corr_Padj": cal["neighbour_corr_Padj"],
            "smoothfield_fpr_v1": cal["v1"]["fpr_abs_z_gt2"], "smoothfield_fpr_v2": cal["v2n"]["fpr_abs_z_gt2"],
            "smoothfield_fpr_v2_free": cal["v2f"]["fpr_abs_z_gt2"], "noise_fpr_v2": cal["iid_noise_v2n"]["fpr_abs_z_gt2"],
            "n_enriched_free": sf.get("enriched", 0), "n_depleted_free": sf.get("depleted", 0), "n_pairs_free": summary["n_pairs_free"]}


# --- partner markers ------------------------------------------------------------------------------


def partner_markers(counts, genes: Sequence[str], labels: Sequence[Any], top_n: int = 10) -> pd.DataFrame:
    """Top ``top_n`` genes per label by a one-vs-rest Wilcoxon test on log1p CP10k of raw counts
    (scanpy ``rank_genes_groups``, ``tie_correct=False``, ranked by the test score).

    Returns one row per (label, rank): ``label, n_cells, rank, gene, wilcoxon_score, logfoldchange, pval_adj``.
    Labels with a single cell get no markers (the test needs at least two cells per group).
    """
    import anndata as ad
    import scanpy as sc

    X = sparse.csr_matrix(counts).astype(np.float32)
    lab = np.asarray([str(x) for x in labels], dtype=object)
    b = ad.AnnData(X=X, obs=pd.DataFrame({"label": lab}, index=[str(i) for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=pd.Index([str(g) for g in genes], name="gene")))
    if not b.var_names.is_unique:
        raise ValueError("partner_markers: gene names must be unique")
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    b.obs["label"] = b.obs["label"].astype("category")
    counts_per_label = b.obs["label"].value_counts()
    groups = [t for t in b.obs["label"].cat.categories if counts_per_label[t] >= 2]
    if not groups:
        return pd.DataFrame(columns=["label", "n_cells", "rank", "gene", "wilcoxon_score", "logfoldchange", "pval_adj"])
    sc.tl.rank_genes_groups(b, groupby="label", groups=groups, method="wilcoxon", reference="rest", use_raw=False, tie_correct=False,
                            n_genes=b.n_vars)
    res = b.uns["rank_genes_groups"]
    rows = []
    for t in groups:
        n = int((b.obs["label"] == t).sum())
        for r in range(min(int(top_n), b.n_vars)):
            rows.append({"label": t, "n_cells": n, "rank": r + 1, "gene": str(res["names"][t][r]),
                         "wilcoxon_score": float(res["scores"][t][r]), "logfoldchange": float(res["logfoldchanges"][t][r]),
                         "pval_adj": float(res["pvals_adj"][t][r])})
    return pd.DataFrame(rows)


def markers_by_group(markers: pd.DataFrame, group_of: Mapping[str, str]) -> Dict[str, set]:
    """Union of the marker genes of every label of a group (``group_of``: label → group;
    labels absent from it are their own group)."""
    out: Dict[str, set] = {}
    for lab, g in markers.groupby("label"):
        out.setdefault(str(group_of.get(str(lab), str(lab))), set()).update(str(x) for x in g["gene"])
    return out


def foreign_fractions(counts, genes: Sequence[str], markers: Mapping[str, Iterable[str]]) -> Dict[str, np.ndarray]:
    """Per group, the fraction of each cell's raw counts that falls on that group's markers."""
    Xc = sparse.csc_matrix(counts)
    tot = np.asarray(Xc.sum(1)).ravel().astype(np.float64)
    pos = {str(g): i for i, g in enumerate(genes)}
    out = {}
    for g, s in markers.items():
        idx = sorted(pos[x] for x in s if x in pos)
        missing = [x for x in s if x not in pos]
        if missing:
            raise KeyError(f"markers of {g!r} not in the panel: {missing}")
        out[g] = np.asarray(Xc[:, idx].sum(1)).ravel() / np.maximum(tot, 1)
    return out


# --- tercile check --------------------------------------------------------------------------------


class PairSubtest:
    """One pair's cells and cross-group edges with the edges split into terciles of the partner-marker
    fraction; the null permutes within block × exact label of the pair's cells (several score
    columns at once)."""

    def __init__(self, a: str, b: str, group: np.ndarray, exact: np.ndarray, coords: np.ndarray, edges: np.ndarray,
                 foreign: Mapping[str, np.ndarray], block: float, seed: int, tercile_seed: int = 0):
        cells = np.where((group == a) | (group == b))[0]
        self.cells = cells
        loc = np.full(len(group), -1)
        loc[cells] = np.arange(len(cells))
        e0, e1 = edges[:, 0], edges[:, 1]
        keep = (loc[e0] >= 0) & (loc[e1] >= 0) & (group[e0] != group[e1])
        self.e0, self.e1 = loc[e0[keep]], loc[e1[keep]]
        self.n = int(len(self.e0))
        gc = group[cells]
        ff = np.where(gc == a, foreign[b][cells], foreign[a][cells])
        ev = 0.5 * (ff[self.e0] + ff[self.e1])
        rng = np.random.default_rng(int(tercile_seed))
        order = np.lexsort((rng.random(self.n), ev))
        terc = np.empty(self.n, dtype=np.int64)
        terc[order] = np.minimum((np.arange(self.n) * 3) // max(self.n, 1), 2)
        self.tidx = [np.where(terc == k)[0] for k in range(3)]
        self.terc_mean = [float(ev[terc == k].mean()) if (terc == k).any() else float("nan") for k in range(3)]
        self.ff_mean = float(ev.mean()) if self.n else float("nan")
        t = np.floor(coords[cells] / float(block)).astype(np.int64)
        t -= t.min(0)
        ex = pd.factorize(pd.Series(exact[cells]))[0].astype(np.int64)
        key = (t[:, 0] * (t[:, 1].max() + 1) + t[:, 1]) * (ex.max() + 1) + ex
        self.sid = np.unique(key, return_inverse=True)[1]
        self.seed = int(seed)

    def stats(self, V: np.ndarray) -> np.ndarray:
        """``(4, n_columns)``: all edges, then terciles 1-3."""
        S = 0.5 * (V[self.e0] + V[self.e1])
        out = np.empty((4, V.shape[1]))
        out[0] = S.mean(0)
        for k in range(3):
            out[k + 1] = S[self.tidx[k]].mean(0) if len(self.tidx[k]) else np.nan
        return out

    def run(self, Vfull: np.ndarray, n_perm: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(obs, null mean, null sd, z)`` of shape ``(4, n_columns)`` for the full-length columns ``Vfull``."""
        V = np.asarray(Vfull, dtype=np.float64)[self.cells]
        rng = np.random.default_rng(self.seed)
        obs = self.stats(V)
        null = np.zeros((int(n_perm),) + obs.shape)
        base = np.argsort(self.sid, kind="stable")
        for k in range(int(n_perm)):
            po = np.lexsort((rng.random(len(self.sid)), self.sid))
            W = np.empty_like(V)
            W[base] = V[po]
            null[k] = self.stats(W)
        mu, sd = null.mean(0), null.std(0, ddof=1)
        return obs, mu, sd, (obs - mu) / np.where(sd > 1e-12, sd, np.nan)


def block_bootstrap(rows: pd.DataFrame, col: str = "g", B: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Median of ``col`` over the rows with its 95 % interval from ``B`` resamples of whole pairs."""
    rows = rows[np.isfinite(rows[col].to_numpy(dtype=float))]
    rng = np.random.default_rng(int(seed))
    pl = sorted(rows["pair"].unique())
    by = {p: rows.loc[rows["pair"] == p, col].to_numpy(dtype=float) for p in pl}
    med = float(np.median(rows[col])) if len(rows) else float("nan")
    if not pl:
        return {"median_g": med, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_rows": 0, "n_pairs": 0}
    bs = np.empty(int(B))
    for b in range(int(B)):
        pick = rng.integers(0, len(pl), len(pl))
        bs[b] = np.median(np.concatenate([by[pl[k]] for k in pick]))
    return {"median_g": med, "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
            "n_rows": int(len(rows)), "n_pairs": len(pl)}


def tercile_verdict(ci_lo: float, ci_hi: float) -> str:
    if ci_lo <= 0 <= ci_hi:
        return "CI includes 0"
    return "CI excludes 0 (negative)" if ci_hi < 0 else "CI excludes 0 (positive)"


def called_pairs(free_tables: Mapping[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """The niche-free rows of every model with a ``pair`` key, and the sorted pairs called
    (enriched or depleted) in at least one model."""
    calls = pd.concat([d.assign(pair=d["group_a"].astype(str) + "|" + d["group_b"].astype(str), model=m)
                       [["pair", "model", "z_v2_free", "zstar_v2_free", "call_v2_free", "n_edges"]] for m, d in free_tables.items()],
                      ignore_index=True)
    called = calls[calls["call_v2_free"] != "not supported"]
    return calls, sorted(called["pair"].unique())


def partner_marker_tercile(free_tables: Mapping[str, pd.DataFrame], scores: Mapping[str, np.ndarray], coords: np.ndarray,
                           exact: np.ndarray, group: np.ndarray, counts, genes: Sequence[str], markers: Mapping[str, Iterable[str]],
                           params: InterfaceParams, raw_scores: Optional[Mapping[str, np.ndarray]] = None, n_perm: int = 100,
                           seed_base: int = 5000, tercile_seed: int = 0, boot: int = 2000, boot_seed: int = 0,
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partner-marker tercile check of the niche-free calls over several models.

    For every pair called in at least one model, the pair's cross-group edges are split into
    terciles of the partner-marker fraction (raw counts on the partner group's markers / total
    counts, averaged over the edge's two cells); ``g = (z_t3 - z_t1) / |z_all|`` per (pair,
    model) from a within-block × exact-label null (``n_perm`` draws, seed ``seed_base`` + pair
    index, the models permuted together); the median ``g`` with a block bootstrap over pairs.
    ``scores`` are the models' ``P_adj`` (``raw_scores`` their ``P``, for the effect size).
    Returns ``(rows, summary)``; the summary's first row is the primary set (all called pairs × models).
    """
    models = list(free_tables)
    calls, pairs = called_pairs(free_tables)
    called = calls[calls["call_v2_free"] != "not supported"]
    foreign = foreign_fractions(counts, genes, markers)
    edges = build_edges(np.asarray(coords, dtype=np.float64)[:, :2], params.edge_radius)
    A = np.column_stack([np.asarray(scores[m], dtype=np.float64) for m in models])
    U = None if raw_scores is None else np.column_stack([np.asarray(raw_scores[m], dtype=np.float64) for m in models])
    rows = []
    for i, pr in enumerate(pairs):
        a, b = pr.split("|")
        sub = PairSubtest(a, b, group, exact, np.asarray(coords, dtype=np.float64)[:, :2], edges, foreign, params.block, seed_base + i, tercile_seed)
        o, mu, _sd, z = sub.run(A, n_perm)
        eff = None if U is None else 100 * (o - mu) / sub.stats(U[sub.cells])
        own_a, own_b = set(markers.get(a, ())), set(markers.get(b, ()))
        for j, m in enumerate(models):
            this = called[(called["pair"] == pr) & (called["model"] == m)]
            allm = calls[(calls["pair"] == pr) & (calls["model"] == m)]
            rows.append({"pair": pr, "model": m, "call_this_model": this["call_v2_free"].iloc[0] if len(this) else "not supported",
                         "z_v2_free": float(allm["z_v2_free"].iloc[0]) if len(allm) else np.nan,
                         "z_all": z[0, j], "z_t1": z[1, j], "z_t2": z[2, j], "z_t3": z[3, j],
                         "effect_pct_all": np.nan if eff is None else eff[0, j], "effect_pct_t1": np.nan if eff is None else eff[1, j],
                         "effect_pct_t3": np.nan if eff is None else eff[3, j], "n_edges": sub.n, "foreign_frac_mean": sub.ff_mean,
                         "foreign_frac_t1": sub.terc_mean[0], "foreign_frac_t2": sub.terc_mean[1], "foreign_frac_t3": sub.terc_mean[2],
                         "n_markers_a": len(own_a), "n_markers_b": len(own_b), "n_markers_shared_a_b": len(own_a & own_b)})
    D = pd.DataFrame(rows)
    if not len(D):
        return D, pd.DataFrame()
    D["g"] = (D["z_t3"] - D["z_t1"]) / D["z_all"].abs()
    direction = called.groupby("pair")["call_v2_free"].agg(lambda x: "/".join(sorted(set(x))))
    D["pair_call_direction"] = D["pair"].map(direction)
    summ = []
    for name, sel in (("all called pairs x models", np.ones(len(D), dtype=bool)), ("pairs called enriched", D["pair_call_direction"] == "enriched"),
                      ("pairs called depleted", D["pair_call_direction"] == "depleted"), ("calling (pair, model) rows only", D["call_this_model"] != "not supported")):
        S = D[sel]
        if not len(S):
            continue
        r = block_bootstrap(S, "g", B=boot, seed=boot_seed)
        r.update({"set": name, "primary": name == "all called pairs x models", "bootstrap": f"block over pairs ({boot}, seed {boot_seed})",
                  "verdict": tercile_verdict(r["ci_lo"], r["ci_hi"]), "median_z_t1": float(S["z_t1"].median()),
                  "median_z_t3": float(S["z_t3"].median()), "median_z_all": float(S["z_all"].median())})
        summ.append(r)
    return D, pd.DataFrame(summ)


# --- fired-on genes and the marker-excluded QC --------------------------------------------------------


def fired_on_genes(store_root: Union[str, Path], layer: str, encoder, resolver, gene_keys: Sequence[str],
                   feature_idx: Optional[Sequence[int]] = None, max_shards: Optional[int] = None, threshold: float = 0.15,
                   rate: float = 0.2, mass: float = 0.8, batch_size: int = 8192) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Genes on which each feature fires, from the gene-token rows of the first ``max_shards`` shards.

    A feature fires on a row when its latent exceeds ``threshold``. Its fired-on set is the genes
    whose rows fire it in at least ``rate`` of their occurrences, plus the smallest set of top
    genes holding ``mass`` of its activation mass (sum of the latent over firing rows), ordered by
    decreasing mass. ``resolver`` maps stored token ids to gene keys (rows of special or
    unknown tokens are skipped). Returns ``(table, info)``: one row per feature (``feature``,
    ``fired_on`` as ``;``-joined gene keys, ``n_fired``, ``n_rate``, ``n_mass``, ``total_mass``,
    ``n_firing_rows``) and the rows/shards used.
    """
    import torch

    from interpretfm.stores import list_shards, load_shard, select_shards

    keys = [str(k) for k in gene_keys]
    key_to_row = {k: i for i, k in enumerate(keys)}
    nG = len(keys)
    feats = np.arange(encoder.n_latents) if feature_idx is None else np.asarray(feature_idx, dtype=np.int64)
    fsel = torch.as_tensor(feats)
    nF = len(feats)
    g_cnt = torch.zeros(nG, dtype=torch.float64)
    g_fire = torch.zeros(nG, nF, dtype=torch.float64)
    g_mass = torch.zeros(nG, nF, dtype=torch.float64)
    n_rows, used = 0, []
    for shard in select_shards(list_shards(store_root, layer), max_shards):
        X, idx = load_shard(shard)
        gi = resolver.resolve_rows(idx.get("token_ids", []), key_to_row)
        keep = np.flatnonzero(gi >= 0)
        for i0 in range(0, len(keep), int(batch_size)):
            rows = keep[i0:i0 + int(batch_size)]
            z = encoder.encode(X[torch.as_tensor(rows)])[:, fsel]
            g = torch.as_tensor(gi[rows])
            on = z > float(threshold)
            g_cnt.index_add_(0, g, torch.ones(len(rows), dtype=torch.float64))
            g_fire.index_add_(0, g, on.double())
            g_mass.index_add_(0, g, (z * on).double())
            n_rows += len(rows)
        used.append(shard.name)
    rate_m = (g_fire / g_cnt.clamp(min=1)[:, None]).numpy()
    mass_m = g_mass.numpy()
    fired, n_rate, n_mass = [], [], []
    for j in range(nF):
        s = set(np.flatnonzero(rate_m[:, j] >= float(rate)))
        n_rate.append(len(s))
        m = mass_m[:, j]
        sm: set = set()
        if m.sum() > 0:
            o = np.argsort(-m)
            cum = np.cumsum(m[o]) / m.sum()
            sm = set(o[: int(np.searchsorted(cum, float(mass)) + 1)])
        n_mass.append(len(sm))
        s |= sm
        fired.append(";".join(keys[i] for i in sorted(s, key=lambda i: -mass_m[i, j])))
    table = pd.DataFrame({"feature": feats, "fired_on": fired, "n_fired": [len(x.split(";")) if x else 0 for x in fired],
                          "n_rate": n_rate, "n_mass": n_mass, "total_mass": mass_m.sum(0), "n_firing_rows": g_fire.sum(0).numpy()})
    info = {"n_rows": int(n_rows), "shards_used": used, "n_features": int(nF), "threshold": float(threshold), "rate": float(rate),
            "mass": float(mass)}
    return table, info


def fired_on_sets(table: pd.DataFrame, column: str = "fired_on") -> List[set]:
    """The fired-on gene sets of a :func:`fired_on_genes` table (row order)."""
    return [set(str(x).split(";")) if isinstance(x, str) and x else set() for x in table[column]]


class FreeCalibrator:
    """Niche-free null and calibration with the permutations drawn once (they do not depend on
    the score), for re-testing many reduced scores (:func:`marker_excluded_calls`)."""

    def __init__(self, Tf: PairTest, edges: np.ndarray, params: InterfaceParams):
        self.Tf, self.edges, self.params = Tf, edges, params
        sid = Tf.strata(params.block, use_niche=False)
        self.perm_real = Tf.permutations(sid, params.n_perm, params.seed("free"))
        self.perm_cal = {rep: Tf.permutations(sid, params.calib_perm, params.seed("calib_free_rep") + rep) for rep in range(int(params.calib_reps))}
        rng = np.random.default_rng(params.seed("calibration"))
        self.candidates: List[Dict[str, Any]] = []
        for L in params.length_scales:
            for rep in range(int(params.calib_reps)):
                S = smooth_field(Tf.coords, float(L), rng)
                N = rng.normal(size=len(S))
                for s in params.signal_fractions:
                    f = np.sqrt(s) * S + np.sqrt(1 - s) * N
                    self.candidates.append({"L": float(L), "rep": rep, "s": float(s), "r": neighbour_corr(f, edges),
                                            "z": Tf.null(f, self.perm_cal[rep]).z})
        self.pair_index = {k: i for i, k in enumerate(Tf.pair_keys())}
        self._edges_of = {}
        for k, i in self.pair_index.items():
            msk = Tf.cid == i
            self._edges_of[k] = (Tf.e0[msk], Tf.e1[msk])

    def zstar(self, padj: np.ndarray) -> Dict[str, Any]:
        """``z*`` for a score: per (length scale, replicate) the candidate field whose neighbour
        correlation is closest to the score's; ``max(2, q95 |z|)`` pooled."""
        r_real = neighbour_corr(padj, self.edges)
        zs, chosen = [], []
        for L in self.params.length_scales:
            for rep in range(int(self.params.calib_reps)):
                best = None
                for c in self.candidates:
                    if c["L"] != float(L) or c["rep"] != rep:
                        continue
                    if best is None or abs(c["r"] - r_real) < abs(best["r"] - r_real):
                        best = c
                zs.append(best["z"])
                chosen.append(best["s"])
        a = np.abs(np.concatenate(zs))
        a = a[np.isfinite(a)]
        q95 = float(np.quantile(a, 0.95))
        return {"zstar": max(2.0, q95), "q95": q95, "neighbour_corr": r_real, "signal_fractions": chosen}

    def z_all(self, v: np.ndarray) -> PermResult:
        return self.Tf.null(v, self.perm_real)

    def z_pair(self, v: np.ndarray, key: str) -> Tuple[float, float, float]:
        """``(z, observed, null mean)`` of one pair."""
        e0, e1 = self._edges_of[key]
        v = np.asarray(v, dtype=np.float64)
        o = float((0.5 * (v[e0] + v[e1])).mean())
        null = np.array([0.5 * (v[p][e0] + v[p][e1]).mean() for p in self.perm_real])
        mu, sd = null.mean(), null.std(ddof=1)
        return ((o - mu) / sd if sd > 1e-12 else float("nan")), o, float(mu)


def marker_excluded_calls(free_tables: Mapping[str, pd.DataFrame], features: Mapping[str, Any], fired: Mapping[str, Sequence[set]],
                          coords: np.ndarray, exact: np.ndarray, group: np.ndarray, cov: pd.DataFrame, counts, genes: Sequence[str],
                          markers: Mapping[str, Iterable[str]], params: InterfaceParams, concept_names: Optional[Mapping[str, Sequence[str]]] = None,
                          n_perm_tercile: int = 100, seed_base: int = 5000, tercile_seed: int = 0, boot: int = 2000, boot_seed: int = 0,
                          min_models: int = 2, log=None) -> Dict[str, Any]:
    """Marker-excluded (spillover-robust) re-test of the niche-free calls over several models.

    For every pair called in at least one model and every model: drop the features whose
    fired-on genes include a marker of either group, recompute ``P`` and ``P_adj`` on the
    remaining features, re-run the niche-free test on the same permutations (seed ``free``) and
    recalibrate ``z*`` on the smooth fields for the reduced score; a pair is
    ``spillover_robust`` when ``|z| >= z*`` with the sign of its original call in at least
    ``min_models`` models. The partner-marker tercile gradient of the reduced and the full score
    is computed on the same permutations. ``features[model]`` is the cells × features matrix,
    ``fired[model]`` the fired-on gene set of each feature column.

    Returns ``{"pairs_by_model", "verdicts", "tercile_summary", "controls", "calibration"}`` (tables / mappings).
    """
    models = list(free_tables)
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    edges = build_edges(coords, params.edge_radius)
    Tf = PairTest(coords, group, exact, niche=None, edges=edges, exclude=params.exclude, min_edges=params.min_edges)
    cal = FreeCalibrator(Tf, edges, params)
    calls, pairs = called_pairs(free_tables)
    called = calls[calls["call_v2_free"] != "not supported"]
    foreign = foreign_fractions(counts, genes, markers)
    dense = {m: (features[m].toarray() if sparse.issparse(features[m]) else np.asarray(features[m])) for m in models}
    P_full = {m: program_score(dense[m], params.clip_quantile) for m in models}
    Padj_full = {m: adjust_score(P_full[m], exact, cov) for m in models}
    controls = {}
    for m in models:
        z = cal.z_all(Padj_full[m]).z
        ref = free_tables[m].assign(k=free_tables[m]["group_a"].astype(str) + "|" + free_tables[m]["group_b"].astype(str)).set_index("k")["z_v2_free"]
        zs = cal.zstar(Padj_full[m])
        controls[m] = {"max_abs_diff_z_v2_free": float(np.nanmax(np.abs(z - ref.reindex(Tf.pair_keys()).to_numpy()))),
                       "zstar": zs["zstar"], "zstar_table": float(free_tables[m]["zstar_v2_free"].iloc[0]) if len(free_tables[m]) else None,
                       "neighbour_corr_Padj": zs["neighbour_corr"]}
        if log is not None:
            log.info("marker-excluded control %s: %s", m, json.dumps(controls[m]))
    rows, cal_rows = [], []
    for i, pr in enumerate(pairs):
        a, b = pr.split("|")
        mk_ab = set(markers.get(a, ())) | set(markers.get(b, ()))
        ocalls = called[called["pair"] == pr]
        signs = set(np.where(ocalls["call_v2_free"] == "enriched", 1, -1))
        orig_sign = signs.pop() if len(signs) == 1 else 0
        padj_red: Dict[str, Optional[np.ndarray]] = {}
        for m in models:
            rem = np.array([len(s & mk_ab) > 0 for s in fired[m]], dtype=bool)
            kept = ~rem
            this = calls[(calls["pair"] == pr) & (calls["model"] == m)]
            rec: Dict[str, Any] = {"pair": pr, "model": m, "n_features": int(len(rem)), "n_removed": int(rem.sum()), "n_kept": int(kept.sum()),
                                   "removed_concepts": ";".join(list(np.asarray(concept_names[m], dtype=object)[rem])[:200]) if concept_names else "",
                                   "z_orig": float(this["z_v2_free"].iloc[0]) if len(this) else np.nan,
                                   "zstar_orig": float(this["zstar_v2_free"].iloc[0]) if len(this) else np.nan,
                                   "call_orig": this["call_v2_free"].iloc[0] if len(this) else "not tested",
                                   "orig_sign_of_pair_call": orig_sign, "n_markers_union": len(mk_ab)}
            if kept.sum() == 0:
                rec.update({"z_red": np.nan, "zstar_red": np.nan, "call_red": "no features left", "holds": False})
                padj_red[m] = None
            else:
                Pr = program_score(dense[m][:, kept], params.clip_quantile)
                Pa = adjust_score(Pr, exact, cov)
                padj_red[m] = Pa
                z, o, mu = cal.z_pair(Pa, pr)
                zs = cal.zstar(Pa)
                rec.update({"z_red": z, "zstar_red": zs["zstar"], "q95_red": zs["q95"], "neighbour_corr_Padj_red": zs["neighbour_corr"],
                            "effect_pct_red": 100 * (o - mu) / Tf.stat(Pr)[cal.pair_index[pr]] if np.isfinite(z) else np.nan,
                            "call_red": "enriched" if z >= zs["zstar"] else ("depleted" if z <= -zs["zstar"] else "not supported"),
                            "holds": bool(orig_sign != 0 and np.isfinite(z) and orig_sign * z >= zs["zstar"])})
                cal_rows.append({"pair": pr, "model": m, "neighbour_corr": zs["neighbour_corr"], "chosen_signal_fraction": zs["signal_fractions"],
                                 "q95": zs["q95"], "zstar": zs["zstar"]})
            rows.append(rec)
        sub = PairSubtest(a, b, group, exact, coords, edges, foreign, params.block, seed_base + i, tercile_seed)
        cols, names = [], []
        for m in models:
            if padj_red[m] is not None:
                cols.append(padj_red[m])
                names.append(("red", m))
            cols.append(Padj_full[m])
            names.append(("full", m))
        _o, _mu, _sd, Z = sub.run(np.column_stack(cols), n_perm_tercile)
        for j, (kind, m) in enumerate(names):
            r_ = next(r for r in rows if r["pair"] == pr and r["model"] == m)
            r_.update({f"z_all_{kind}": Z[0, j], f"z_t1_{kind}": Z[1, j], f"z_t2_{kind}": Z[2, j], f"z_t3_{kind}": Z[3, j],
                       f"g_{kind}": (Z[3, j] - Z[1, j]) / abs(Z[0, j]) if Z[0, j] != 0 else np.nan})
        for m in models:
            r_ = next(r for r in rows if r["pair"] == pr and r["model"] == m)
            r_.update({"n_edges_sub": sub.n, "foreign_frac_t1": sub.terc_mean[0], "foreign_frac_t2": sub.terc_mean[1], "foreign_frac_t3": sub.terc_mean[2]})
        if log is not None:
            rr = [r for r in rows if r["pair"] == pr]
            log.info("marker-excluded %s (sign %+d): %s", pr, orig_sign, "; ".join(
                f"{r['model']} removed {r['n_removed']}/{r['n_features']} z {r['z_orig']:+.2f}->{r['z_red']:+.2f} (z* {r.get('zstar_red', np.nan):.2f}) "
                f"{'holds' if r['holds'] else '-'}" for r in rr))
    Dm = pd.DataFrame(rows)
    if not len(Dm):
        return {"pairs_by_model": Dm, "verdicts": pd.DataFrame(), "tercile_summary": pd.DataFrame(), "controls": controls,
                "calibration": pd.DataFrame(cal_rows)}
    V = (Dm.groupby("pair").agg(orig_sign=("orig_sign_of_pair_call", "first"),
                                n_models_called_orig=("call_orig", lambda x: int((x != "not supported").sum())),
                                n_models_holding=("holds", "sum"),
                                removed_per_model=("n_removed", lambda x: "/".join(str(int(v)) for v in x)),
                                z_orig=("z_orig", lambda x: "/".join(f"{v:+.2f}" for v in x)),
                                z_red=("z_red", lambda x: "/".join(f"{v:+.2f}" for v in x)),
                                zstar_red=("zstar_red", lambda x: "/".join(f"{v:.2f}" for v in x))).reset_index())
    V["n_models_holding"] = V["n_models_holding"].astype(int)
    V["spillover_robust"] = V["n_models_holding"] >= int(min_models)
    TS = []
    for col, label in (("g_red", "reduced (marker-excluded) score"), ("g_full", "full score (same permutations)")):
        if col not in Dm.columns:
            continue
        for nm, s_ in (("all called pairs x models", Dm), ("pairs whose call was enriched", Dm[Dm["orig_sign_of_pair_call"] == 1]),
                       ("pairs whose call was depleted", Dm[Dm["orig_sign_of_pair_call"] == -1])):
            if len(s_) and np.isfinite(s_[col].to_numpy(dtype=float)).any():
                r = block_bootstrap(s_, col, B=boot, seed=boot_seed)
                r.update({"score": label, "set": nm, "verdict": tercile_verdict(r["ci_lo"], r["ci_hi"]),
                          "median_z_t1": float(s_[col.replace("g_", "z_t1_")].median()), "median_z_t3": float(s_[col.replace("g_", "z_t3_")].median()),
                          "median_z_all": float(s_[col.replace("g_", "z_all_")].median())})
                TS.append(r)
    return {"pairs_by_model": Dm, "verdicts": V, "tercile_summary": pd.DataFrame(TS), "controls": controls, "calibration": pd.DataFrame(cal_rows)}


__all__ = ["CALLS", "DEFAULT_SEEDS", "FreeCalibrator", "InterfaceParams", "PairSubtest", "PairTest", "PermResult", "adjust_score",
           "block_bootstrap", "calibrate", "called_pairs", "calls_from_z", "cell_area", "covariates", "fired_on_genes", "fired_on_sets",
           "foreign_fractions", "group_labels", "interface_test", "load_group_map", "marker_excluded_calls", "markers_by_group",
           "matched_field", "neighbour_corr", "norm_label", "partner_marker_tercile", "partner_markers", "program_score",
           "smooth_field", "summary_row", "tercile_verdict"]
