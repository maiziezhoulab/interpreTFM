"""Pipeline configuration: a YAML file with typed validation.

The schema is a set of dataclasses with explicit checks (no third-party validation library, so
it works on Python 3.8–3.11 with PyYAML only). Sections:

``run``
    ``name``, ``out_dir`` (the run directory, created by the runner), ``seed``.
``dataset``
    ``h5ad`` (path), ``counts_layer`` (``null`` = ``X`` holds raw counts), ``gene_id``
    (``column``: ``index`` or a ``var`` column; ``kind``: ``symbol`` or ``ensembl``),
    ``labels`` (optional: ``column`` in ``obs``), ``spatial`` (optional: ``obsm`` key and
    ``um_per_unit``; omitted = scRNA-seq mode, spatial stages are skipped), ``qc``.
``model``
    ``name``, ``checkpoint``, ``env`` (where extraction runs; see :mod:`interpretfm.envs`),
    ``layers``, ``cell_pooling``, ``special_tokens``, ``representative_layer``, ``vocab``
    (optional JSON gene vocabulary, used by ``ingest`` to report gene coverage).
``stages``
    One mapping per stage name (``extract``, ``tis``, ``sae``, ``concepts``, ``cells``,
    ``layers``, ``spatial``, ``report``) holding that stage's parameters. The runner hashes the
    section of a stage together with ``dataset``, ``model`` and ``run.seed`` to decide whether a
    cached result is still valid; the stage implementation reads it through
    :meth:`Config.stage_params`.
``output``
    ``html`` and ``json`` report switches and ``keep_store`` (whether the activation store is
    kept after the run).

Relative paths are resolved against the directory of the configuration file.

Example::

    run: {name: toy, out_dir: runs/toy, seed: 0}
    dataset:
      h5ad: data/toy.h5ad
      gene_id: {column: index, kind: symbol}
      labels: {column: cell_type}
      spatial: {obsm: spatial, um_per_unit: 0.5}
    model:
      name: scgpt
      checkpoint: /path/to/scgpt
      env: {conda: interpretfm}
      layers: [layer_1.norm2]
    stages:
      tis: {trials: 800}
    output: {html: true, json: true}
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

#: Stage names in pipeline order. The runner numbers stage directories in this order.
STAGE_ORDER = ["ingest", "extract", "tis", "sae", "concepts", "concept_eval", "gene_readout", "cells", "layers",
               "spatial", "report"]

#: Stages that take parameters from the ``stages`` section (``ingest`` has none beyond ``dataset.qc``).
PARAM_STAGES = [s for s in STAGE_ORDER if s != "ingest"]

_TOP_LEVEL = {"run", "dataset", "model", "stages", "output"}


class ConfigError(ValueError):
    """A configuration file is invalid. The message lists every problem found."""


# --- helpers ------------------------------------------------------------------------------------


def _require_mapping(value: Any, where: str, errors: List[str]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{where}: expected a mapping, got {type(value).__name__}")
        return {}
    return value


def _unknown_keys(section: Dict[str, Any], allowed: set, where: str, errors: List[str]) -> None:
    extra = sorted(set(section) - allowed)
    if extra:
        errors.append(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def _get_str(section: Dict[str, Any], key: str, where: str, errors: List[str], required: bool = False,
             default: Optional[str] = None) -> Optional[str]:
    if key not in section or section[key] is None:
        if required:
            errors.append(f"{where}.{key}: required")
        return default
    v = section[key]
    if not isinstance(v, str) or not v:
        errors.append(f"{where}.{key}: expected a non-empty string, got {v!r}")
        return default
    return v


def _get_number(section: Dict[str, Any], key: str, where: str, errors: List[str], default=None,
                positive: bool = False, integer: bool = False):
    if key not in section or section[key] is None:
        return default
    v = section[key]
    ok = isinstance(v, int) and not isinstance(v, bool) if integer else isinstance(v, (int, float)) and not isinstance(v, bool)
    if not ok:
        kind = "an integer" if integer else "a number"
        errors.append(f"{where}.{key}: expected {kind}, got {v!r}")
        return default
    if positive and v <= 0:
        errors.append(f"{where}.{key}: must be > 0, got {v!r}")
        return default
    return v


def _get_bool(section: Dict[str, Any], key: str, where: str, errors: List[str], default: bool) -> bool:
    if key not in section or section[key] is None:
        return default
    v = section[key]
    if not isinstance(v, bool):
        errors.append(f"{where}.{key}: expected true/false, got {v!r}")
        return default
    return v


def _get_str_list(section: Dict[str, Any], key: str, where: str, errors: List[str], required: bool = False) -> List[str]:
    if key not in section or section[key] is None:
        if required:
            errors.append(f"{where}.{key}: required (a non-empty list)")
        return []
    v = section[key]
    if not isinstance(v, list) or (required and not v) or not all(isinstance(x, str) and x for x in v):
        errors.append(f"{where}.{key}: expected a non-empty list of strings, got {v!r}")
        return []
    if len(set(v)) != len(v):
        errors.append(f"{where}.{key}: duplicate entries in {v!r}")
    return list(v)


# --- sections -----------------------------------------------------------------------------------


@dataclass
class RunSection:
    name: str
    out_dir: Path
    seed: int = 0

    @classmethod
    def parse(cls, raw: Dict[str, Any], base: Path, errors: List[str]) -> "RunSection":
        where = "run"
        _unknown_keys(raw, {"name", "out_dir", "seed"}, where, errors)
        name = _get_str(raw, "name", where, errors, required=True, default="run") or "run"
        out_dir = _get_str(raw, "out_dir", where, errors, required=True, default="runs/" + name) or ("runs/" + name)
        seed = _get_number(raw, "seed", where, errors, default=0, integer=True)
        if seed is not None and seed < 0:
            errors.append(f"{where}.seed: must be >= 0")
        return cls(name=name, out_dir=_resolve(base, out_dir), seed=int(seed or 0))


@dataclass
class GeneId:
    column: str = "index"  #: ``index`` (``var_names``) or a column of ``var``
    kind: str = "symbol"  #: ``symbol`` or ``ensembl``


@dataclass
class Labels:
    column: str  #: ``obs`` column with cell labels
    groups: Optional[Path] = None  #: optional CSV mapping labels to coarser groups


@dataclass
class Spatial:
    """Spatial coordinates and their unit.

    ``um_per_unit`` is the size in micrometres of one unit of the stored coordinates (for
    example 0.18 for CosMx pixels, 1.0 when coordinates are already in µm). It is the only
    place where a coordinate unit is declared; :mod:`interpretfm.units` does every conversion.
    """

    obsm: str = "spatial"
    um_per_unit: float = 1.0


@dataclass
class QC:
    """Cell quality flags: cells below either threshold are flagged at ingest, never dropped."""

    min_transcripts: int = 20
    min_genes: int = 10


@dataclass
class DatasetSection:
    h5ad: Path
    counts_layer: Optional[str] = None
    gene_id: GeneId = field(default_factory=GeneId)
    labels: Optional[Labels] = None
    spatial: Optional[Spatial] = None
    qc: QC = field(default_factory=QC)

    @property
    def is_spatial(self) -> bool:
        return self.spatial is not None

    @classmethod
    def parse(cls, raw: Dict[str, Any], base: Path, errors: List[str]) -> "DatasetSection":
        where = "dataset"
        _unknown_keys(raw, {"h5ad", "counts_layer", "gene_id", "labels", "spatial", "qc"}, where, errors)
        h5ad = _get_str(raw, "h5ad", where, errors, required=True, default="") or ""
        counts_layer = _get_str(raw, "counts_layer", where, errors)

        g = _require_mapping(raw.get("gene_id"), where + ".gene_id", errors)
        _unknown_keys(g, {"column", "kind"}, where + ".gene_id", errors)
        gene_id = GeneId(
            column=_get_str(g, "column", where + ".gene_id", errors, default="index") or "index",
            kind=_get_str(g, "kind", where + ".gene_id", errors, default="symbol") or "symbol",
        )
        if gene_id.kind not in ("symbol", "ensembl"):
            errors.append(f"{where}.gene_id.kind: expected 'symbol' or 'ensembl', got {gene_id.kind!r}")

        labels = None
        if raw.get("labels") is not None:
            lab = _require_mapping(raw.get("labels"), where + ".labels", errors)
            _unknown_keys(lab, {"column", "groups"}, where + ".labels", errors)
            col = _get_str(lab, "column", where + ".labels", errors, required=True)
            groups = _get_str(lab, "groups", where + ".labels", errors)
            if col:
                labels = Labels(column=col, groups=_resolve(base, groups) if groups else None)

        spatial = None
        if "spatial" in raw and raw["spatial"] is not None:
            sp = _require_mapping(raw["spatial"], where + ".spatial", errors)
            _unknown_keys(sp, {"obsm", "um_per_unit"}, where + ".spatial", errors)
            if "um_per_unit" not in sp:
                errors.append(
                    f"{where}.spatial.um_per_unit: required when 'spatial' is set "
                    "(micrometres per coordinate unit, e.g. 1.0 if coordinates are already in µm)"
                )
            upu = _get_number(sp, "um_per_unit", where + ".spatial", errors, default=1.0, positive=True)
            spatial = Spatial(obsm=_get_str(sp, "obsm", where + ".spatial", errors, default="spatial") or "spatial",
                              um_per_unit=float(upu))

        q = _require_mapping(raw.get("qc"), where + ".qc", errors)
        _unknown_keys(q, {"min_transcripts", "min_genes"}, where + ".qc", errors)
        qc = QC(
            min_transcripts=int(_get_number(q, "min_transcripts", where + ".qc", errors, default=20, integer=True) or 0),
            min_genes=int(_get_number(q, "min_genes", where + ".qc", errors, default=10, integer=True) or 0),
        )
        return cls(h5ad=_resolve(base, h5ad), counts_layer=counts_layer, gene_id=gene_id, labels=labels,
                   spatial=spatial, qc=qc)


@dataclass
class ModelEnv:
    """Where model code runs. Exactly one of the fields is set; all unset = the current interpreter.

    ``conda`` is an environment name, ``prefix`` an environment directory, ``python`` an
    interpreter path, ``container`` an image (reserved; containers are not supported yet).
    """

    conda: Optional[str] = None
    prefix: Optional[Path] = None
    python: Optional[Path] = None
    container: Optional[str] = None

    @property
    def kind(self) -> str:
        for k in ("conda", "prefix", "python", "container"):
            if getattr(self, k) is not None:
                return k
        return "current"

    @classmethod
    def parse(cls, raw: Any, base: Path, errors: List[str]) -> "ModelEnv":
        where = "model.env"
        env = _require_mapping(raw, where, errors)
        _unknown_keys(env, {"conda", "prefix", "python", "container"}, where, errors)
        set_keys = [k for k in ("conda", "prefix", "python", "container") if env.get(k) is not None]
        if len(set_keys) > 1:
            errors.append(f"{where}: set only one of conda/prefix/python/container, got {set_keys}")
        conda = _get_str(env, "conda", where, errors)
        prefix = _get_str(env, "prefix", where, errors)
        python = _get_str(env, "python", where, errors)
        container = _get_str(env, "container", where, errors)
        return cls(conda=conda, prefix=_resolve(base, prefix) if prefix else None,
                   python=_resolve(base, python) if python else None, container=container)


@dataclass
class CellPooling:
    kind: str = "token"  #: ``token`` (a cell-level token, e.g. scGPT's CLS) or ``mean``
    token: Optional[str] = None


@dataclass
class ModelSection:
    name: str
    checkpoint: Optional[Path]
    env: ModelEnv
    layers: List[str]
    cell_pooling: CellPooling = field(default_factory=CellPooling)
    special_tokens: List[str] = field(default_factory=list)
    representative_layer: Optional[str] = None
    vocab: Optional[Path] = None  #: JSON gene vocabulary of the model (``{name: id}``), for coverage at ingest

    @classmethod
    def parse(cls, raw: Dict[str, Any], base: Path, errors: List[str]) -> "ModelSection":
        where = "model"
        _unknown_keys(raw, {"name", "checkpoint", "env", "layers", "cell_pooling", "special_tokens",
                            "representative_layer", "vocab"}, where, errors)
        name = _get_str(raw, "name", where, errors, required=True, default="") or ""
        checkpoint = _get_str(raw, "checkpoint", where, errors)
        env = ModelEnv.parse(raw.get("env"), base, errors)
        layers = _get_str_list(raw, "layers", where, errors, required=True)
        cp = _require_mapping(raw.get("cell_pooling"), where + ".cell_pooling", errors)
        _unknown_keys(cp, {"kind", "token"}, where + ".cell_pooling", errors)
        pooling = CellPooling(kind=_get_str(cp, "kind", where + ".cell_pooling", errors, default="token") or "token",
                              token=_get_str(cp, "token", where + ".cell_pooling", errors))
        if pooling.kind not in ("token", "mean"):
            errors.append(f"{where}.cell_pooling.kind: expected 'token' or 'mean', got {pooling.kind!r}")
        if pooling.kind == "token" and pooling.token is None:
            errors.append(f"{where}.cell_pooling.token: required when kind is 'token'")
        special = _get_str_list(raw, "special_tokens", where, errors)
        rep = _get_str(raw, "representative_layer", where, errors)
        if rep is not None and layers and rep not in layers:
            errors.append(f"{where}.representative_layer: {rep!r} is not in model.layers")
        vocab = _get_str(raw, "vocab", where, errors)
        return cls(name=name, checkpoint=_resolve(base, checkpoint) if checkpoint else None, env=env, layers=layers,
                   cell_pooling=pooling, special_tokens=special, representative_layer=rep,
                   vocab=_resolve(base, vocab) if vocab else None)


@dataclass
class OutputSection:
    html: bool = True
    json: bool = True
    keep_store: bool = True

    @classmethod
    def parse(cls, raw: Dict[str, Any], errors: List[str]) -> "OutputSection":
        where = "output"
        _unknown_keys(raw, {"html", "json", "keep_store"}, where, errors)
        return cls(html=_get_bool(raw, "html", where, errors, True), json=_get_bool(raw, "json", where, errors, True),
                   keep_store=_get_bool(raw, "keep_store", where, errors, True))


@dataclass
class Config:
    """A validated pipeline configuration."""

    run: RunSection
    dataset: DatasetSection
    model: ModelSection
    stages: Dict[str, Dict[str, Any]]
    output: OutputSection
    source: Optional[Path] = None  #: the file the configuration was loaded from

    def stage_params(self, stage: str) -> Dict[str, Any]:
        """Parameters of one stage (an empty mapping when the section is absent)."""
        return dict(self.stages.get(stage, {}))

    def to_dict(self) -> Dict[str, Any]:
        """Plain, JSON-serialisable form (paths as strings), used for ``config.resolved.yaml`` and hashing."""
        d = dataclasses.asdict(self)
        d.pop("source", None)
        return _plain(d)

    def stage_hash(self, stage: str) -> str:
        """Hash of everything that can change the result of ``stage``.

        Covers the stage's own parameters, ``dataset``, ``model`` and ``run.seed``, so that a
        change to another stage's section, or to ``run.name``/``out_dir``, does not invalidate
        a cached result.
        """
        d = self.to_dict()
        relevant = {
            "dataset": d["dataset"],
            "model": d["model"],
            "seed": d["run"]["seed"],
            "stage": stage,
            "params": d["stages"].get(stage, {}),
            "output": d["output"] if stage == "report" else None,
        }
        return hash_dict(relevant)

    def write_resolved(self, path: Union[str, Path]) -> Path:
        """Write the resolved configuration (absolute paths, defaults filled in) as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return path


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def hash_dict(d: Dict[str, Any]) -> str:
    """Stable short hash of a JSON-serialisable mapping."""
    s = json.dumps(_plain(d), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _resolve(base: Path, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (base / q).resolve()


# --- entry points -------------------------------------------------------------------------------


def parse_config(raw: Dict[str, Any], base: Union[str, Path] = ".", source: Optional[Path] = None) -> Config:
    """Validate a loaded YAML mapping and return a :class:`Config`.

    Raises :class:`ConfigError` listing every problem found. Relative paths are resolved
    against ``base``.
    """
    base = Path(base).resolve()
    errors: List[str] = []
    if not isinstance(raw, dict):
        raise ConfigError("the configuration must be a YAML mapping with the sections "
                          f"{sorted(_TOP_LEVEL)}")
    _unknown_keys(raw, _TOP_LEVEL, "top level", errors)
    if "models" in raw:
        errors.append("top level: 'models' is not supported; one model per run (use one configuration per model)")
    for key in ("run", "dataset", "model"):
        if key not in raw:
            errors.append(f"top level: section '{key}' is required")

    run = RunSection.parse(_require_mapping(raw.get("run"), "run", errors), base, errors)
    dataset = DatasetSection.parse(_require_mapping(raw.get("dataset"), "dataset", errors), base, errors)
    model = ModelSection.parse(_require_mapping(raw.get("model"), "model", errors), base, errors)
    output = OutputSection.parse(_require_mapping(raw.get("output"), "output", errors), errors)

    stages_raw = _require_mapping(raw.get("stages"), "stages", errors)
    stages: Dict[str, Dict[str, Any]] = {}
    for name, params in stages_raw.items():
        if name not in PARAM_STAGES:
            errors.append(f"stages.{name}: unknown stage; expected one of {PARAM_STAGES}")
            continue
        stages[name] = dict(_require_mapping(params, f"stages.{name}", errors))

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    return Config(run=run, dataset=dataset, model=model, stages=stages, output=output, source=source)


def load_config(path: Union[str, Path]) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if raw is None:
        raise ConfigError(f"{path}: the file is empty")
    return parse_config(raw, base=path.parent, source=path.resolve())


def check_files(cfg: Config) -> List[str]:
    """Problems with files the configuration points to (missing dataset, label groups, checkpoint)."""
    problems = []
    if not cfg.dataset.h5ad.is_file():
        problems.append(f"dataset.h5ad: file not found: {cfg.dataset.h5ad}")
    if cfg.dataset.labels and cfg.dataset.labels.groups and not cfg.dataset.labels.groups.is_file():
        problems.append(f"dataset.labels.groups: file not found: {cfg.dataset.labels.groups}")
    if cfg.model.checkpoint is not None and not cfg.model.checkpoint.exists():
        problems.append(f"model.checkpoint: path not found: {cfg.model.checkpoint}")
    if cfg.model.vocab is not None and not cfg.model.vocab.is_file():
        problems.append(f"model.vocab: file not found: {cfg.model.vocab}")
    return problems
