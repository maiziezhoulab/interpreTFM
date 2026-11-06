"""Stage registry and the cached, resumable runner.

A stage is a function ``f(ctx: StageContext) -> dict | None`` registered under a name with
its upstream stages (``inputs``), the files it promises to write (``outputs``, relative to its
stage directory) and where it runs (``env="core"`` in the current interpreter, ``env="model"``
in the model's environment through :mod:`interpretfm.envs`)::

    from interpretfm.runner import register_stage, StageContext

    @register_stage("tis", inputs=["extract"], outputs=["tis.json"])
    def run_tis(ctx: StageContext):
        store = ctx.input_dir("extract")          # 02_extract/
        params = ctx.params                       # config.stages.tis
        ...
        ctx.write_json("tis.json", result)
        return {"n_layers": len(ctx.config.model.layers)}   # goes to manifest["extra"]

Run directory (``run.out_dir``)::

    config.resolved.yaml   the validated configuration with absolute paths and defaults
    manifest.json          run-level manifest
    log.txt                run log
    01_ingest/             one directory per stage: manifest.json, log.txt and the outputs
    02_extract/ ...

A stage is skipped when its directory holds a manifest with status ``done``, the same
configuration hash (:meth:`interpretfm.config.Config.stage_hash`) and all declared outputs
present; ``force`` re-runs it. Stages run in pipeline order; a stage whose ``applies``
predicate is false for the configuration (for example ``niches`` without coordinates) is
recorded as ``skipped``.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from interpretfm.config import STAGE_ORDER, Config
from interpretfm.manifest import MANIFEST_NAME, RunManifest, StageManifest, write_json

log = logging.getLogger("interpretfm")

StageFunc = Callable[["StageContext"], Optional[Dict[str, Any]]]
RESOLVED_CONFIG = "config.resolved.yaml"
RUN_LOG = "log.txt"


class StageError(RuntimeError):
    """A stage failed, or the run cannot proceed (missing upstream results, unknown stage)."""


@dataclass
class Stage:
    name: str
    func: StageFunc
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    env: str = "core"  #: ``core`` or ``model``
    applies: Optional[Callable[[Config], bool]] = None
    needs_model_env: Optional[Callable[[Config], bool]] = None  #: ``env="model"`` stages: False -> run in-process for this config
    order: int = 99
    doc: str = ""
    uses_labels: bool = False  #: whether the stage reads ``dataset.labels`` (then a label change invalidates its cache)
    hash_ignore: List[str] = field(default_factory=list)  #: stage parameters that cannot change the result (left out of the hash)
    input_overrides: Dict[str, str] = field(default_factory=dict)  #: input stage -> stage parameter that replaces it when set

    def runs_in_model_env(self, cfg: Config) -> bool:
        """Whether the runner starts this stage in ``model.env`` for ``cfg`` (a model stage with an
        environment other than the current interpreter, unless ``needs_model_env`` says no)."""
        if self.env != "model" or cfg.model.env.kind == "current":
            return False
        return self.needs_model_env is None or bool(self.needs_model_env(cfg))

    @property
    def dir_name(self) -> str:
        return f"{self.order:02d}_{self.name}"


_REGISTRY: Dict[str, Stage] = {}


def register_stage(name: str, *, inputs: Sequence[str] = (), outputs: Sequence[str] = (), env: str = "core",
                   applies: Optional[Callable[[Config], bool]] = None, order: Optional[int] = None,
                   replace: bool = False, uses_labels: bool = False,
                   needs_model_env: Optional[Callable[[Config], bool]] = None,
                   hash_ignore: Sequence[str] = (), input_overrides: Optional[Dict[str, str]] = None,
                   ) -> Callable[[StageFunc], StageFunc]:
    """Decorator registering ``func`` as the implementation of stage ``name``.

    ``order`` defaults to the position in :data:`interpretfm.config.STAGE_ORDER` (1-based) and
    99 for other names. Registering a name twice is an error unless ``replace`` is set (tests
    and plug-ins use it to swap an implementation). ``uses_labels`` declares that the stage reads
    ``dataset.labels``: only then does a change of the label column, file, groups or group map
    change the stage's configuration hash (:meth:`interpretfm.config.Config.stage_hash`), so that
    the label-free stages (extraction, TIS, SAE, concept evaluation) keep their cached results.
    ``env="model"`` stages run in ``model.env`` through :mod:`interpretfm.envs`;
    ``needs_model_env(cfg)`` can return False to keep a particular configuration in-process
    (``extract`` with a cached store), and ``model.env: {}`` always runs in-process.
    ``hash_ignore`` names stage parameters that only change how the stage computes, never what
    (``n_jobs``, thread counts): they are left out of the stage's configuration hash.
    ``input_overrides`` maps an input stage to the stage parameter that replaces its output
    (``niches``: ``cells_h5ad`` instead of the ``cells`` stage); when that parameter is set, the
    input stage need not have run.
    """
    if env not in ("core", "model"):
        raise ValueError(f"stage {name!r}: env must be 'core' or 'model', got {env!r}")

    def deco(func: StageFunc) -> StageFunc:
        if name in _REGISTRY and not replace:
            raise ValueError(f"stage {name!r} is already registered (use replace=True to override)")
        pos = order if order is not None else (STAGE_ORDER.index(name) + 1 if name in STAGE_ORDER else 99)
        _REGISTRY[name] = Stage(name=name, func=func, inputs=list(inputs), outputs=list(outputs), env=env,
                                applies=applies, needs_model_env=needs_model_env, order=pos,
                                doc=(func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else "",
                                uses_labels=bool(uses_labels), hash_ignore=[str(k) for k in hash_ignore],
                                input_overrides=dict(input_overrides or {}))
        return func

    return deco


def unregister_stage(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_stage(name: str) -> Stage:
    _ensure_builtin_stages()
    if name not in _REGISTRY:
        raise StageError(f"unknown stage {name!r}; registered: {[s.name for s in list_stages()]}")
    return _REGISTRY[name]


def list_stages() -> List[Stage]:
    """Registered stages in pipeline order."""
    _ensure_builtin_stages()
    return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.name))


def _ensure_builtin_stages() -> None:
    # Importing the package registers the built-in stages; done lazily to avoid an import cycle.
    import interpretfm.stages  # noqa: F401


def stage_hash_ignore(name: str) -> List[str]:
    """Parameters of a registered stage that are left out of its configuration hash (empty for
    names that are not registered)."""
    _ensure_builtin_stages()
    st = _REGISTRY.get(name)
    return [] if st is None else list(st.hash_ignore)


def stage_uses_labels(name: str) -> bool:
    """Whether a registered stage reads ``dataset.labels`` (``True`` for names that are not registered,
    the conservative choice for the run-level hash and for plug-in names)."""
    _ensure_builtin_stages()
    st = _REGISTRY.get(name)
    return True if st is None else bool(st.uses_labels)


# --- context ------------------------------------------------------------------------------------


@dataclass
class StageContext:
    """What a stage function receives."""

    config: Config
    run_dir: Path
    stage: Stage
    stage_dir: Path
    log: logging.Logger
    dry_run: bool = False

    @property
    def params(self) -> Dict[str, Any]:
        """The stage's parameters from ``config.stages.<name>``."""
        return self.config.stage_params(self.stage.name)

    def path(self, *parts: str) -> Path:
        """A path under the stage directory (parents created)."""
        p = self.stage_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def input_dir(self, stage_name: str) -> Path:
        """The directory of an upstream stage (it must be declared in ``inputs`` and be done)."""
        if stage_name not in self.stage.inputs:
            raise StageError(f"stage {self.stage.name!r} reads {stage_name!r} but does not declare it in inputs")
        d = stage_dir(self.run_dir, stage_name)
        m = StageManifest.read(d)
        if m is None or m.status not in ("done", "skipped"):
            raise StageError(f"stage {self.stage.name!r} needs {stage_name!r}, which has not completed ({d})")
        return d

    def write_json(self, name: str, obj: Any) -> Path:
        return write_json(self.path(name), obj)


def stage_dir(run_dir: Union[str, Path], stage_name: str) -> Path:
    return Path(run_dir) / get_stage(stage_name).dir_name


# --- caching ------------------------------------------------------------------------------------


def stage_is_current(st: Stage, run_dir: Path, cfg: Config) -> bool:
    """True when the stage's cached result matches the configuration and all outputs exist."""
    d = run_dir / st.dir_name
    m = StageManifest.read(d)
    if m is None or m.status != "done":
        return False
    if m.config_hash != cfg.stage_hash(st.name):
        return False
    return all((d / out).exists() for out in st.outputs)


def plan(cfg: Config, stages: Optional[Iterable[str]] = None, force: bool = False) -> List[Dict[str, Any]]:
    """What ``run`` would do: one entry per stage with ``action`` in run/skip-cached/skip-na."""
    run_dir = cfg.run.out_dir
    selected = list(stages) if stages else [s.name for s in list_stages()]
    for name in selected:
        get_stage(name)  # unknown stage -> StageError
    ordered = [s for s in list_stages() if s.name in selected]
    out = []
    for st in ordered:
        if st.applies is not None and not st.applies(cfg):
            action, reason = "skip-na", "not applicable to this configuration"
        elif not force and stage_is_current(st, run_dir, cfg):
            action, reason = "skip-cached", "outputs present and configuration unchanged"
        else:
            action, reason = "run", "forced" if force else "no valid cached result"
        out.append({"stage": st.name, "dir": st.dir_name, "env": "model" if st.runs_in_model_env(cfg) else "core",
                    "action": action, "reason": reason, "config_hash": cfg.stage_hash(st.name)})
    return out


# --- execution ----------------------------------------------------------------------------------


def _check_inputs(st: Stage, run_dir: Path, done_now: set, cfg: Optional[Config] = None) -> None:
    params = cfg.stage_params(st.name) if cfg is not None else {}
    for dep in st.inputs:
        if dep in done_now:
            continue
        if dep in st.input_overrides and params.get(st.input_overrides[dep]):
            continue  # the stage parameter supplies what the input stage would have produced
        m = StageManifest.read(stage_dir(run_dir, dep))
        if m is None or m.status not in ("done", "skipped"):
            raise StageError(f"stage {st.name!r} needs {dep!r}, which has not been run in {run_dir} "
                             f"(run it first, or include it in --stages)")


def _file_handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    h.setLevel(logging.INFO)
    return h


def run_stage_inprocess(cfg: Config, name: str, run_dir: Optional[Path] = None) -> StageManifest:
    """Execute one stage in this interpreter and write its manifest (used by ``run`` and by the
    ``interpretfm stage`` child process inside a model environment)."""
    st = get_stage(name)
    run_dir = Path(run_dir or cfg.run.out_dir)
    sdir = run_dir / st.dir_name
    sdir.mkdir(parents=True, exist_ok=True)
    handler = _file_handler(sdir / "log.txt")
    log.addHandler(handler)
    manifest = StageManifest(stage=name, config_hash=cfg.stage_hash(name),
                             inputs={dep: get_stage(dep).dir_name for dep in st.inputs},
                             outputs={out: f"{st.dir_name}/{out}" for out in st.outputs})
    manifest.write(sdir)
    ctx = StageContext(config=cfg, run_dir=run_dir, stage=st, stage_dir=sdir, log=log)
    log.info("stage %s: start (%s)", name, sdir)
    try:
        extra = st.func(ctx)
        missing = [out for out in st.outputs if not (sdir / out).exists()]
        if missing:
            raise StageError(f"stage {name!r} finished without writing its declared outputs: {missing}")
        manifest.extra = dict(extra or {})
        manifest.finish("done")
        log.info("stage %s: done in %.1f s", name, manifest.duration_s or 0.0)
    except Exception as e:  # noqa: BLE001 - recorded in the manifest, then re-raised
        manifest.finish("failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        manifest.write(sdir)
        log.error("stage %s: failed: %s", name, e)
        log.removeHandler(handler)
        handler.close()
        raise StageError(f"stage {name!r} failed: {e}") from e
    manifest.write(sdir)
    log.removeHandler(handler)
    handler.close()
    return manifest


def model_stage_args(cfg: Config, st: Stage, run_dir: Path) -> List[str]:
    """The ``interpretfm.cli`` arguments that run ``st`` in a child process: the ``extract`` verb
    (with the model and layers, checked by the child against the resolved configuration) or the
    generic ``stage`` verb."""
    resolved = run_dir / RESOLVED_CONFIG
    if st.name == "extract":
        return ["extract", "--config", str(resolved), "--run-dir", str(run_dir), "--model", cfg.model.name,
                "--layers", ",".join(cfg.model.layers)]
    return ["stage", st.name, "--config", str(resolved), "--run-dir", str(run_dir)]


def _run_stage_in_model_env(cfg: Config, st: Stage, run_dir: Path) -> StageManifest:
    from interpretfm.envs import EnvError, config_binds, env_summary, run_in_env

    sdir = run_dir / st.dir_name
    sdir.mkdir(parents=True, exist_ok=True)
    binds = config_binds(cfg, run_dir)
    summary = env_summary(cfg.model.env, binds=binds, cwd=run_dir)
    log.info("stage %s: running in model environment: %s", st.name, summary.get("description") or summary.get("error"))
    if "error" in summary:
        m = StageManifest(stage=st.name, config_hash=cfg.stage_hash(st.name), model_env=summary)
        m.finish("failed", error=summary["error"]).write(sdir)
        raise StageError(f"stage {st.name!r}: {summary['error']}")
    args = model_stage_args(cfg, st, run_dir)
    try:
        run_in_env(cfg.model.env, "interpretfm.cli", args, log_file=sdir / "log.txt", cwd=run_dir, binds=binds)
    except EnvError as e:
        m = StageManifest.read(sdir)
        if m is None or m.status != "failed":
            m = StageManifest(stage=st.name, config_hash=cfg.stage_hash(st.name), model_env=summary)
            m.finish("failed", error=str(e)).write(sdir)
        raise StageError(f"stage {st.name!r} failed in the model environment: {e}") from e
    m = StageManifest.read(sdir)
    if m is None or m.status != "done":
        raise StageError(f"stage {st.name!r}: the child process exited 0 but left no 'done' manifest in {sdir}")
    m.model_env = summary
    m.write(sdir)
    return m


def run(cfg: Config, stages: Optional[Iterable[str]] = None, force: bool = False, dry_run: bool = False,
        ) -> List[Dict[str, Any]]:
    """Run the pipeline (or the selected stages) for ``cfg``; returns the plan with a ``status`` per stage.

    ``force`` re-runs the selected stages even when cached; ``dry_run`` only reports the plan.
    Stops at the first failing stage with :class:`StageError` (the manifest records the error).
    """
    steps = plan(cfg, stages, force=force)
    if dry_run:
        for s in steps:
            s["status"] = "dry-run"
        return steps

    run_dir = cfg.run.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = _file_handler(run_dir / RUN_LOG)
    log.addHandler(handler)
    try:
        cfg.write_resolved(run_dir / RESOLVED_CONFIG)
        rm = RunManifest.read(run_dir) or RunManifest(name=cfg.run.name, config_hash=cfg.stage_hash("__run__"),
                                                      config_file=str(cfg.source) if cfg.source else None)
        rm.config_hash = cfg.stage_hash("__run__")
        rm.stages = [s.dir_name for s in list_stages()]
        rm.write(run_dir)
        log.info("run %s in %s (%d stages)", cfg.run.name, run_dir, len(steps))

        done_now: set = set()
        for s in steps:
            st = get_stage(s["stage"])
            if s["action"] == "skip-na":
                sdir = run_dir / st.dir_name
                sdir.mkdir(parents=True, exist_ok=True)
                StageManifest(stage=st.name, config_hash=cfg.stage_hash(st.name)).finish("skipped").write(sdir)
                s["status"] = "skipped"
                log.info("stage %s: skipped (%s)", st.name, s["reason"])
                done_now.add(st.name)
                continue
            if s["action"] == "skip-cached":
                s["status"] = "cached"
                log.info("stage %s: cached (%s)", st.name, s["reason"])
                done_now.add(st.name)
                continue
            _check_inputs(st, run_dir, done_now, cfg)
            if st.runs_in_model_env(cfg):
                m = _run_stage_in_model_env(cfg, st, run_dir)
            else:
                m = run_stage_inprocess(cfg, st.name, run_dir)
            s["status"] = m.status
            done_now.add(st.name)
        rm.write(run_dir)
    finally:
        log.removeHandler(handler)
        handler.close()
    return steps


def status(run_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Status of every registered stage in ``run_dir`` from the stage manifests."""
    run_dir = Path(run_dir)
    rows = []
    for st in list_stages():
        m = StageManifest.read(run_dir / st.dir_name)
        rows.append({
            "stage": st.name,
            "dir": st.dir_name,
            "status": m.status if m else "-",
            "config_hash": m.config_hash if m else "",
            "started": m.started if m else "",
            "duration_s": m.duration_s if m else None,
            "error": (m.error or "").splitlines()[0] if m and m.error else "",
        })
    return rows


__all__ = ["MANIFEST_NAME", "Stage", "StageContext", "StageError", "get_stage", "list_stages", "plan",
           "register_stage", "run", "run_stage_inprocess", "stage_dir", "stage_is_current", "stage_uses_labels", "status",
           "unregister_stage"]
