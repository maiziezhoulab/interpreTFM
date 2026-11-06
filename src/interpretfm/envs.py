"""The per-model environment boundary.

The three foundation models need conflicting software stacks, so the stages that import model
code (``extract``, optionally ``sae``) run in the model's own environment as a subprocess. The
environment comes from ``model.env`` in the configuration (:class:`interpretfm.config.ModelEnv`):

- ``{conda: <name>}``: ``conda run -n <name> python ...``;
- ``{prefix: <dir>}``: ``conda run -p <dir> python ...``;
- ``{python: <interpreter>}``: that interpreter directly (a venv, a module-loaded Python, ...);
- ``{container: <image>}``: ``apptainer exec --nv <image> python ...`` (or ``singularity``,
  ``docker run --gpus all``, ``podman run --device nvidia.com/gpu=all``) with the run directory
  and the configured data paths bound into the container; the runtime is the first of
  ``apptainer``, ``singularity``, ``docker``, ``podman`` on ``PATH`` or the one named by
  ``INTERPRETFM_CONTAINER_RUNTIME``;
- nothing: the current interpreter, in-process (one environment for everything).

No interpreter path is hard-coded. ``conda`` is found through ``$CONDA_EXE`` (set by
``conda activate``) or on ``PATH``; set ``INTERPRETFM_CONDA`` to point at a specific binary.

The child runs ``python -m interpretfm.cli extract --config <resolved.yaml> --run-dir <dir>
--model <name> --layers ...`` (``interpretfm.cli stage <name> ...`` for other model stages),
i.e. the same stage function as an in-process stage, with the run directory as its working
context. The package must be importable there: installed in the environment, or on
``PYTHONPATH``, which the child inherits. The parent never imports model code; it reads the
manifests the child writes. :func:`child_environment` is what the child records in them
(interpreter, torch and CUDA versions, every installed distribution).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from interpretfm.config import ModelEnv

CONTAINER_RUNTIMES = ("apptainer", "singularity", "docker", "podman")


class EnvError(RuntimeError):
    """The model environment cannot be used (missing, not a conda env, conda not installed, ...)."""


def find_conda() -> Optional[str]:
    """The conda executable, or ``None`` if none can be found."""
    for cand in (os.environ.get("INTERPRETFM_CONDA"), os.environ.get("CONDA_EXE"), shutil.which("conda"),
                 shutil.which("mamba")):
        if cand and Path(cand).is_file():
            return cand
    return None


def find_container_runtime() -> Optional[str]:
    """The container runtime: ``INTERPRETFM_CONTAINER_RUNTIME`` (a name or a path) or the first of
    :data:`CONTAINER_RUNTIMES` on ``PATH``; ``None`` when there is none."""
    named = os.environ.get("INTERPRETFM_CONTAINER_RUNTIME")
    if named:
        return named if Path(named).is_file() else shutil.which(named)
    for name in CONTAINER_RUNTIMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def conda_env_names(conda: str) -> Dict[str, str]:
    """Mapping ``name -> prefix`` of the environments known to ``conda``."""
    try:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, check=True, timeout=120)
        prefixes = json.loads(out.stdout).get("envs", [])
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        raise EnvError(f"could not list conda environments with {conda!r}: {e}") from e
    names = {}
    for p in prefixes:
        p = str(p)
        names[Path(p).name] = p
    # conda reports the base environment by its install directory, not as "base"
    if prefixes:
        names.setdefault("base", str(prefixes[0]))
    return names


def _bind_paths(binds: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for b in binds:
        if b is None:
            continue
        p = Path(str(b)).expanduser()
        if not p.exists():
            continue
        p = p.resolve()
        if p.is_file():
            p = p.parent
        s = str(p)
        if s != "/" and not any(s == o or s.startswith(o + "/") for o in out):
            out = [o for o in out if not o.startswith(s + "/")] + [s]
    return out


def container_command(runtime: str, image: str, binds: Sequence[str] = (), cwd: Optional[str] = None) -> List[str]:
    """The command prefix that starts ``python`` inside ``image`` with ``binds`` mounted at their own paths."""
    name = Path(runtime).name
    if name in ("apptainer", "singularity"):
        cmd = [runtime, "exec", "--nv"]
        for b in binds:
            cmd += ["--bind", b]
        if cwd:
            cmd += ["--pwd", cwd]
        return cmd + [image, "python"]
    cmd = [runtime, "run", "--rm"]
    cmd += ["--device", "nvidia.com/gpu=all"] if name == "podman" else ["--gpus", "all"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for b in binds:
        cmd += ["-v", f"{b}:{b}"]
    if cwd:
        cmd += ["-w", cwd]
    return cmd + [image, "python"]


def resolve_env(env: ModelEnv, binds: Iterable[Any] = (), cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Turn ``model.env`` into a command prefix that starts the right Python.

    Returns ``{"kind": ..., "command": [...], "description": ...}`` where ``command`` is the
    argument list to prepend to ``-m module ...``. ``binds`` are host paths a container must
    see (ignored for the other kinds). Raises :class:`EnvError` with a clear message when the
    environment does not exist.
    """
    kind = env.kind
    if kind == "current":
        return {"kind": kind, "command": [sys.executable], "description": f"current interpreter {sys.executable}"}
    if kind == "python":
        py = Path(env.python)
        if not py.is_file():
            raise EnvError(f"model.env.python: interpreter not found: {py}")
        if not os.access(py, os.X_OK):
            raise EnvError(f"model.env.python: not executable: {py}")
        return {"kind": kind, "command": [str(py)], "description": f"interpreter {py}"}
    if kind == "container":
        runtime = find_container_runtime()
        if runtime is None:
            raise EnvError("model.env.container: no container runtime found (none of "
                           f"{', '.join(CONTAINER_RUNTIMES)} on PATH, INTERPRETFM_CONTAINER_RUNTIME unset); "
                           "install one or use env.conda, env.prefix or env.python")
        mounts = _bind_paths(list(binds) + ([cwd] if cwd else []))
        cmd = container_command(runtime, env.container, mounts, str(Path(cwd).resolve()) if cwd else None)
        return {"kind": kind, "command": cmd, "description": f"container {env.container} via {Path(runtime).name}",
                "runtime": runtime, "binds": mounts}
    conda = find_conda()
    if conda is None:
        raise EnvError(f"model.env.{kind}: conda was not found (not on PATH, and neither CONDA_EXE nor "
                       "INTERPRETFM_CONDA is set); install conda or point model.env.python at an interpreter")
    if kind == "conda":
        names = conda_env_names(conda)
        if env.conda not in names:
            raise EnvError(f"model.env.conda: conda environment '{env.conda}' does not exist "
                           f"(known: {sorted(names)}); create it, e.g. `conda env create -f envs/{env.conda}.yml`")
        return {"kind": kind, "command": [conda, "run", "--no-capture-output", "-n", env.conda, "python"],
                "description": f"conda env {env.conda} ({names[env.conda]})", "conda": conda, "prefix": names[env.conda]}
    prefix = Path(env.prefix)
    if not (prefix / "bin" / "python").is_file() and not (prefix / "python.exe").is_file():
        raise EnvError(f"model.env.prefix: no Python interpreter under {prefix} (expected {prefix / 'bin' / 'python'})")
    return {"kind": kind, "command": [conda, "run", "--no-capture-output", "-p", str(prefix), "python"],
            "description": f"conda prefix {prefix}", "conda": conda, "prefix": str(prefix)}


def run_in_env(env: ModelEnv, module: str, args: Sequence[str], log_file: Optional[Path] = None,
               cwd: Optional[Path] = None, check: bool = True, binds: Iterable[Any] = (),
               extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run ``python -m <module> <args>`` in the model environment.

    Output goes to ``log_file`` when given (appended), otherwise to the parent's stdout/stderr.
    The child inherits the environment (``PYTHONPATH``, ``CUDA_VISIBLE_DEVICES``, thread
    limits ...) plus ``extra_env``. With ``check`` a non-zero exit raises :class:`EnvError`
    naming the log file.
    """
    resolved = resolve_env(env, binds=binds, cwd=cwd)
    cmd: List[str] = list(resolved["command"]) + ["-m", module] + list(args)
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.update(extra_env or {})
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=child_env)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=child_env)
    if check and proc.returncode != 0:
        where = f"; see {log_file}" if log_file else ""
        raise EnvError(f"`{' '.join(cmd)}` exited with status {proc.returncode}{where}")
    return proc


def env_summary(env: ModelEnv, binds: Iterable[Any] = (), cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Resolution of ``env`` for the manifest, without raising (errors are recorded as text)."""
    try:
        r = resolve_env(env, binds=binds, cwd=cwd)
        out = {"kind": r["kind"], "description": r["description"], "command": r["command"]}
        if "binds" in r:
            out["binds"] = r["binds"]
        return out
    except EnvError as e:
        return {"kind": env.kind, "error": str(e)}


def config_binds(cfg, run_dir: Optional[Path] = None) -> List[Any]:
    """Host paths a container needs for a run: the run directory, the dataset, labels, weights,
    vocabulary and the paths named in ``stages.extract``."""
    paths: List[Any] = [run_dir or cfg.run.out_dir, cfg.dataset.h5ad, cfg.model.checkpoint, cfg.model.vocab]
    if cfg.dataset.labels is not None:
        paths += [cfg.dataset.labels.file, cfg.dataset.labels.groups]
    for key in ("store_root", "cell_subset", "token_vocab"):
        v = cfg.stage_params("extract").get(key)
        if isinstance(v, str):
            paths.append(v)
    if cfg.source is not None:
        paths.append(Path(cfg.source).parent)
    return paths


def installed_distributions() -> List[str]:
    """``name==version`` of every installed distribution (what ``pip freeze`` reports), sorted."""
    from importlib import metadata

    out = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            out.add(f"{name}=={dist.version}")
    return sorted(out, key=str.lower)


def child_environment() -> Dict[str, Any]:
    """What a model-environment child records in its manifests: interpreter, platform, torch and
    CUDA versions, the visible GPUs and every installed distribution."""
    import platform
    import socket

    info: Dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                            "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "packages": installed_distributions()}
    try:
        import torch

        t: Dict[str, Any] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())}
        if hasattr(torch.backends, "cudnn"):
            t["cudnn"] = torch.backends.cudnn.version()
        if t["cuda_available"]:
            t["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["torch"] = t
    except ImportError:
        info["torch"] = None
    return info
