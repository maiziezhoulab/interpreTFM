"""The per-model environment boundary.

The three foundation models need conflicting software stacks, so the stages that import model
code (``extract``, optionally ``sae``) run in the model's own environment as a subprocess. The
environment comes from ``model.env`` in the configuration (:class:`interpretfm.config.ModelEnv`):

- ``{conda: <name>}``: ``conda run -n <name> python ...``;
- ``{prefix: <dir>}``: ``conda run -p <dir> python ...``;
- ``{python: <interpreter>}``: that interpreter directly (a venv, a module-loaded Python, ...);
- nothing: the current interpreter (``sys.executable``);
- ``{container: <image>}``: reserved for the container release; not supported yet.

No interpreter path is hard-coded. ``conda`` is found through ``$CONDA_EXE`` (set by
``conda activate``) or on ``PATH``; set ``INTERPRETFM_CONDA`` to point at a specific binary.

The child runs ``python -m interpretfm.cli stage <name> --config <resolved.yaml> --run-dir
<dir>``, i.e. the same stage function as an in-process stage, with the run directory as its
working context. The parent never imports model code.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from interpretfm.config import ModelEnv


class EnvError(RuntimeError):
    """The model environment cannot be used (missing, not a conda env, conda not installed, ...)."""


def find_conda() -> Optional[str]:
    """The conda executable, or ``None`` if none can be found."""
    for cand in (os.environ.get("INTERPRETFM_CONDA"), os.environ.get("CONDA_EXE"), shutil.which("conda"),
                 shutil.which("mamba")):
        if cand and Path(cand).is_file():
            return cand
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


def resolve_env(env: ModelEnv) -> Dict[str, Any]:
    """Turn ``model.env`` into a command prefix that starts the right Python.

    Returns ``{"kind": ..., "command": [...], "description": ...}`` where ``command`` is the
    argument list to prepend to ``-m module ...``. Raises :class:`EnvError` with a clear
    message when the environment does not exist.
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
        raise EnvError("model.env.container: containers are not supported in this version; "
                       "use env.conda, env.prefix or env.python")
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
               cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run ``python -m <module> <args>`` in the model environment.

    Output goes to ``log_file`` when given (appended), otherwise to the parent's stdout/stderr.
    With ``check`` a non-zero exit raises :class:`EnvError` naming the log file.
    """
    resolved = resolve_env(env)
    cmd: List[str] = list(resolved["command"]) + ["-m", module] + list(args)
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
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


def env_summary(env: ModelEnv) -> Dict[str, Any]:
    """Resolution of ``env`` for the manifest, without raising (errors are recorded as text)."""
    try:
        r = resolve_env(env)
        return {"kind": r["kind"], "description": r["description"], "command": r["command"]}
    except EnvError as e:
        return {"kind": env.kind, "error": str(e)}
