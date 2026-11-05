"""Run and stage manifests (``manifest.json``).

The run directory holds one ``manifest.json`` for the run and one per stage directory. A stage
manifest records what the stage consumed and produced, the hash of the configuration that
determines its result, the software environment, timing and status. The runner uses it to
decide whether a stage can be skipped (:func:`interpretfm.runner.stage_is_current`).

Status values: ``running``, ``done``, ``failed``, ``skipped`` (not applicable, for example a
spatial stage on a dataset without coordinates).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import platform
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from interp_pipeline import __version__

MANIFEST_NAME = "manifest.json"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def environment_info() -> Dict[str, Any]:
    """Interpreter, platform and the versions of the packages the pipeline depends on."""
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {"interpretfm": __version__}
    for dist in ("numpy", "pandas", "scipy", "torch", "anndata", "scanpy", "pyyaml"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "versions": versions,
    }


@dataclass
class StageManifest:
    """What one stage run consumed, produced and in which environment it ran."""

    stage: str
    status: str = "running"
    config_hash: str = ""
    package_version: str = __version__
    inputs: Dict[str, str] = field(default_factory=dict)  #: name -> path (relative to the run dir)
    outputs: Dict[str, str] = field(default_factory=dict)  #: name -> path (relative to the run dir)
    environment: Dict[str, Any] = field(default_factory=environment_info)
    model_env: Optional[Dict[str, Any]] = None  #: how the model environment was resolved, when used
    started: str = field(default_factory=_now)
    ended: Optional[str] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  #: free-form, stage-specific summary

    def finish(self, status: str, error: Optional[str] = None) -> "StageManifest":
        self.status = status
        self.error = error
        self.ended = _now()
        t0 = _dt.datetime.fromisoformat(self.started)
        t1 = _dt.datetime.fromisoformat(self.ended)
        self.duration_s = round((t1 - t0).total_seconds(), 3)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, stage_dir: Union[str, Path]) -> Path:
        return write_json(Path(stage_dir) / MANIFEST_NAME, self.to_dict())

    @classmethod
    def read(cls, stage_dir: Union[str, Path]) -> Optional["StageManifest"]:
        """The manifest in ``stage_dir`` or ``None`` when there is none (or it is unreadable)."""
        path = Path(stage_dir) / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunManifest:
    """The manifest at the root of a run directory."""

    name: str
    config_hash: str
    config_file: Optional[str]
    package_version: str = __version__
    environment: Dict[str, Any] = field(default_factory=environment_info)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    stages: List[str] = field(default_factory=list)  #: stage directory names in order
    command: List[str] = field(default_factory=lambda: list(sys.argv))

    def write(self, run_dir: Union[str, Path]) -> Path:
        self.updated = _now()
        return write_json(Path(run_dir) / MANIFEST_NAME, dataclasses.asdict(self))

    @classmethod
    def read(cls, run_dir: Union[str, Path]) -> Optional["RunManifest"]:
        path = Path(run_dir) / MANIFEST_NAME
        if not path.is_file():
            return None
        d = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def write_json(path: Union[str, Path], obj: Any) -> Path:
    """Write ``obj`` as indented JSON, atomically (write to a temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    tmp.replace(path)
    return path
