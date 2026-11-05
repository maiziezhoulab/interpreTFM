"""Reading stage parameters (``config.stages.<name>``) with defaults and clear errors.

Stage sections are free mappings (not schema-checked); these helpers validate inside a
stage and fail early with a message naming the key.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


def get_param(params: Dict[str, Any], key: str, default: Any, stage: str, kind: Optional[type] = None,
              choices: Optional[Sequence[Any]] = None) -> Any:
    """``params[key]`` or ``default``; checks the type (``bool`` is not accepted for numbers) and choices."""
    value = params.get(key, default)
    if value is None:
        return None
    if kind is not None:
        ok = isinstance(value, kind) and not (kind in (int, float) and isinstance(value, bool))
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            ok = True
        if not ok:
            raise ValueError(f"stages.{stage}.{key}: expected {kind.__name__}, got {value!r}")
    if choices is not None and value not in choices:
        raise ValueError(f"stages.{stage}.{key}: expected one of {list(choices)}, got {value!r}")
    return value


def get_list(params: Dict[str, Any], key: str, default: Sequence[Any], stage: str,
             choices: Optional[Sequence[Any]] = None) -> list:
    value = params.get(key, default)
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"stages.{stage}.{key}: expected a list, got {value!r}")
    value = list(value)
    if choices is not None:
        bad = [v for v in value if v not in choices]
        if bad:
            raise ValueError(f"stages.{stage}.{key}: unknown entries {bad}; expected a subset of {list(choices)}")
    return value


def get_mapping(params: Dict[str, Any], key: str, stage: str) -> Dict[str, Any]:
    value = params.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"stages.{stage}.{key}: expected a mapping, got {value!r}")
    return dict(value)


def param_path(cfg, value: Union[str, Path], stage: str, key: str, must_exist: bool = True) -> Path:
    """A path given in a stage parameter: absolute, or relative to the configuration file's directory.

    Stage parameters are not resolved when the configuration is loaded, so a relative path
    is interpreted when the stage runs; a stage running in the model environment reads the
    resolved configuration from the run directory, so absolute paths are the safe choice.
    """
    p = Path(value).expanduser()
    if not p.is_absolute():
        base = Path(cfg.source).parent if cfg.source is not None else Path.cwd()
        p = (base / p).resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"stages.{stage}.{key}: {p} does not exist")
    return p
