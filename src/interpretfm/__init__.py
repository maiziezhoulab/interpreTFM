"""interpreTFM v1 pipeline: configuration, stage runner and command-line entry point.

The scientific code lives in :mod:`interp_pipeline` (the 0.9 package, unchanged and still
importable). This package adds the layer that turns it into a pipeline:

- :mod:`interpretfm.config`: the YAML configuration schema and its validation;
- :mod:`interpretfm.units`: the single place where spatial coordinates are converted to µm;
- :mod:`interpretfm.runner`: stage registry, caching and resumption;
- :mod:`interpretfm.manifest`: the ``manifest.json`` written for the run and for each stage;
- :mod:`interpretfm.envs`: the subprocess boundary to per-model conda environments;
- :mod:`interpretfm.stages`: the pipeline stages (several are stubs in this development version);
- :mod:`interpretfm.cli`: the ``interpretfm`` command.
"""
from interp_pipeline import __version__

from interpretfm.config import Config, ConfigError, load_config
from interpretfm.runner import StageContext, get_stage, list_stages, register_stage, run

__all__ = [
    "Config",
    "ConfigError",
    "StageContext",
    "__version__",
    "get_stage",
    "list_stages",
    "load_config",
    "register_stage",
    "run",
]
