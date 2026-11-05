"""Built-in pipeline stages, registered on import in pipeline order.

``ingest`` is implemented (minimal version); the others are stubs with the final signatures
and declared inputs/outputs, to be filled in by the scientific work packages. A stub raises
:class:`NotImplementedError` with a clear message when it is run.
"""
from interpretfm.stages import cells, concepts, extract, ingest, layers, report, sae, spatial, tis  # noqa: F401
