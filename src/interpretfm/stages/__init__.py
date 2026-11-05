"""Built-in pipeline stages, registered on import in pipeline order.

``ingest`` is implemented (minimal version) and ``extract`` has a cached-store mode; the others are stubs with the final signatures
and declared inputs/outputs, to be filled in by the scientific work packages. A stub raises
:class:`NotImplementedError` with a clear message when it is run.
"""
from interpretfm.stages import (  # noqa: F401
    cells, concept_eval, concepts, extract, gene_readout, ingest, layers, report, sae, spatial, tis,
)
