"""Built-in pipeline stages, registered on import in pipeline order.

Every stage of the v1 pipeline is implemented; ``extract`` runs from a cached activation store
(``stages.extract.store_root``) until the model adapters are wired in. The spatial stages
(``niches``; the interface test and the ligand-receptor screen follow) apply only when the
dataset declares coordinates and are recorded as ``skipped`` otherwise.
"""
from interpretfm.stages import (  # noqa: F401
    cells, concept_eval, concepts, extract, gene_readout, ingest, layers, niches, report, sae, tis,
)
