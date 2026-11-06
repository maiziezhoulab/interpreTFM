# Development and reproduction shortcuts. `make help` lists the targets.
#
# The reproduction targets read the data locations from an env file (default
# configs/paper/paths.env, see configs/paper/paths.env.example and docs/reproducing.md).

PYTHON   ?= python
ENV_FILE ?= configs/paper/paths.env
CONFIG   ?= cosmx_scgpt_layer2
STAGES   ?= ingest,extract,tis,sae,concepts,concept_eval,gene_readout,cells,layers,niches
PYTEST   ?= $(PYTHON) -m pytest

.PHONY: help lint test docs repro repro-validate repro-test repro-status

help:
	@echo "lint            ruff check ."
	@echo "test            unit tests (reproduction tests are skipped without the data variables)"
	@echo "docs            sphinx-build -W of docs/"
	@echo "repro           run the paper configuration CONFIG=$(CONFIG) (stages STAGES=$(STAGES)) from the caches"
	@echo "repro-validate  validate every configs/paper/*.yaml against ENV_FILE=$(ENV_FILE)"
	@echo "repro-status    stage status of the CONFIG run"
	@echo "repro-test      the reproduction tests (pytest -m reproduction) against finished runs"

lint:
	ruff check .

test:
	$(PYTEST)

docs:
	sphinx-build -W -b html docs docs/_build/html

repro-validate:
	@for f in configs/paper/*.yaml; do $(PYTHON) -m interpretfm.cli validate $$f --env-file $(ENV_FILE) || exit 1; done

repro:
	$(PYTHON) -m interpretfm.cli run configs/paper/$(CONFIG).yaml --env-file $(ENV_FILE) --stages $(STAGES)

repro-status:
	@set -a; . ./$(ENV_FILE); set +a; $(PYTHON) -m interpretfm.cli status "$$INTERPRETFM_RUNS/$(CONFIG)"

repro-test:
	INTERPRETFM_ENV_FILE=$(ENV_FILE) $(PYTEST) tests/reproduction -m reproduction -rA -s
