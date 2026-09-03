VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

install:
	$(PYTHON) -m pip install '.[dev]'

run:
	$(PYTHON) -m mini_inference_engine.server

test:
	$(PYTHON) -m pytest -q

benchmark:
	$(PYTHON) -m mini_inference_engine.benchmarks.cli --endpoint http://127.0.0.1:8000 --requests 100 --concurrency 8
