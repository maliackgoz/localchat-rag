VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
HF_OFFLINE_ENV = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.PHONY: install test chunk probe

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m unittest

chunk:
	$(PYTHON) -m chunking.splitter --raw data/raw --out data/chunks

probe:
	$(HF_OFFLINE_ENV) $(PYTHON) -m embedding.encoder --probe
