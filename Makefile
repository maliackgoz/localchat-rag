VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
HF_OFFLINE_ENV = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ROSTER ?= data/roster.json
RAW_DIR ?= data/raw
CHUNKS_DIR ?= data/chunks

.PHONY: install test chunk probe ingest build data run-ui run-cli eval

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m unittest

ingest:
	$(PYTHON) -m ingest.wikipedia --roster $(ROSTER) --out $(RAW_DIR)

chunk:
	$(PYTHON) -m chunking.splitter --raw $(RAW_DIR) --out $(CHUNKS_DIR)

build:
	$(PYTHON) -m store.vector_store --build

# Full pipeline: ingest+chunk+build. Omit when submission already contains data/{raw,chunks,chroma,...}; rebuild Chroma-only with target `build`.
data: ingest chunk build

probe:
	$(HF_OFFLINE_ENV) $(PYTHON) -m embedding.encoder --probe

eval:
	$(HF_OFFLINE_ENV) $(PYTHON) -m eval.run_eval --golden eval/golden.jsonl --report eval/results/

run-ui:
	$(PYTHON) -m streamlit run app/localchat_rag.py

run-cli:
	$(PYTHON) -m app.cli
