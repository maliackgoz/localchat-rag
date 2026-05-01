# localchat-rag — Local Wikipedia RAG Assistant

A ChatGPT-style assistant that answers questions about a fixed roster of famous people and places using only **locally-running components**: Ollama for the LLM, sentence-transformers for embeddings, Chroma for the vector store, and a Streamlit chat UI. **No external LLM API** is used — the system runs entirely on the user's machine.

This is **Project 3** for **BLG 483E (AI-Aided Computer Engineering)**, combining the indexer-style retrieval from Project 1 with the AI workflows from Project 2 into a complete **retrieval-augmented generation (RAG)** application.

---

## How it's built — multi-agent development

The codebase is organized around **multi-agent development**: nine developer roles, one per area of the system, each with its own contract document under [`agents/`](agents/). Agents do **not** talk to each other at runtime — the running system is one Python process. **"Agent" here means a developer role with a defined scope, not a runtime entity.**

| # | Agent | Scope |
|---|-------|-------|
| 0 | [Orchestrator](agents/00_orchestrator.md) | PRD, roadmap, integration |
| 1 | [Ingestion](agents/01_ingestion.md) | Wikipedia REST → `data/raw/` |
| 2 | [Chunking + Embedding](agents/02_chunking_embedding.md) | Sentence-aware splitter + MiniLM |
| 3 | [Vector Store](agents/03_vector_store.md) | Chroma persistence (one collection, `type` metadata) |
| 4 | [Retrieval](agents/04_retrieval.md) | Query router + filtered similarity search |
| 5 | [Generation](agents/05_generation.md) | Ollama + grounded prompt + IDK guard |
| 6 | [Chat UI](agents/06_chat_ui.md) | Streamlit (primary), CLI (fallback) |
| 7 | [QA + Eval](agents/07_qa_eval.md) | Golden-question harness, latency report |
| 8 | [Docs](agents/08_docs.md) | README, recommendation, demo script |

Supporting documents:

- [`AGENTS.md`](AGENTS.md) — agent index, shared contract, lifecycle, prompt template, conflict resolution
- [`product_prd.md`](product_prd.md) — authoritative requirements and locked design decisions
- [`roadmap.md`](roadmap.md) — phased milestones (M0–M8) with exit criteria
- [`recommendation.md`](recommendation.md) — production-deployment notes and scale-up trade-offs

### How the multi-agent system was actually used during development

This project was built with two AI coding tools playing different roles. The split was driven by token budgets:

1. **Architecture phase — Claude Code with Opus 4.7 (extra-high thinking).** The strongest model was used up front to design the full multi-agent contract surface: the PRD, the nine agent files, the roadmap, the workflow document, and the locked decisions. Heavier reasoning at this stage paid off because every later edit defers to these documents.
2. **Implementation phase — Cursor with smaller / cheaper models.** Claude Code's session token limits exhausted quickly, so the per-milestone implementation work moved to Cursor. To keep the smaller models faithful to the contracts produced in step 1, **each milestone was started with an explicit `@` mention of the relevant agent file and the PRD**, e.g.:

   ```
   @agents/01_ingestion.md @product_prd.md implement M1.
   ```

   That single line forces Cursor's context window to include the contract before any code is written.
3. **Cursor rules in `.cursor/rules/` enforce the contracts automatically.** A per-area `*.mdc` rule (with a `globs:` pattern) auto-attaches the right agent contract whenever a file under that area is edited. The orchestrator rule (`alwaysApply: true`) carries the global constraints (no external LLM API, locked decisions, ownership map) into every prompt. This catches drift that would otherwise creep in when a smaller model edits, say, `generation/` without re-reading `agents/05_generation.md`.

The takeaway: a strong model produces the contracts once; a cheaper model implements against them — and Cursor's rules system is what keeps the cheaper model honest.

---

## Prerequisites

| Requirement | Why |
|------|-----|
| [GNU Make](https://www.gnu.org/software/make/) | Recommended path uses `Makefile` targets (bundled on macOS/Linux) |
| Python 3.10+ | Type-annotation syntax used in the codebase |
| ~2 GB free disk | Python packages, HF MiniLM cache, optional local `data/chroma/` |
| [Ollama](https://ollama.com/download) installed and running | Local LLM (no external API allowed per project spec) |
| Internet | **Always** needed for **`make install`** (PyPI wheels) and **first-time** `ollama pull` / MiniLM (~80 MB to `~/.cache/huggingface/`). **Wikipedia** is only touched when you run **`make ingest`** (or Streamlit ingest) against missing or forced-refetch docs—**skip those if `data/` is already complete** in the clone |

---

## Instructor quickstart (copy in order)

Assumes **[GNU Make](https://www.gnu.org/software/make/)** (included on macOS/Linux; Windows: Visual Studio Build Tools **or** MSYS2 **or** WSL — if you cannot use Make, follow the **`python -m` fallbacks** in each section).

This repository is intended to ship a **complete `data/` workspace** (`data/roster.json`, **`data/raw/`**, **`data/chunks/`**, **`data/chroma/`**, and **`data/store_manifest.json`**) so a grader avoids bulk Wikipedia ingestion and **`make build`**. Paths under `data/chroma/` are **binary persistence**—if Chroma errors on your machine, run **`make build`** once to rebuild the index from committed chunks (no Wikipedia needed).

1. Obtain the project: `git clone <URL>` **or** unzip the submission archive, then `cd` into the project root (`Makefile` + `requirements.txt` live here).
2. **`make install`** — creates `.venv/` and installs `requirements.txt` (this is usually the **longest** step on a clean machine: large wheels such as PyTorch via `sentence-transformers`).
3. [Run the local LLM](#run-the-local-llm-ollama): install Ollama, **`ollama pull llama3.2:3b`**, confirm **`curl http://127.0.0.1:11434/api/tags`** (Makefile does not manage Ollama).
4. **Indexing from Wikipedia (skip if `data/` is complete):** run **`make data`** only when raw JSON, chunk JSONL, or Chroma is missing, you wiped `data/`, or you changed `data/roster.json` and need a full refresh. That target runs **`make ingest`** then **`make chunk`** then **`make build`** and can take many minutes (API + embedding). See [Ingest data](#ingest-data).
5. **`make run-ui`** for Streamlit (**`make run-cli`** for CLI). The first chat query may still trigger a **one-time MiniLM download** into `~/.cache/huggingface/` (~80 MB) even when Chroma is present.
6. Try the [example queries](#example-queries).

| Target | Purpose |
|--------|---------|
| `make install` | Venv + `pip install -r requirements.txt` |
| `make ingest` | Fetch roster → `data/raw/` |
| `make chunk` | Raw → `data/chunks/` |
| `make build` | Chroma index from chunks (use after clone if shipped Chroma fails to open) |
| `make data` | `ingest` + `chunk` + `build` (long run; omit when full `data/` is in the repo) |
| `make probe` | Offline check that MiniLM loads |
| `make run-ui` / `make run-cli` | Streamlit / CLI |
| `make eval` | Golden-question harness (needs Ollama) |
| `make test` | Unit tests |

Override paths when needed: `make ingest ROSTER=path/to/roster.json` or `make chunk RAW_DIR=... CHUNKS_DIR=...`.

After `make install`, **offline mode** is fine for chat and eval **if** `data/chroma/` works and Hugging Face has already cached MiniLM on that machine (`make probe` / `make eval` use offline flags where applicable).

---

## Install dependencies

**Preferred (matches quickstart):**

```bash
make install
```

**Manual equivalent** (if you skip Make entirely):

```bash
# From the repo root (the folder that contains README.md and requirements.txt)
python3 -m venv .venv
source .venv/bin/activate         # Windows CMD: .venv\Scripts\activate.bat
                                # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The `.venv/` directory is intentionally ignored by git and should stay in the project root. The `Makefile` uses `.venv/bin/python` for all targets. The `probe` target sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so the already-downloaded MiniLM model is loaded from cache.

The first import of `sentence-transformers` downloads the MiniLM model (~80 MB) into `~/.cache/huggingface/`. Subsequent runs are offline.

**Windows paths:** Commands below use POSIX-style `.venv/bin/python`. On Windows after activating the venv, use `python -m …` instead, or invoke `.venv\Scripts\python.exe` (and `.venv\Scripts\streamlit.exe` for the UI).

---

## Run the local LLM (Ollama)

Install Ollama, then pull the model:

```bash
ollama pull llama3.2:3b
```

On macOS the daemon runs automatically once Ollama is installed. On Linux:

```bash
ollama serve
```

Verify it's reachable:

```bash
curl http://127.0.0.1:11434/api/tags
```

> The system uses Ollama only for **generation**. Embeddings run in-process via `sentence-transformers`. If you'd rather use Ollama for embeddings too, run `ollama pull nomic-embed-text` and pass `--encoder ollama` on the build step.

---

## Ingest data

Rebuild **raw Wikipedia JSON**, **chunk JSONL**, and the **Chroma** index—only when `data/` is incomplete, you changed **`data/roster.json`**, or you want fresh text from Wikipedia (`--force` / Streamlit Force refetch).

Three steps the Makefile chains: **fetch** → **chunk** → **embed + index**.

**With Make (preferred):**

```bash
make data
# or step-by-step:
make ingest
make chunk
make build
```

**Rebuild Chroma only (no Wikipedia):** if `data/raw/` and `data/chunks/` match the bundled submission but **`data/chroma/`** fails on your platform, **`make build`** alone is usually enough—fast compared to ingest, slow compared to skipping because it still walks every chunk embedding.

**Time:** **`make data` can take several minutes or longer** when Wikipedia is contacted for every roster entry and embeddings run for every chunk. With **`data/raw/`** already cloned and only **`make chunk` + `make build`** needed (or **`make build`** only), wall time drops sharply versus a cold ingest.

**Without Make** (same commands the Makefile runs):

```bash
.venv/bin/python -m ingest.wikipedia --roster data/roster.json --out data/raw
.venv/bin/python -m chunking.splitter --raw data/raw --out data/chunks
.venv/bin/python -m store.vector_store --build
```

All three steps are **idempotent** — re-running uses cached docs and upserts existing chunks. To force a refetch of one entity:

```bash
.venv/bin/python -m ingest.wikipedia --refresh "Albert Einstein"
```

The default roster is in [`data/roster.json`](data/roster.json) and contains the 20 + 20 minimum entities specified by the homework. Add more freely.

---

## Start the chat application

**Streamlit (recommended for the demo):**

```bash
make run-ui
```

Equivalent: `.venv/bin/python -m streamlit run app/localchat_rag.py`

Opens at <http://localhost:8501> with programmatic navigation (**localchat-rag** overview, **Chat**, **Ingestion**). Chat provides history, source citations, intent badge ("searched in: people / places / both"), and per-answer latency.

**CLI (no browser needed):**

```bash
make run-cli
```

Equivalent: `.venv/bin/python -m app.cli`

| Command | Effect |
|---------|--------|
| `:sources` | Toggle citation display |
| `:reset`   | Clear chat history (keeps the index) |
| `:stats`   | Print vector store stats |
| `:build`   | Rebuild the index from local chunks |
| `:quit`    | Exit |

---

## Example queries

These are the questions from the homework spec — the eval harness ([`agents/07_qa_eval.md`](agents/07_qa_eval.md)) verifies the system handles all of them.

### People
- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo.
- What is Frida Kahlo known for?

### Places
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What is Machu Picchu?
- What was the Colosseum used for?
- Where is Mount Everest?

### Mixed
- Which famous place is located in Turkey?
- Which person is associated with electricity?
- Compare Albert Einstein and Nikola Tesla.
- Compare the Eiffel Tower and the Statue of Liberty.

### Failure cases (must refuse)
- Who is the president of Mars?
- Tell me about a random unknown person John Doe.

For these the system must return `"I don't know based on the indexed data."` rather than fabricate an answer.

---

## Run evaluation

After the index is built and Ollama is running, run the golden-question harness:

```bash
make eval
```

The latest M7 report is [`eval/results/20260427T202149Z.md`](eval/results/20260427T202149Z.md): 20/20 cases passed, including both refusal cases. Total latency was p50 11167 ms / p95 18499 ms; retrieval was p50 46 ms / p95 123 ms, so local LLM generation is the bottleneck.

---

## Project layout

```
localchat-rag/
├── ingest/             # Agent 1 — Wikipedia REST client
├── chunking/           # Agent 2 — sentence-aware splitter
├── embedding/          # Agent 2 — local encoder (MiniLM default)
├── store/              # Agent 3 — Chroma persistence (one collection, type metadata)
├── retrieval/          # Agent 4 — query router + filtered search
├── generation/         # Agent 5 — Ollama client + grounded prompt
├── app/                # Agent 6 — Streamlit + CLI
├── eval/               # Agent 7 — golden-question harness
├── tests/              # per-area unit tests
├── agents/             # nine developer-role contracts
├── data/               # bundled: roster, raw, chunks, chroma — commit for submit; graders skip make data unless rebuild needed
├── AGENTS.md           # agent index, lifecycle, prompt template, conflict resolution
├── product_prd.md      # locked requirements
├── roadmap.md          # M0–M8 plan
├── recommendation.md   # production-deployment notes
├── demo_script.md      # 5-minute video walkthrough script
├── Makefile             # install, ingest, chunk, build, data, run-ui, run-cli, eval, test
└── requirements.txt
```

---

## Reset / clear

| Action | Command |
|--------|---------|
| Clear chat history (in-memory) | `:reset` in CLI, "Reset chat" in sidebar |
| Drop the vector store only | `.venv/bin/python -m store.vector_store --reset` |
| Wipe everything (raw + chunks + Chroma) | `rm -rf data/raw data/chunks data/chroma data/store_manifest.json` |

After a wipe, run **`make data`** (or follow the **`python -m` fallback** commands in this section).

---

## Documents at a glance

| File | What it is |
|------|------------|
| [`README.md`](README.md) | This file — install, run, query |
| [`AGENTS.md`](AGENTS.md) | Multi-agent role index, lifecycle, prompt template, conflict resolution |
| [`product_prd.md`](product_prd.md) | Authoritative requirements |
| [`roadmap.md`](roadmap.md) | Phased milestones |
| [`recommendation.md`](recommendation.md) | Production-deployment notes |
| [`agents/`](agents/) | One contract file per role |
