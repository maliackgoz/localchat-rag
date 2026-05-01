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
| Python 3.10+ | Type-annotation syntax used in the codebase |
| ~2 GB free disk | sentence-transformers model + Chroma DB + raw Wikipedia text |
| [Ollama](https://ollama.com/download) installed and running | Local LLM (no external API allowed per project spec) |
| Internet (one-time) | Pulling the LLM model and the Wikipedia corpus |

---

## Install dependencies

```bash
git clone <this repo>
cd localchat-rag

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The `.venv/` directory is intentionally ignored by git and should stay in the project root. After the one-time install, use either the activated `python` command or the explicit `.venv/bin/python` path; the project `Makefile` uses `.venv/bin/python` so dependencies are reused instead of reinstalled. Its `probe` target also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so the already-downloaded MiniLM model is loaded from cache.

The first import of `sentence-transformers` downloads the MiniLM model (~80 MB) into `~/.cache/huggingface/`. Subsequent runs are offline.

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

Three steps: **fetch** Wikipedia articles for the roster, **chunk** them, then **embed + index** them.

```bash
# 1. Pull Wikipedia articles for the roster (20 people + 20 places)
.venv/bin/python -m ingest.wikipedia --roster data/roster.json --out data/raw

# 2. Chunk the raw articles
make chunk

# 3. Embed and write to Chroma
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
.venv/bin/streamlit run app/localchat_rag.py
```

Opens at <http://localhost:8501> with programmatic navigation (**localchat-rag** overview, **Chat**, **Ingestion**). Chat provides history, source citations, intent badge ("searched in: people / places / both"), and per-answer latency.

**CLI (no browser needed):**

```bash
.venv/bin/python -m app.cli
```

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
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m eval.run_eval --golden eval/golden.jsonl --report eval/results/
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
├── data/               # raw docs, chunks, Chroma DB, roster
├── AGENTS.md           # agent index, lifecycle, prompt template, conflict resolution
├── product_prd.md      # locked requirements
├── roadmap.md          # M0–M8 plan
├── recommendation.md   # production-deployment notes
├── demo_script.md      # 5-minute video walkthrough script
└── requirements.txt
```

---

## Reset / clear

| Action | Command |
|--------|---------|
| Clear chat history (in-memory) | `:reset` in CLI, "Reset chat" in sidebar |
| Drop the vector store only | `.venv/bin/python -m store.vector_store --reset` |
| Wipe everything (raw + chunks + Chroma) | `rm -rf data/raw data/chunks data/chroma data/store_manifest.json` |

After a wipe, re-run `ingest.wikipedia`, `chunking.splitter`, then `store.vector_store --build`.

---

## Troubleshooting

- **`Connection refused` on port 11434** — Ollama isn't running. Start it (`ollama serve` on Linux; relaunch the desktop app on macOS / Windows).
- **First query is slow (~5–15 s)** — the encoder loads lazily on first use. Subsequent queries are sub-second on the embedding side; total latency is dominated by the LLM.
- **`Disambiguation page` error during ingest** — a roster entry mapped to a Wikipedia disambiguation page. Edit `data/roster.json` to use the canonical title (e.g., `"Wolfgang Amadeus Mozart"` instead of `"Mozart"`).
- **Streamlit shows stale data after re-ingest** — refresh the page; the encoder + store are reloaded.
- **`ImportError: ...torch` on install** — pip is picking the wrong PyTorch wheel for your platform. See <https://pytorch.org/get-started/locally/> for the right index URL, or upgrade pip.

---

## Implementation status

This repo is implemented through **M8 docs** except for the external demo-video URL. The latest integrated eval passed 20/20 golden cases; track milestone status and verification notes in [`roadmap.md`](roadmap.md).

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
