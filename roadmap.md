# Roadmap

Phased plan from empty repo to a runnable, demo-ready local Wikipedia RAG. Each milestone has a single owner, concrete deliverables, and an exit criterion. The Orchestrator updates `Status` as work lands.

Legend: ☐ pending · ◐ in progress · ☑ done

---

## M0 — Setup and contracts (Orchestrator)

**Owner:** Agent 0 · **Status:** ☑

- ☑ `AGENTS.md` and `agents/*.md` role files
- ☑ `roadmap.md` (this file)
- ☑ `product_prd.md` with locked decisions
- ☑ `requirements.txt` (chromadb, sentence-transformers, streamlit, ollama)
- ☑ `data/roster.json` with the 20 + 20 minimum entities
- ☑ Repo skeleton: `ingest/`, `chunking/`, `embedding/`, `store/`, `retrieval/`, `generation/`, `app/`, `eval/`, `tests/`

**Exit:** every later milestone has unambiguous interfaces to implement against; `pip install -r requirements.txt` succeeds in a clean venv.

---

## M1 — Wikipedia ingestion (Agent 1)

**Owner:** Agent 1 · **Status:** ☑

- ☑ `ingest/wikipedia.py` — `fetch_entity`, `ingest_roster`
- ☑ `data/raw/<type>/<slug>.json` for all 40 minimum entities
- ☑ `data/raw/_manifest.json` with SHA-256 hashes
- ☑ Disambiguation guard test (e.g., requesting "Mercury" without a hint should fail loudly)
- ☑ Polite throttling (100–200 ms between requests)

**Exit:** `python -m ingest.wikipedia --roster data/roster.json --out data/raw` writes 40 non-empty docs; rerun is a no-op.

---

## M2 — Chunking + embedding (Agent 2)

**Owner:** Agent 2 · **Status:** ☐

- ☐ `chunking/splitter.py` — sentence-aware sliding window (400 / 60)
- ☐ `embedding/encoder.py` — `Encoder` interface, `sentence_transformers` backend
- ☐ Ollama backend stub (`get_encoder("ollama")`) — minimal, behind a try/except for missing daemon
- ☐ `data/chunks/<type>/<slug>.jsonl` for all 40 entities
- ☐ Probe script printing model id, dim, vector norm

**Exit:** chunks exist, embeddings unit-normalized, average chunks per entity in 5–15.

---

## M3 — Vector store (Agent 3)

**Owner:** Agent 3 · **Status:** ☐

- ☐ `store/vector_store.py` — `VectorStore` class with `upsert_entity`, `query`, `remove_entity`, `reset`, `stats`
- ☐ Persistent Chroma directory at `data/chroma/`
- ☐ `data/store_manifest.json` for idempotency
- ☐ CLI: `--build`, `--stats`, `--query`, `--type`, `--reset`

**Exit:** `--build` is idempotent, `stats()` shows ≥40 entities; filtered query returns only the requested type.

---

## M4 — Retrieval (Agent 4)

**Owner:** Agent 4 · **Status:** ☐

- ☐ `retrieval/router.py` — `classify_intent`
- ☐ `retrieval/retriever.py` — `Retriever` with `retrieve(query, k)`
- ☐ Two-pass merge for `intent=both`
- ☐ Entity-pinned search when roster names are in the query
- ☐ `min_sim` threshold to drop noise

**Exit:** routing matches expected intent on the HW examples; entity-pinned queries surface the right entity even for one-word queries.

---

## M5 — Generation (Agent 5)

**Owner:** Agent 5 · **Status:** ☐

- ☐ `generation/llm.py` — `OllamaClient` (sync + streaming)
- ☐ `generation/answerer.py` — `Answerer` with grounded prompt and IDK shortcut
- ☐ Refusal detection
- ☐ Source list in `Answer` payload
- ☐ Optional: parallel comparator for two LLMs (HW stretch)

**Exit:** `answer()` returns IDK for empty / low-similarity retrieval without calling Ollama; example questions produce grounded answers citing the right entity.

---

## M6 — Chat UI (Agent 6)

**Owner:** Agent 6 · **Status:** ☐

- ☐ `app/streamlit_app.py` with sidebar, chat history, sources expander, intent badge, latency
- ☐ `app/cli.py` REPL with `:sources`, `:reset`, `:stats`, `:quit`
- ☐ "Reset chat" and (separately) "Clear index" controls
- ☐ Streaming token rendering

**Exit:** `streamlit run app/streamlit_app.py` and `python -m app.cli` both answer all 14 example questions and refuse the 2 failure cases.

---

## M7 — QA, eval, latency (Agent 7)

**Owner:** Agent 7 · **Status:** ☐

- ☐ `eval/golden.jsonl` with ≥20 cases (16 from HW + 4 stretch)
- ☐ `eval/run_eval.py` producing JSON + Markdown reports
- ☐ p50 / p95 latency report
- ☐ ≥18 / 20 passing including both failure cases

**Exit:** eval can run from a clean checkout (after M3 build) and writes a report under `eval/results/`.

---

## M8 — Docs, recommendation, demo (Agent 8)

**Owner:** Agent 8 · **Status:** ☐

- ☐ `README.md` complete (install → ingest → run → query)
- ☐ `recommendation.md` covering production trade-offs
- ☐ `demo_script.md` rehearsed under 5 minutes
- ☐ Demo video recorded (Loom or unlisted YouTube), link added to README

**Exit:** instructor can run the project from README alone; demo video is linked.

---

## Stretch (post-MVP)

These are optional and only attempted if M0–M8 are clean:

- ☐ Cross-encoder re-rank in Agent 4 (`ms-marco-MiniLM-L-6-v2`)
- ☐ Hybrid BM25 + vector retrieval (reuse the Project 1 indexer for BM25)
- ☐ Side-by-side LLM compare (llama3.2:3b vs phi3) in Agent 5
- ☐ Response cache keyed on `(query, retrieval_hash)`
- ☐ Persistent chat history under `data/chat_log.jsonl`
- ☐ Query-rewriting for short / ambiguous queries

---

## Dependency order

```
M0 ── M1 ── M2 ── M3 ── M4 ── M5 ── M6 ── M7 ── M8
                              │
                              └── (Stretch items branch off here)
```

M7 (eval) integrates everything from M3 onward — running it is the gate for declaring an end-to-end build healthy.
