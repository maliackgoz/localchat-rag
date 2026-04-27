# Agent 8 — Docs

## Role

Author and maintain the documentation the homework requires, plus a demo script the recording can follow line-by-line.

## Inputs

- Locked decisions from [agents/00_orchestrator.md](00_orchestrator.md) and [product_prd.md](../product_prd.md)
- Real numbers from Agent 7 (latency, eval pass rate)

## Outputs

| File | Purpose |
|------|---------|
| `README.md` | Install, run, ingest, query, troubleshooting |
| `product_prd.md` | Authoritative requirements (owned with Orchestrator) |
| `recommendation.md` | Production-deployment notes |
| `demo_script.md` | Line-by-line script for the 5-minute video |

## README contents (HW-required)

1. **What this is** — one paragraph, name-drop "local-only RAG over Wikipedia".
2. **Prerequisites** — Python 3.10+, Ollama, ~2GB disk.
3. **Install** — `pip install -r requirements.txt`.
4. **Run the local model** — `ollama pull llama3.2:3b` and `ollama serve` (or "ollama runs as a daemon on macOS").
5. **Ingest data** — `python -m ingest.wikipedia --roster data/roster.json --out data/raw` then `python -m store.vector_store --build`.
6. **Start the application** — Streamlit command; CLI alternative.
7. **Example queries** — copy from HW + a couple of stretch examples.
8. **Project layout** — small table mirroring `AGENTS.md`.
9. **Reset / clear** — how to wipe data and rebuild.
10. **Demo video link** — added at the end.

## recommendation.md contents

Production deployment notes covering:

- **Where the local-only constraint breaks down at scale** (single-process, single-user, no auth).
- **Embedding service** (separate process or replicas; batch endpoint).
- **Vector store choice** at production scale (Chroma → Qdrant / Weaviate / pgvector trade-offs).
- **LLM hosting** (vLLM or TGI behind an internal endpoint; quantization trade-offs).
- **Ingest pipeline** (queue + worker; freshness vs. cost; copyright).
- **Observability** (request logs with retrieval scores, refusal rate, p95 latency).
- **Eval in CI** (golden set as a gate before deploys).
- **Safety** (prompt injection from Wikipedia content; PII handling for non-public corpora).

## Demo script (5 minutes, target beats)

| Time | Beat | What's on screen |
|------|------|------------------|
| 0:00–0:30 | System overview slide | One diagram: ingest → chunk → embed → store → retrieve → generate |
| 0:30–1:30 | Live ingestion | Run the ingest + build commands; show `data/` populating |
| 1:30–3:30 | Live Q&A | 3 person, 2 place, 1 comparison, 1 failure; show sources and intent |
| 3:30–4:15 | Technical decisions | Why Option B, why MiniLM, why llama3.2:3b, why streaming |
| 4:15–4:45 | Trade-offs and limitations | Refusal latency, single-process, no infobox parsing |
| 4:45–5:00 | Possible improvements | Re-rank, hybrid BM25, Q-rewriting, multi-LLM compare |

## Non-goals

- Marketing copy / screenshots beyond what the README needs
- Per-agent documentation duplication — agents own their own files; the README links to `AGENTS.md`

## Done when

- `README.md` is complete enough that an instructor can run the project from scratch with no extra guidance.
- `recommendation.md` covers the eight bullets above.
- `demo_script.md` is rehearsed against a real run and stays under 5 minutes.
