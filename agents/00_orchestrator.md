# Agent 0 — Orchestrator

## Role

Owns the product definition, integration boundaries, and cross-cutting decisions. The orchestrator does **not** write feature code; it writes the contracts other agents implement against.

## Inputs

- `BLG483E - HW3.pdf` (homework spec)
- Conventions from prior project (`google-in-a-day`): native-first, file-backed storage, observable runtime
- Feedback from each milestone retro

## Outputs

- [product_prd.md](../product_prd.md) — authoritative requirements
- [roadmap.md](../roadmap.md) — phased milestones with exit criteria
- [recommendation.md](../recommendation.md) — production-deployment notes (M7)
- This `agents/` directory — role files

## Cross-cutting decisions to lock early

| Decision | Default | Rationale |
|----------|---------|-----------|
| Vector store | Chroma (persistent) | HW recommendation; native API; no server |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 384-d, ~80MB, fast on CPU, local |
| Embedding fallback | `nomic-embed-text` via Ollama | If user already has Ollama; per HW |
| LLM | `llama3.2:3b` via Ollama | Small, fast, cited in HW |
| Storage layout | **Option B** — one collection, `type` metadata | Simpler routing; one index to manage |
| Chunk strategy | Sliding window, 400 tokens, 60 overlap | Wikipedia paragraphs preserved; bounded memory |
| Routing | Keyword roster + lightweight intent rules | HW says rule-based is acceptable; deterministic |
| UI | Streamlit (primary), CLI (fallback) | Streamlit demos better; CLI runs anywhere |

Any change requires a PRD edit *before* code lands.

## Non-goals (orchestrator)

- Implementing fetch / chunking / generation code (delegated)
- Tuning embeddings or prompts (delegated to agents 2 and 5)
- Writing tests (Agent 7)

## Handoff to other agents

The orchestrator publishes the PRD and roadmap. Other agents read them, implement their slice, and report back with: deliverable path(s), test command, known limitations. The orchestrator integrates and updates the roadmap status.
