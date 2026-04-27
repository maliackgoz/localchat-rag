# Product PRD — localchat-rag

Authoritative requirements for a local Wikipedia RAG assistant. Agents implement against this document. Changes to this document are routed through the Orchestrator (Agent 0).

## Goal

A ChatGPT-style assistant that answers questions about a fixed roster of famous people and places using only locally-running components. The system runs end-to-end on a laptop with no external LLM API.

## Hard requirements (from HW spec)

1. **Local only.** No external LLM API. Embedding and generation run on the user's machine.
2. **Roster.** At minimum the 20 + 20 entities listed in the HW PDF; more are allowed.
3. **Ingest from Wikipedia.** Persist raw documents locally.
4. **Chunk** documents (own size + strategy).
5. **Embed locally** (sentence-transformers or Ollama).
6. **Vector store** (Chroma).
7. **Routing** to person / place / both. Rule-based is acceptable.
8. **Generate grounded answers** with `"I don't know"` fallback when context is insufficient.
9. **Chat interface** (Streamlit or CLI) with ask / answer / show-context / reset.
10. **Documentation:** `README.md`, `product_prd.md`, `recommendation.md`.
11. **Demo:** 5-minute Loom or unlisted YouTube video.

## Locked design decisions

| Concern | Decision | Why |
|---------|----------|-----|
| Storage layout | Option B — single Chroma collection, `type` and `entity_name` metadata | Routing is metadata, not infrastructure; mixed queries are free; one reset path |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (default) | 384-d, ~80MB, fast on CPU; one fewer process for the grader |
| Embedding fallback | `nomic-embed-text` via Ollama | Available with `--encoder ollama` flag |
| Chunk strategy | Sentence-aware sliding window | Wikipedia paragraphs carry the signal; preserves quotability |
| Chunk size | 400 tokens, 60 overlap | Fits MiniLM's 256-token guidance with margin; overlap keeps cross-paragraph facts |
| LLM | `llama3.2:3b` via Ollama | Listed in HW; small enough for CPU; quality good enough for short factual answers |
| LLM temperature | 0.1 | Deterministic-ish answers, fewer hallucinations |
| LLM max tokens | 512 | Enough for compare questions; bounded cost |
| Routing | Roster substring + keyword cues, deterministic | HW says rule-based is acceptable; debuggable |
| `min_sim` threshold | 0.25 (cosine) | Cuts hallucination triggers; tuned via Agent 7 |
| `k` (chunks per query) | 6 | Empirically reasonable for 3B model context budget |
| UI | Streamlit primary, CLI fallback | Streamlit for the demo, CLI for portability |
| Streaming | Yes | Better UX, satisfies HW optional extension |

A change to any row above is a PRD edit owned by the Orchestrator.

## Storage schema

```
data/
├── roster.json                       # owned by Orchestrator
├── raw/                              # owned by Agent 1
│   ├── _manifest.json
│   ├── people/<slug>.json
│   └── places/<slug>.json
├── chunks/                           # owned by Agent 2
│   ├── people/<slug>.jsonl
│   └── places/<slug>.jsonl
├── chroma/                           # owned by Agent 3 (Chroma persist dir)
└── store_manifest.json               # owned by Agent 3
```

## Chunk record

```json
{
  "chunk_id": "<entity_slug>__<position:04>",
  "entity_name": "Albert Einstein",
  "type": "person",
  "wikipedia_url": "...",
  "position": 7,
  "n_tokens": 397,
  "text": "Albert Einstein — <chunk text>"
}
```

## Vector store metadata (Chroma)

`{ type, entity_name, wikipedia_url, position }` — exactly these four keys, no more. Anything else lives in the chunk file on disk to keep the store light.

## Generation contract

- Input: `RetrievalResult` (query, intent, matched entities, ranked chunks).
- If `len(chunks) == 0` or top similarity `< min_sim`: return IDK without calling the LLM.
- Otherwise: render the locked prompt template (see [agents/05_generation.md](agents/05_generation.md)), call Ollama, return `Answer`.
- `Answer.refused = answer.text.lower().startswith("i don't know")`.

## Roster file shape

```json
{
  "people": ["Albert Einstein", "Marie Curie", "..."],
  "places": ["Eiffel Tower", "Great Wall of China", "..."]
}
```

Entity name strings are passed straight to the Wikipedia API. Disambiguation responses fail loudly (Agent 1).

## Failure modes the system must handle

| Case | Expected behavior |
|------|-------------------|
| No retrieved context | IDK refusal, no LLM call |
| Top similarity below threshold | IDK refusal |
| Off-roster entity ("John Doe", "president of Mars") | IDK refusal |
| Disambiguation page during ingest | Hard error with a clear message |
| Ollama not running | UI shows actionable error; ingest / search still work |
| Cleared index | UI offers a "Run ingest" button; `query` returns no results gracefully |

## Out of scope (MVP)

- Multi-turn memory injected back into retrieval
- Auth / multi-user
- Languages other than English
- Updating Wikipedia content on a schedule
- Server-mode Chroma

## Open questions

- Should `position` in chunk metadata index from 0 or 1? (Decision: 0, matches Python.)
- Should refusal be a different LLM prompt or a hard skip? (Decision: hard skip; cheaper and more reliable.)
- Should comparison queries fan-out to entity-pinned retrieval per side? (Decision: yes, when ≥2 roster entities are detected.)

This file is the single source of truth. Update it before code changes that affect the contract.
