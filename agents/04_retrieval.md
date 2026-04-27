# Agent 4 — Retrieval

## Role

Given a user query, decide whether it's about a person, a place, or both, then retrieve the top-k chunks from the vector store with the right filter.

## Inputs

- `VectorStore.query(...)` from Agent 3
- Roster from `data/roster.json` (drives keyword matching)
- Stop word list (small, English)

## Outputs

```python
@dataclass
class RetrievalResult:
    query: str
    intent: Literal["person", "place", "both"]
    matched_entities: list[str]      # roster entries detected in the query
    chunks: list[RetrievedChunk]     # ranked, deduped, with score
```

## Interfaces

```python
# retrieval/router.py
def classify_intent(query: str, roster: Roster) -> tuple[Intent, list[str]]: ...

# retrieval/retriever.py
class Retriever:
    def __init__(self, store: VectorStore, roster: Roster): ...
    def retrieve(self, query: str, *, k: int = 6) -> RetrievalResult: ...
```

## Routing rules (deterministic, in order)

1. **Direct entity match.** Substring match (case-insensitive, word-boundary) of any roster name in the query → that entity's `type` wins. Multiple matches across types → `both`.
2. **Keyword cues.** Place keywords (`where`, `located`, `country`, `mountain`, `river`, `built`, `landmark`, `city`) push to `place`. Person keywords (`who`, `born`, `discovered`, `invented`, `wrote`, `played`, `painted`, `compose`) push to `person`. Both sets present → `both`.
3. **Default.** No signal → `both` (broader recall, let generation guard against irrelevance).

The router records its decision so the UI can show "I searched in: people" — that single line debugged 80% of bad answers in similar systems.

## Retrieval strategy

- Search with `k * 2` candidates under the chosen filter.
- If `intent == both`, do two searches (`person`, `place`) with `k` each, merge by score, dedupe by `chunk_id`, keep top `k`.
- If `matched_entities` is non-empty, run an additional pass with `entity_filter=matched_entities` and merge — this guarantees that named entities are represented in context even when their chunks have lower raw similarity (e.g., one-word queries like "Tesla").
- If a query matches multiple entities and uses comparison language (`compare`, `difference`, `similar`, `versus`, etc.), add an entity-name overview search for each matched entity and prefer each entity's earliest qualifying chunk. This keeps comparisons balanced without hard-coding entity-specific metadata.
- Score: cosine similarity (Chroma returns distances; convert with `1 - distance`).
- Drop chunks with similarity below `min_sim = 0.25` (configurable) to reduce hallucination triggers.

## Non-goals

- Cross-encoder re-rank (M6 stretch — `cross-encoder/ms-marco-MiniLM-L-6-v2` if time)
- Query rewriting via LLM (M6 stretch)

## Done when

- Routing maps the homework's example questions to the expected intent (Agent 7's golden set covers this).
- `retrieve("Compare Eiffel Tower and Statue of Liberty")` returns chunks from **both** entities.
- `retrieve("Who is the president of Mars")` returns either no chunks above threshold or chunks the generator can correctly refuse on (verified by Agent 7).
