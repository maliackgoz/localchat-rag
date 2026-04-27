# Agent 3 — Vector Store

## Role

Persist chunks and embeddings in a Chroma collection, expose typed query methods, and keep storage idempotent across re-ingest.

## Storage choice — Option B (single store + metadata)

One Chroma collection (`wikipedia_rag`) holds everything. Every chunk carries metadata:

```json
{
  "type": "person",        // or "place"
  "entity_name": "Albert Einstein",
  "wikipedia_url": "...",
  "position": 7
}
```

### Why Option B over Option A (two stores)

- **Routing is metadata, not infrastructure.** Filtering by `type` is a one-line `where` clause; spinning up two collections doubles the lifecycle code (init, persist, clear, migrate) for no retrieval-quality gain.
- **Mixed queries are free.** "Compare Eiffel Tower and Statue of Liberty" or "Which person is associated with electricity" can search across types without a union step.
- **Simpler clear/reset.** One `collection.delete()` resets everything.
- **Easier eval.** One ranking distribution to inspect, not two.

The trade-off: a buggy router can leak across types. Mitigation: the router is deterministic (Agent 4) and surfaces its decision in the response so failures are visible.

## Inputs

- Chunks + vectors from Agent 2

## Outputs

- `data/chroma/` — Chroma persistent client directory
- `data/store_manifest.json` — mapping `entity_name → list[chunk_id]` for idempotency

## Interfaces

```python
# store/vector_store.py
class VectorStore:
    def __init__(self, persist_dir: str, encoder: Encoder): ...
    def upsert_entity(self, entity_name: str, type_: str, chunks: list[Chunk]) -> None: ...
    def remove_entity(self, entity_name: str) -> None: ...
    def query(
        self,
        text: str,
        *,
        k: int = 6,
        type_filter: Literal["person", "place", "any"] = "any",
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]: ...
    def stats(self) -> dict: ...        # counts by type, total chunks, dim
    def reset(self) -> None: ...
```

CLI:
```bash
python -m store.vector_store --build         # ingest all chunks (idempotent)
python -m store.vector_store --stats
python -m store.vector_store --query "What did Marie Curie discover?" --type person
```

## Idempotency rule

`upsert_entity` deletes existing chunks for that entity before inserting new ones. Re-running build after re-ingest produces the same store regardless of order.

## Non-goals

- Hybrid BM25 + vector (could be added in M6 stretch)
- Sharding / multi-collection (the HW spec doesn't justify it)

## Done when

- `stats()` reports `>= 40` entities split across both types and dim matches encoder.
- `query("Eiffel Tower", type_filter="place", k=3)` returns 3 chunks all with `type=place` and `entity_name=Eiffel Tower` (or close neighbors).
- `reset()` followed by `--build` produces an identical `store_manifest.json` (modulo timestamps).
