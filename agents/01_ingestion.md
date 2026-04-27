# Agent 1 — Ingestion

## Role

Fetch Wikipedia pages for the required roster (20 people + 20 places minimum), extract clean article text, and persist raw documents locally in an idempotent, inspectable form.

## Inputs

- Roster file: `data/roster.json`
  ```json
  {
    "people": ["Albert Einstein", "Marie Curie", ...],
    "places": ["Eiffel Tower", "Great Wall of China", ...]
  }
  ```
- Wikipedia REST API (no scraping libraries; `urllib` is enough — same approach as the prior project's crawler)

## Outputs

- `data/raw/<type>/<slug>.json` — one file per entity:
  ```json
  {
    "entity_name": "Albert Einstein",
    "type": "person",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
    "fetched_at": "2026-04-27T20:30:00Z",
    "title": "Albert Einstein",
    "summary": "...",
    "text": "...full plain-text body..."
  }
  ```
- `data/raw/_manifest.json` — list of ingested entities with hash + fetch timestamp

## Interfaces

```python
# ingest/wikipedia.py
def fetch_entity(name: str, entity_type: Literal["person", "place"]) -> RawDocument: ...
def ingest_roster(roster_path: str, out_dir: str, *, force: bool = False) -> IngestReport: ...
```

CLI:
```bash
python -m ingest.wikipedia --roster data/roster.json --out data/raw
python -m ingest.wikipedia --refresh "Albert Einstein"
```

## Endpoints to use

- `GET https://en.wikipedia.org/api/rest_v1/page/summary/<title>` — short summary + canonical URL
- `GET https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=<title>&format=json` — full plain-text article

Set `User-Agent: localchat-rag/1.0 (educational; contact: <email>)`.

## Required behavior

- **Idempotent:** re-running with the same roster is a no-op unless `--force` is set.
- **Disambiguation guard:** if the API returns a disambiguation page, fail loudly — do not silently pick a wrong entity.
- **Polite:** sleep 100–200 ms between requests; respect HTTP errors with backoff.
- **Plain text only:** strip wiki markup, refs, and infobox markers from extracts.
- **Manifest hash:** SHA-256 of `text` so chunking can detect changes.

## Non-goals

- Chunking (Agent 2)
- Embedding (Agent 2)
- Multi-language ingest (English only, MVP)

## Done when

- All 40 minimum-roster entities are present under `data/raw/`.
- `_manifest.json` lists 40 entities with non-empty `text` and a SHA-256 hash.
- A second run prints "0 fetched, 40 cached" without hitting the network.
