# Agent 2 — Chunking + Embedding

## Role

Split each raw document into overlapping chunks, compute local embeddings, and emit a stream of `(chunk_id, text, metadata, vector)` tuples for the vector store.

## Inputs

- `data/raw/<type>/<slug>.json` (from Agent 1)
- Chunk config in [product_prd.md](../product_prd.md):
  - `chunk_tokens = 400`
  - `chunk_overlap = 60`
  - Token boundary: whitespace-split with sentence-aware soft preference (`. `, `? `, `! `, `\n\n`)

## Outputs

- `data/chunks/<type>/<slug>.jsonl` — one chunk per line:
  ```json
  {
    "chunk_id": "albert_einstein__0007",
    "entity_name": "Albert Einstein",
    "type": "person",
    "wikipedia_url": "...",
    "position": 7,
    "n_tokens": 397,
    "text": "..."
  }
  ```
- Vectors are streamed straight into the vector store by Agent 3 — Agent 2 does not persist vectors separately.

## Interfaces

```python
# chunking/splitter.py
def split_document(text: str, *, target_tokens: int = 400, overlap: int = 60) -> list[Chunk]: ...

# embedding/encoder.py
class Encoder:
    def encode(self, texts: list[str]) -> list[list[float]]: ...   # batched
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...

def get_encoder(backend: Literal["sentence_transformers", "ollama"] = "sentence_transformers") -> Encoder: ...
```

CLI:
```bash
python -m chunking.splitter --raw data/raw --out data/chunks
python -m embedding.encoder --probe   # prints model_id, dim, sample vector norm
```

## Chunking strategy

**Sliding window, sentence-aware soft boundary.**

1. Split text into sentences using a regex on `[.!?]\s+(?=[A-Z])` (cheap, no NLTK dep).
2. Greedily pack sentences until adding the next would exceed `target_tokens`.
3. When emitting the next chunk, back up by ~`overlap` tokens worth of sentences so context bleeds across chunks.
4. Prepend the entity name to every chunk's text in storage so tiny chunks are still retrievable by entity. e.g., `"Albert Einstein — <chunk text>"`.

Why this and not fixed-size byte chunks: Wikipedia paragraphs carry the bulk of the signal; cutting mid-sentence hurts retrieval and quoting in answers.

## Embedding choice

**Default:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, ~80MB, fast on CPU).

**Why default this, not Ollama nomic-embed-text:**
- One fewer process for evaluators to run (no Ollama needed for ingest).
- Stable quality on short factual queries.
- Trivial to swap via `get_encoder("ollama")` when Ollama is installed.

Both backends must produce L2-normalized vectors so the vector store can use cosine similarity uniformly.

## Non-goals

- Cross-encoder re-ranking (Agent 4 may add this later)
- Multi-vector retrieval (out of scope for MVP)
- Fine-tuning (out of scope)

## Done when

- All 40 entities have a `.jsonl` chunk file under `data/chunks/`.
- `Encoder.encode([...])` returns vectors of `dim` length, unit-normalized (norm ≈ 1.0 ± 1e-3).
- Probe script prints `model_id`, `dim`, average chunks/entity (sanity check, expect 5–15).
