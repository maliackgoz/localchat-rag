# Production Deployment Recommendations

> **Status:** stub. Final content lands in **M8** ([agents/08_docs.md](agents/08_docs.md)) once the system is end-to-end runnable and Agent 7 has produced real latency numbers. The bullets below are the topics M8 must cover.

---

## Where the local-only constraint breaks down at scale

The HW spec mandates a local, single-process system. That's correct for the assignment. For real-world deployment, the same architecture would not survive contact with:

- Multiple concurrent users (single Python process; Ollama serializes generation)
- A corpus larger than a few hundred documents (Chroma persistent client is fine for thousands; tens of thousands wants a server)
- Freshness requirements (no scheduled re-ingest; no change detection beyond manual `--force`)
- Authentication, rate limiting, audit logging
- Multi-tenant isolation

## Topics this document will cover (filled in M8)

### Embedding service
- Move from in-process MiniLM to a small encoder service (FastAPI + `sentence-transformers` or text-embeddings-inference)
- Batched `/embed` endpoint with caching by content hash
- Replication for throughput; pinning to CPU vs. GPU based on traffic shape

### Vector store
- Chroma → Qdrant / Weaviate / pgvector trade-offs at >100k chunks
- Index parameters (HNSW M, ef_construction) and the cost of getting them wrong
- Backup / restore strategy
- Multi-tenant collections vs. metadata partitioning

### LLM hosting
- Ollama for local dev; vLLM or TGI behind an internal endpoint for production
- Quantization trade-offs (Q4_K_M vs. Q8 vs. fp16) — quality vs. latency vs. VRAM
- Model selection: when llama3.2:3b is too small (multi-step reasoning, comparisons) and the upgrade path
- Streaming infrastructure: SSE vs. WebSocket vs. chunked HTTP

### Ingest pipeline
- Queue + worker (Redis / SQS / similar) instead of one synchronous CLI invocation
- Freshness vs. cost — re-ingest cadence, change detection via Wikipedia's `revid`
- Copyright + ToS considerations (Wikipedia content is CC BY-SA; downstream content may not be)
- Schema migrations when chunk size or metadata shape changes

### Observability
- Per-request logs with retrieval scores, intent decision, refusal flag
- Refusal-rate dashboard — both too-low (hallucination risk) and too-high (over-refusal) are bad
- p95 latency by stage (route / retrieve / generate)
- Drift detection: similarity score distributions over time

### Evaluation in CI
- Golden set as a deploy gate — no merge if regressions on critical questions
- Eval cost (LLM calls) vs. cadence — daily full sweep, per-PR smoke set
- A/B harness for swapping models or prompts

### Safety
- Prompt injection from Wikipedia content (a vandalized article injecting "ignore previous instructions") — context isolation, output filters
- PII handling for non-public corpora
- Output policy (refuse on medical / legal / dangerous queries even if answer is in context)
- Rate limiting per user and per IP

### Cost vs. quality dial
- Embedding model size: MiniLM (84 MB, fast) → mpnet-base (438 MB, better) → e5-large (1.3 GB, best)
- LLM size: 3B (this project) → 8B (default upgrade) → 70B (specialty)
- Re-ranker on top: cross-encoder adds ~100 ms but cuts hallucination on ambiguous queries
- Hybrid retrieval (BM25 + vector) catches lexical-match cases (rare names, dates, numbers) that pure vector misses

---

This document is owned by **Agent 8 (Docs)** and updated at M8 once Agents 1–7 have shipped real numbers. Until then, treat it as a checklist of topics to cover, not a final reference.
