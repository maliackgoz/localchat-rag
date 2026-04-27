# Production Deployment Recommendations

`localchat-rag` is intentionally a laptop-scale homework system: one Python process, local embeddings, local Ollama generation, local Chroma persistence, and a fixed Wikipedia roster. That design is appropriate for grading and demoing because it proves the full RAG loop without any hosted LLM API. A production version should keep the same behavioral contract, but split the single process into services with clear ownership, monitoring, and deployment gates.

## Where the Local-Only Constraint Breaks Down

The current system assumes a single user and a small, inspectable corpus. It has no authentication layer, request queue, tenant isolation, rate limit, or service boundary between retrieval and generation. Ollama is also not a production concurrency layer: long generations serialize or contend for local CPU/GPU resources, and the M7 eval showed generation dominates latency (`total` p50 11167 ms / p95 18499 ms; `retrieve` p50 46 ms / p95 123 ms).

For production, keep local-only as a development mode and introduce an internal deployment mode: private network endpoints for embedding and LLM inference, a server vector database, and an API/UI tier that can enforce auth, quotas, and observability.

## Embedding Service

The in-process `sentence-transformers/all-MiniLM-L6-v2` encoder is simple and fast enough for 1163 chunks, but production should move embeddings behind a small internal service. A batch `/embed` endpoint lets ingestion send many chunks at once and lets query traffic share a warm model instead of reloading it per process.

Recommended first step:

- Serve the same MiniLM model through FastAPI or a purpose-built embedding server.
- Batch ingestion embeddings and cache by `(model_id, text_sha256)`.
- Expose model id and vector dimension in health checks so index builds can fail fast on dimension mismatches.
- Scale replicas by CPU/GPU utilization and queue depth, not only request count.

Larger corpora can trade latency for recall by moving from MiniLM to stronger encoders such as `all-mpnet-base-v2` or an E5-family model. That change requires a full index rebuild because embedding dimensions and similarity spaces are not compatible.

## Vector Store Choice

Chroma persistent mode is a good local default, but a production service should use a server-backed vector store with backups, access controls, and operational tooling.

Reasonable upgrade paths:

- **Qdrant:** strong HNSW performance, filtering, snapshots, and straightforward ops for pure vector workloads.
- **Weaviate:** richer schema and module ecosystem, useful when the vector database owns more search semantics.
- **pgvector:** best when the corpus is already relational and operational simplicity matters more than maximum vector-search throughput.

The current single-collection metadata design should carry forward: keep `type`, `entity_name`, `wikipedia_url`, and `position` as filterable metadata, while heavier text and audit data live outside the vector index. At larger scale, add backup/restore tests and measure recall before changing HNSW parameters.

## LLM Hosting

Ollama is excellent for a local demo, but production should put generation behind an internal inference endpoint. vLLM or Text Generation Inference would provide continuous batching, GPU scheduling, streaming responses, and clearer deployment controls.

The current `llama3.2:3b` model is small enough for laptops. In production, choose the model by measured answer quality and latency:

- 3B class: low cost and acceptable for short factual answers.
- 8B class: better comparisons and instruction following, still feasible on a modest GPU.
- Larger models: reserve for difficult reasoning or higher-risk domains after eval justifies the cost.

Quantization is a product decision, not only an infra decision. Q4 models are cheaper and faster but may lose citation discipline; Q8 or fp16 improves quality at higher VRAM cost. Preserve the existing hard IDK shortcut before the LLM call, because it reduces hallucination and saves inference work.

## Ingest Pipeline

The CLI ingest is idempotent and easy to inspect, but production needs a queue and workers. Treat each entity or document revision as a job with explicit states: fetched, chunked, embedded, indexed, failed, and superseded.

Recommended design:

- Use a queue such as Redis, SQS, or a database-backed job table.
- Store source revision ids, fetch timestamps, content hashes, and chunking parameters.
- Re-index only changed documents when the upstream revision or chunking config changes.
- Keep old index versions available until the new build passes eval.
- Track Wikipedia's CC BY-SA attribution requirements and review copyright rules before indexing non-Wikipedia corpora.

Freshness is a cost trade-off. A daily or weekly crawl is enough for this roster; news-like corpora need change feeds, backpressure, and a rollback plan.

## Observability

Production RAG needs stage-level telemetry, not just answer logs. Every request should record:

- query text or a redacted hash, user/session id, route intent, matched roster entities
- retrieved chunk ids, similarity scores, and top similarity
- refusal flag and reason (`empty_context`, `below_min_sim`, `model_refusal`)
- latency by route, retrieve, and generate stage
- model id, embedding model id, prompt version, and index version

Dashboards should track p50/p95 latency, refusal rate, low-score retrieval rate, empty-index errors, and answer failures from the eval harness. Retrieval score distributions are especially useful drift signals: if top scores fall after a rebuild, the index or embedding model likely changed behavior.

## Eval in CI

The M7 golden set passed 20/20 with both failure cases covered, and that harness should become a deployment gate. In CI, run a fast smoke subset on every PR and the full golden set before release. A failed critical refusal case should block deployment even if the aggregate pass rate remains high.

The CI report should store:

- pass/fail by case id
- answer text and source entities
- latency percentiles by stage
- model, prompt, encoder, and index versions
- diff against the previous accepted run

For prompt, model, chunking, or retrieval-threshold changes, use the same golden set plus a small adversarial set for off-roster entities and prompt-injection text.

## Safety

Wikipedia content can still contain adversarial or vandalized text. Treat retrieved passages as untrusted data and keep them separated from system instructions in the prompt. The generator should never obey instructions found inside retrieved context.

Production safety additions:

- Add prompt-injection eval cases, including chunks that say to ignore system instructions.
- Keep the current grounded-answer requirement and hard IDK fallback.
- Redact or avoid logging sensitive user queries.
- Add auth, rate limits, and abuse monitoring before exposing the app outside a trusted classroom network.
- For private corpora, define PII retention and deletion rules before indexing.

## Cost and Quality Roadmap

The highest-return upgrades are retrieval-side because generation is already the bottleneck. Start with hybrid BM25 + vector retrieval and a lightweight cross-encoder re-ranker, then re-evaluate whether a larger LLM is still needed. Add query rewriting only after measuring failures from ambiguous or underspecified questions.

A conservative production path is:

1. Keep MiniLM and `llama3.2:3b`, but deploy them behind internal services.
2. Add request logs, eval-in-CI, and index versioning.
3. Add hybrid retrieval and re-ranking.
4. Upgrade to an 8B model only if the eval set shows answer-quality failures that retrieval cannot fix.
