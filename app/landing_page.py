"""Landing page: overview and locked technical decisions."""

from __future__ import annotations

import streamlit as st

PIPELINE_DOT = """
digraph pipeline {
    rankdir=LR;
    graph [fontname="sans-serif"];
    node [shape=box style=rounded fontname="sans-serif"];
    subgraph cluster_ingest {
        label="Corpus build";
        labelloc=t;
        style=dashed;
        W [label="Wikipedia REST"];
        RAW [label="data/raw JSON"];
        CHK [label="data/chunks JSONL"];
        EMB [label="Local embeddings"];
        VDB [label="Chroma"];
        W -> RAW -> CHK -> EMB -> VDB;
    }
    subgraph cluster_qa {
        label="Ask";
        labelloc=t;
        style=dashed;
        Q [label="User question"];
        R [label="Retriever"];
        G [label="Ollama LLM"];
        A [label="Answer + sources"];
        Q -> R -> G -> A;
    }
    VDB -> R;
}
"""


OVERVIEW_MARKDOWN = """### System overview

**localchat-rag** answers questions about rostered Wikipedia people and places using **only** local
models and retrieved article text — no hosted LLM API.
"""

TECHNICAL_DECISIONS_MARKDOWN = """
### Technical decisions

| Topic | Decision |
|-------|-----------|
| **LLM** | `llama3.2:3b` via Ollama · temperature **0.1** · max tokens **512** |
| **Embeddings** | Default **`sentence-transformers/all-MiniLM-L6-v2`** (384-d); optional **`nomic-embed-text`** via Ollama (`--encoder ollama`) |
| **Chunk method** | Sentence-aware sliding window · target **400** whitespace tokens · **60** overlap · entity prefix on each chunk for retrieval stability |
| **Vector store** | **Chroma** · single collection `wikipedia_rag` · cosine distance · chunk metadata **`{type, entity_name, wikipedia_url, position}`** exactly |
| **Retrieval / routing** | Deterministic **`classify_intent`**: roster word-boundary match first, then person/place keyword cues · default **`both`** · filtered vector search (**`k=6`**), entity-pinned search when roster names appear · comparison-shaped queries blend entity overviews **`min_sim=0.25`** (cosine) drops weak matches · generation refuses when retrieval is empty or below threshold (**no LLM world knowledge**) |
"""


TRADEOFFS_LIMITATIONS_MARKDOWN = """
### Tradeoffs and limitations

- **Coverage** · Only the **roster** plus indexed article text can ground answers · no live Wikipedia or web · weak or off-roster matches yield refusals (`I don't know ...`).

- **Chunking** · **Sentence boundaries** use a punctuation + capital heuristic, not a linguistic segmenter · long sentences are forced into token windows · overlap duplicates some text.

- **Routing** · **Rule-based** roster + keyword routing has blind spots · sloppy or implicit wording can search the wrong subset or default to **both** with no entity pin.

- **Retrieval** · **MiniLM** trades maximum quality for local speed and size · **`min_sim`** rejects borderline chunks and can miss paraphrases · **k=6** caps context breadth.

- **LLM** · **3B** local model favors homework hardware over frontier reasoning · **max tokens** constrains long answers.

- **Ingest** · **Wikipedia APIs** can rate-limit or fail on bulk refetch · ingest is the only stage that needs network for article text.

- **Shipped vector store** · Prebuilt **`data/chroma`** can misbehave across OS/Python/Chroma bumps; **`make build`** rebuilds from committed chunks without refetch.
"""


POSSIBLE_IMPROVEMENTS_MARKDOWN = """
### Possible improvements

1. **Hybrid + reranking** · Add **BM25** (or sparse index) fused with vectors, or a **cross-encoder** after retrieving a larger candidate set to trim to the best **`k`** chunks for the LLM.

2. **Encoders** · Larger or domain-tuned embeddings, optional **quantize** paths, telemetry on latency versus hit rate.

3. **Structure-aware chunking** · Split on wiki headings/sections, smarter sentence segmentation, variable window sizes by article type.

4. **Routing** · Embedding-based intent, richer **aliases** from Wikipedia redirects/disambiguation pages (still guarded against wrong articles).
"""


def render() -> None:
    st.title("localchat-rag")
    st.caption(
        "Local Wikipedia RAG: ingest roster articles into Chroma, retrieve with deterministic routing, answer with Ollama using only retrieved chunks."
    )
    st.markdown(OVERVIEW_MARKDOWN)
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.caption("Pipeline overview")
    st.markdown(
        "**Boundaries.** Embeddings + generation stay on-machine. Outbound Wikipedia HTTP calls occur only during ingest (not during chat)."
    )
    st.markdown(TECHNICAL_DECISIONS_MARKDOWN)
    st.markdown(TRADEOFFS_LIMITATIONS_MARKDOWN)
    st.markdown(POSSIBLE_IMPROVEMENTS_MARKDOWN)
    st.info("Go to **Chat** to query the index or **Ingestion** to fetch articles and rebuild embeddings.")


render()
