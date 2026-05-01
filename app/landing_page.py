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
    st.info("Go to **Chat** to query the index or **Ingestion** to fetch articles and rebuild embeddings.")


render()
