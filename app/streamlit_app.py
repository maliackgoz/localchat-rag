"""Streamlit chat UI for localchat-rag."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import (
    ChatRuntime,
    create_runtime,
    elapsed_ms,
    footer_text,
    ordered_context_chunks,
    rebuild_index,
)
from generation.llm import DEFAULT_MODEL
from store.vector_store import RetrievedChunk


def main() -> None:
    st.set_page_config(page_title="localchat-rag", layout="wide")
    st.title("localchat-rag")
    st.caption("Local Wikipedia RAG over the project roster.")

    try:
        runtime = _load_runtime(DEFAULT_MODEL)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    _ensure_history()

    stats = _safe_stats(runtime)
    show_context = _render_sidebar(runtime, stats)

    if int(stats.get("total_chunks", 0)) == 0:
        _render_empty_index_controls(runtime)

    for turn in st.session_state["turns"]:
        _render_turn(turn, show_context)

    question = st.chat_input("Ask about a roster person or place")
    if question:
        _handle_question(runtime, question, show_context)


@st.cache_resource(show_spinner="Loading local models and vector store...")
def _load_runtime(model: str) -> ChatRuntime:
    return create_runtime(llm_model=model)


def _ensure_history() -> None:
    if "turns" not in st.session_state:
        st.session_state["turns"] = []


def _safe_stats(runtime: ChatRuntime) -> dict[str, Any]:
    try:
        return runtime.store.stats()
    except Exception as exc:
        st.sidebar.error(f"Could not read vector store stats: {exc}")
        return {"total_chunks": 0, "entities": 0, "by_type": {}, "model_id": "unknown"}


def _render_sidebar(runtime: ChatRuntime, stats: dict[str, Any]) -> bool:
    st.sidebar.header("Runtime")
    st.sidebar.write(f"LLM: `{runtime.answerer.client.model}`")
    st.sidebar.write(f"Embeddings: `{stats.get('model_id', 'unknown')}`")
    st.sidebar.metric("Total chunks", int(stats.get("total_chunks", 0)))

    if hasattr(runtime.answerer.client, "health") and not runtime.answerer.client.health():
        st.sidebar.warning(
            "Ollama is not reachable. Start Ollama and run "
            f"`ollama pull {runtime.answerer.client.model}` before asking questions."
        )

    show_context = st.sidebar.toggle("Show retrieved context", value=True)
    if st.sidebar.button("Reset chat", use_container_width=True):
        st.session_state["turns"] = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Index controls")
    confirm_clear = st.sidebar.checkbox("Confirm clear index")
    if st.sidebar.button("Clear index", disabled=not confirm_clear, use_container_width=True):
        runtime.store.reset()
        st.session_state["turns"] = []
        st.sidebar.success("Index cleared.")
        st.rerun()
    st.sidebar.caption("Clear index removes Chroma data only. Chat reset does not touch the index.")
    return show_context


def _render_empty_index_controls(runtime: ChatRuntime) -> None:
    st.warning("The vector index is empty, so questions will refuse until chunks are indexed.")
    if st.button("Build index from local chunks", use_container_width=False):
        with st.spinner("Embedding local chunks and rebuilding Chroma..."):
            report = rebuild_index(runtime)
        st.success(f"Indexed {report.get('chunks_built', 0)} chunks for {report.get('entities_built', 0)} entities.")
        st.rerun()
    st.caption("If local chunks are missing, run `python -m ingest.wikipedia`, `make chunk`, then build the index.")


def _handle_question(runtime: ChatRuntime, question: str, show_context: bool) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        started = perf_counter()
        try:
            retrieval = runtime.retriever.retrieve(question)
            final_chunk = None
            for answer_chunk in runtime.answerer.stream(retrieval):
                final_chunk = answer_chunk
                if answer_chunk.done:
                    break
                placeholder.markdown(f"{answer_chunk.text}▌")
            if final_chunk is None:
                raise RuntimeError("Answer stream ended without a final answer.")

            latency_ms = elapsed_ms(started)
            if final_chunk.refused:
                placeholder.warning(final_chunk.text)
            else:
                placeholder.markdown(final_chunk.text)

            turn = {
                "question": question,
                "answer": final_chunk.text,
                "chunks": retrieval.chunks,
                "intent": final_chunk.intent,
                "latency_ms": latency_ms,
                "model": final_chunk.model,
                "refused": final_chunk.refused,
            }
            _render_answer_details(turn, show_context)
            st.session_state["turns"].append(turn)
        except RuntimeError as exc:
            message = str(exc)
            placeholder.error(message)
            st.info("Start the Ollama daemon and pull the configured model, then ask again.")
            st.session_state["turns"].append({"question": question, "error": message})


def _render_turn(turn: dict[str, Any], show_context: bool) -> None:
    with st.chat_message("user"):
        st.markdown(str(turn["question"]))

    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(str(turn["error"]))
            return
        if turn.get("refused"):
            st.warning(str(turn["answer"]))
        else:
            st.markdown(str(turn["answer"]))
        _render_answer_details(turn, show_context)


def _render_answer_details(turn: dict[str, Any], show_context: bool) -> None:
    st.caption(footer_text(turn["intent"], int(turn["latency_ms"]), bool(turn["refused"])))
    chunks = turn.get("chunks", [])
    if not isinstance(chunks, list):
        chunks = []
    with st.expander("Sources used", expanded=show_context):
        if not chunks:
            st.write("No sources returned.")
            return
        for chunk in ordered_context_chunks(chunks):
            _render_source(chunk, show_context)


def _render_source(chunk: RetrievedChunk, show_context: bool) -> None:
    st.markdown(f"**{chunk.entity_name}** · position `{chunk.position}` · [{chunk.wikipedia_url}]({chunk.wikipedia_url})")
    if show_context:
        st.write(chunk.text)
    else:
        st.caption("Enable 'Show retrieved context' to display the chunk text.")


if __name__ == "__main__":
    main()
