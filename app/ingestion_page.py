"""Streamlit ingestion controls for localchat-rag."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime import (
    ChatRuntime,
    IngestionPipelineReport,
    add_entity_to_roster,
    create_runtime,
    run_ingestion_pipeline,
)
from generation.llm import DEFAULT_MODEL


def main() -> None:
    st.title("Ingestion")
    st.caption("Fetch Wikipedia pages, regenerate chunks, and upsert embeddings into Chroma.")

    try:
        runtime = _load_runtime(DEFAULT_MODEL)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    _render_last_result()
    _render_index_status(runtime)
    st.divider()
    _render_add_entity_form(runtime)
    st.divider()
    _render_refresh_controls(runtime)


@st.cache_resource(show_spinner="Loading local models and vector store...")
def _load_runtime(model: str) -> ChatRuntime:
    return create_runtime(llm_model=model)


def _render_index_status(runtime: ChatRuntime) -> None:
    st.subheader("Current Index")
    try:
        stats = runtime.store.stats()
    except Exception as exc:
        st.error(f"Could not read vector store stats: {exc}")
        return

    by_type = stats.get("by_type", {})
    people_chunks = by_type.get("person", 0) if isinstance(by_type, dict) else 0
    place_chunks = by_type.get("place", 0) if isinstance(by_type, dict) else 0
    left, middle, right = st.columns(3)
    left.metric("Entities", int(stats.get("entities", 0)))
    middle.metric("Total chunks", int(stats.get("total_chunks", 0)))
    right.metric("Embedding dim", stats.get("dim", "unknown"))
    st.caption(
        f"People chunks: {people_chunks} · Place chunks: {place_chunks} · "
        f"Embedding model: `{stats.get('model_id', 'unknown')}`"
    )


def _render_add_entity_form(runtime: ChatRuntime) -> None:
    st.subheader("Add Or Refresh One Entity")
    st.write("Add a Wikipedia article title to the roster, then run the full local indexing pipeline.")

    with st.form("add_entity_form"):
        entity_name = st.text_input("Wikipedia article title", placeholder="Alan Turing")
        entity_type = st.radio(
            "Entity type",
            ["person", "place"],
            format_func=lambda value: "Person" if value == "person" else "Place",
            horizontal=True,
        )
        force = st.checkbox("Force refetch all roster documents", value=False)
        submitted = st.form_submit_button("Add entity and rebuild index", use_container_width=True)

    if not submitted:
        return

    roster_changed = False
    try:
        roster_changed = add_entity_to_roster(entity_name, entity_type)
        action = "Added entity and rebuilt index" if roster_changed else "Entity already in roster; rebuilt index"
        _run_pipeline(runtime, force=force, action=action, roster_changed=roster_changed)
    except Exception as exc:
        if roster_changed:
            st.cache_resource.clear()
        st.error(f"Ingestion failed: {exc}")


def _render_refresh_controls(runtime: ChatRuntime) -> None:
    st.subheader("Refresh Existing Roster")
    st.write("Use this when `data/roster.json` already has the entities you want to index.")
    force = st.checkbox("Force refetch Wikipedia pages", value=False, key="refresh_force")
    if st.button("Run ingest, chunk, and rebuild index", use_container_width=True):
        try:
            _run_pipeline(runtime, force=force, action="Refreshed roster and rebuilt index", roster_changed=False)
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")


def _run_pipeline(
    runtime: ChatRuntime,
    *,
    force: bool,
    action: str,
    roster_changed: bool,
) -> None:
    with st.spinner("Running Wikipedia ingest, chunking, and Chroma upsert..."):
        report = run_ingestion_pipeline(runtime, force=force)
    st.session_state["ingestion_last_result"] = _result_payload(action, roster_changed, report)
    st.cache_resource.clear()
    st.rerun()


def _render_last_result() -> None:
    result = st.session_state.get("ingestion_last_result")
    if not isinstance(result, dict):
        return

    st.success(str(result.get("action", "Ingestion completed.")))
    st.write(
        f"Fetched {result.get('fetched', 0)} docs, reused {result.get('cached', 0)} cached docs, "
        f"wrote {result.get('chunks_written', 0)} chunks, and indexed {result.get('chunks_indexed', 0)} vectors."
    )
    if result.get("roster_changed"):
        st.info("The roster changed, so cached Streamlit runtime state was reloaded.")


def _result_payload(
    action: str,
    roster_changed: bool,
    report: IngestionPipelineReport,
) -> dict[str, Any]:
    return {
        "action": action,
        "roster_changed": roster_changed,
        "fetched": report.ingest.fetched,
        "cached": report.ingest.cached,
        "chunks_written": report.chunking.chunks,
        "entities_chunked": report.chunking.entities,
        "chunks_indexed": report.index.get("chunks_built", 0),
        "entities_indexed": report.index.get("entities_built", 0),
    }


main()
