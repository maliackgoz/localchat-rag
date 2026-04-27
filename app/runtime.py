"""Shared app wiring for the Streamlit UI and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from generation import Answerer, OllamaClient
from generation.llm import DEFAULT_MODEL
from retrieval import DEFAULT_MIN_SIM, Retriever, load_roster
from retrieval.router import DEFAULT_ROSTER_PATH, Intent
from store.vector_store import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PERSIST_DIR,
    RetrievedChunk,
    VectorStore,
    build_store,
)

DEFAULT_K = 6


@dataclass
class ChatRuntime:
    store: VectorStore
    retriever: Retriever
    answerer: Answerer


def create_runtime(
    *,
    roster_path: str = DEFAULT_ROSTER_PATH,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    llm_model: str = DEFAULT_MODEL,
) -> ChatRuntime:
    try:
        store = VectorStore(persist_dir=persist_dir, manifest_path=manifest_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the local embedding/vector store. "
            "Install dependencies with `pip install -r requirements.txt` and make sure "
            "`sentence-transformers/all-MiniLM-L6-v2` is cached or network access is available once."
        ) from exc
    roster = load_roster(roster_path)
    return ChatRuntime(
        store=store,
        retriever=Retriever(store, roster, min_sim=DEFAULT_MIN_SIM),
        answerer=Answerer(OllamaClient(model=llm_model), min_sim=DEFAULT_MIN_SIM),
    )


def rebuild_index(runtime: ChatRuntime, *, chunks_dir: str = DEFAULT_CHUNKS_DIR) -> dict[str, Any]:
    return build_store(chunks_dir, runtime.store)


def intent_label(intent: Intent) -> str:
    if intent == "person":
        return "people"
    if intent == "place":
        return "places"
    return "both"


def footer_text(intent: Intent, latency_ms: int, refused: bool) -> str:
    suffix = " · refused" if refused else ""
    return f"searched in: {intent_label(intent)} · {latency_ms}ms{suffix}"


def elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def ordered_context_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=lambda chunk: (chunk.entity_name, chunk.position, chunk.chunk_id))


def stats_text(stats: dict[str, Any]) -> str:
    by_type = stats.get("by_type", {})
    people = by_type.get("person", 0) if isinstance(by_type, dict) else 0
    places = by_type.get("place", 0) if isinstance(by_type, dict) else 0
    return (
        f"collection: {stats.get('collection', 'unknown')}\n"
        f"total_chunks: {stats.get('total_chunks', 0)}\n"
        f"entities: {stats.get('entities', 0)}\n"
        f"people_chunks: {people}\n"
        f"place_chunks: {places}\n"
        f"embedding_model: {stats.get('model_id', 'unknown')}\n"
        f"dim: {stats.get('dim', 'unknown')}"
    )
