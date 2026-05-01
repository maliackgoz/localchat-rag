"""Shared app wiring for the Streamlit UI and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from chunking.splitter import ChunkingReport, chunk_raw_documents
from generation import Answerer, OllamaClient
from generation.llm import DEFAULT_MODEL
from ingest.wikipedia import IngestReport, ingest_roster
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
DEFAULT_RAW_DIR = "data/raw"
EntityType = Literal["person", "place"]


@dataclass
class ChatRuntime:
    store: VectorStore
    retriever: Retriever
    answerer: Answerer


@dataclass(frozen=True)
class IngestionPipelineReport:
    ingest: IngestReport
    chunking: ChunkingReport
    index: dict[str, Any]


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


def run_ingestion_pipeline(
    runtime: ChatRuntime,
    *,
    roster_path: str = DEFAULT_ROSTER_PATH,
    raw_dir: str = DEFAULT_RAW_DIR,
    chunks_dir: str = DEFAULT_CHUNKS_DIR,
    force: bool = False,
) -> IngestionPipelineReport:
    ingest_report = ingest_roster(roster_path, raw_dir, force=force)
    chunk_report = chunk_raw_documents(raw_dir, chunks_dir)
    index_report = rebuild_index(runtime, chunks_dir=chunks_dir)
    return IngestionPipelineReport(ingest=ingest_report, chunking=chunk_report, index=index_report)


def add_entity_to_roster(
    entity_name: str,
    entity_type: EntityType,
    *,
    roster_path: str = DEFAULT_ROSTER_PATH,
) -> bool:
    normalized_name = _normalize_entity_name(entity_name)
    key = _roster_key(entity_type)
    other_key = "places" if key == "people" else "people"
    path = Path(roster_path)
    roster = _read_roster_payload(path)
    names = _required_roster_names(roster, key, path)
    other_names = _required_roster_names(roster, other_key, path)

    if _has_name(names, normalized_name):
        return False
    if _has_name(other_names, normalized_name):
        raise ValueError(f"{normalized_name!r} already exists under {other_key}")

    names.append(normalized_name)
    _write_roster_payload(path, roster)
    return True


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


def _normalize_entity_name(entity_name: str) -> str:
    normalized = " ".join(entity_name.split())
    if not normalized:
        raise ValueError("Entity name must not be empty")
    return normalized


def _roster_key(entity_type: EntityType) -> Literal["people", "places"]:
    if entity_type == "person":
        return "people"
    if entity_type == "place":
        return "places"
    raise ValueError(f"Unsupported entity type: {entity_type!r}")


def _read_roster_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _required_roster_names(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a {key!r} list")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path} has an invalid {key!r} item: {item!r}")
        value[index] = item.strip()
    return value


def _write_roster_payload(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _has_name(names: list[str], name: str) -> bool:
    folded = name.casefold()
    return any(existing.casefold() == folded for existing in names)
