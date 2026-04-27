"""Filtered similarity retrieval built on the Agent 3 vector store."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from retrieval.router import Intent, Roster, classify_intent
from store.vector_store import RetrievedChunk, TypeFilter, VectorStore


DEFAULT_K = 6
DEFAULT_MIN_SIM = 0.25
COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "contrast",
    "difference",
    "differences",
    "different",
    "similar",
    "similarities",
    "versus",
    "vs",
}


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    intent: Intent
    matched_entities: list[str]
    chunks: list[RetrievedChunk]


class QueryableStore(Protocol):
    def query(
        self,
        text: str,
        *,
        k: int = DEFAULT_K,
        type_filter: TypeFilter = "any",
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return vector-store results ordered by similarity."""


class Retriever:
    def __init__(self, store: VectorStore | QueryableStore, roster: Roster, *, min_sim: float = DEFAULT_MIN_SIM) -> None:
        self.store = store
        self.roster = roster
        self.min_sim = min_sim

    def retrieve(self, query: str, *, k: int = DEFAULT_K) -> RetrievalResult:
        if k < 1:
            raise ValueError("k must be positive")

        intent, matched_entities = classify_intent(query, self.roster)
        if not query.strip():
            return RetrievalResult(query=query, intent=intent, matched_entities=matched_entities, chunks=[])

        comparison_query = _needs_entity_overviews(query, matched_entities)
        candidates = self._primary_search(query, intent, k)
        if matched_entities:
            candidates.extend(self._entity_pinned_search(query, matched_entities, k))
        if comparison_query:
            candidates.extend(self._entity_overview_search(matched_entities))

        chunks = _rank_chunks(candidates, k, self.min_sim, matched_entities, prefer_early_position=comparison_query)
        return RetrievalResult(query=query, intent=intent, matched_entities=matched_entities, chunks=chunks)

    def _primary_search(self, query: str, intent: Intent, k: int) -> list[RetrievedChunk]:
        if intent == "both":
            return [
                *self.store.query(query, k=k, type_filter="person"),
                *self.store.query(query, k=k, type_filter="place"),
            ]
        return self.store.query(query, k=k * 2, type_filter=intent)

    def _entity_pinned_search(self, query: str, matched_entities: list[str], k: int) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for entity_name in matched_entities:
            chunks.extend(
                self.store.query(
                    query,
                    k=k,
                    type_filter=self.roster.entity_type(entity_name),
                    entity_filter=[entity_name],
                )
            )
        return chunks

    def _entity_overview_search(self, matched_entities: list[str]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for entity_name in matched_entities:
            chunks.extend(
                self.store.query(
                    entity_name,
                    k=2,
                    type_filter=self.roster.entity_type(entity_name),
                    entity_filter=[entity_name],
                )
            )
        return chunks


def _rank_chunks(
    chunks: list[RetrievedChunk],
    k: int,
    min_sim: float,
    preferred_entities: list[str],
    *,
    prefer_early_position: bool = False,
) -> list[RetrievedChunk]:
    best_by_id: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        if chunk.similarity < min_sim:
            continue
        existing = best_by_id.get(chunk.chunk_id)
        if existing is None or chunk.similarity > existing.similarity:
            best_by_id[chunk.chunk_id] = chunk

    ranked = sorted(best_by_id.values(), key=_rank_key)
    if not preferred_entities:
        return ranked[:k]

    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    for entity_name in preferred_entities:
        entity_chunk = _preferred_entity_chunk(ranked, entity_name, selected_ids, prefer_early_position)
        if entity_chunk is not None:
            selected.append(entity_chunk)
            selected_ids.add(entity_chunk.chunk_id)
        if len(selected) == k:
            return sorted(selected, key=_rank_key)

    for chunk in ranked:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        if len(selected) == k:
            break
    return sorted(selected, key=_rank_key)


def _rank_key(chunk: RetrievedChunk) -> tuple[float, str, int, str]:
    return (-chunk.similarity, chunk.entity_name, chunk.position, chunk.chunk_id)


def _preferred_entity_chunk(
    chunks: list[RetrievedChunk],
    entity_name: str,
    selected_ids: set[str],
    prefer_early_position: bool,
) -> RetrievedChunk | None:
    candidates = [chunk for chunk in chunks if chunk.entity_name == entity_name and chunk.chunk_id not in selected_ids]
    if not candidates:
        return None
    if prefer_early_position:
        return sorted(candidates, key=lambda chunk: (chunk.position, -chunk.similarity, chunk.chunk_id))[0]
    return candidates[0]


def _needs_entity_overviews(query: str, matched_entities: list[str]) -> bool:
    if len(matched_entities) < 2:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    return bool(tokens & COMPARISON_KEYWORDS)
