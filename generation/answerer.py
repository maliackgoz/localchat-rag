"""Grounded answer construction from retrieved Wikipedia chunks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from generation.llm import OllamaClient
from retrieval.retriever import DEFAULT_MIN_SIM, RetrievalResult
from retrieval.router import Intent
from store.vector_store import RetrievedChunk


IDK_TEXT = "I don't know based on the indexed data."


@dataclass(frozen=True)
class Source:
    entity_name: str
    wikipedia_url: str
    chunk_position: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    intent: Intent
    latency_ms: int
    model: str
    refused: bool


@dataclass(frozen=True)
class AnswerChunk:
    delta: str
    text: str
    sources: list[Source]
    intent: Intent
    latency_ms: int
    model: str
    refused: bool
    done: bool = False


class GenerationClient(Protocol):
    model: str

    def generate(self, prompt: str, *, stream: bool = False) -> Iterator[str] | str:
        """Return generated text or stream token deltas."""


class Answerer:
    def __init__(self, client: OllamaClient | GenerationClient, *, min_sim: float = DEFAULT_MIN_SIM) -> None:
        self.client = client
        self.min_sim = min_sim

    def answer(self, retrieval: RetrievalResult) -> Answer:
        started = perf_counter()
        if _requires_idk_shortcut(retrieval, self.min_sim):
            return Answer(
                text=IDK_TEXT,
                sources=[],
                intent=retrieval.intent,
                latency_ms=_elapsed_ms(started),
                model=self.client.model,
                refused=True,
            )

        prompt = render_prompt(retrieval)
        generated = self.client.generate(prompt, stream=False)
        if not isinstance(generated, str):
            raise TypeError("client.generate(stream=False) must return str")

        text = generated.strip() or IDK_TEXT
        sources = _sources(retrieval.chunks)
        if not _is_refusal(text):
            text = _ensure_source_citation(text, sources)
        return Answer(
            text=text,
            sources=sources,
            intent=retrieval.intent,
            latency_ms=_elapsed_ms(started),
            model=self.client.model,
            refused=_is_refusal(text),
        )

    def stream(self, retrieval: RetrievalResult) -> Iterator[AnswerChunk]:
        started = perf_counter()
        if _requires_idk_shortcut(retrieval, self.min_sim):
            yield AnswerChunk(
                delta=IDK_TEXT,
                text=IDK_TEXT,
                sources=[],
                intent=retrieval.intent,
                latency_ms=_elapsed_ms(started),
                model=self.client.model,
                refused=True,
                done=True,
            )
            return

        generated = self.client.generate(render_prompt(retrieval), stream=True)
        if isinstance(generated, str):
            raise TypeError("client.generate(stream=True) must return an iterator")

        text = ""
        sources = _sources(retrieval.chunks)
        for delta in generated:
            text += delta
            yield AnswerChunk(
                delta=delta,
                text=text,
                sources=sources,
                intent=retrieval.intent,
                latency_ms=_elapsed_ms(started),
                model=self.client.model,
                refused=_is_refusal(text),
            )

        text = text.strip() or IDK_TEXT
        if not _is_refusal(text):
            text = _ensure_source_citation(text, sources)
        yield AnswerChunk(
            delta="",
            text=text,
            sources=sources,
            intent=retrieval.intent,
            latency_ms=_elapsed_ms(started),
            model=self.client.model,
            refused=_is_refusal(text),
            done=True,
        )


def render_prompt(retrieval: RetrievalResult) -> str:
    numbered_chunks = "\n".join(_render_chunk(index, chunk) for index, chunk in enumerate(_prompt_chunks(retrieval.chunks), start=1))
    return (
        "You are a careful assistant answering questions strictly from the provided context.\n\n"
        "RULES:\n"
        "- Use ONLY the context below. Do NOT use outside knowledge.\n"
        f'- If the answer is not present, reply exactly: "{IDK_TEXT}"\n'
        "- Be concise (3-6 sentences) unless the user asks to compare; comparisons may be longer.\n"
        '- When citing a fact, name the entity it came from (e.g., "According to the Albert Einstein article, ...").\n'
        "- For comparison, contrast, similarity, or difference questions, synthesize across chunks when each claim is supported by the context.\n"
        f'- If any side of a requested comparison lacks support in the context, reply exactly: "{IDK_TEXT}"\n'
        '- For discovery questions, prefer explicit named discoveries from the context over research-process details.\n\n'
        "CONTEXT:\n"
        f"{numbered_chunks}\n\n"
        "QUESTION:\n"
        f"{retrieval.query}\n\n"
        "ANSWER:"
    )


def _render_chunk(index: int, chunk: RetrievedChunk) -> str:
    return f"[{index}] ({chunk.entity_name}, {chunk.type}) {chunk.text}"


def _prompt_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=lambda chunk: (chunk.entity_name, chunk.position, chunk.chunk_id))


def _requires_idk_shortcut(retrieval: RetrievalResult, min_sim: float) -> bool:
    if not retrieval.chunks:
        return True
    return retrieval.chunks[0].similarity < min_sim


def _sources(chunks: list[RetrievedChunk]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[str, str, int]] = set()
    for chunk in chunks:
        key = (chunk.entity_name, chunk.wikipedia_url, chunk.position)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                entity_name=chunk.entity_name,
                wikipedia_url=chunk.wikipedia_url,
                chunk_position=chunk.position,
            )
        )
    return sources


def _is_refusal(text: str) -> bool:
    return text.lstrip().lower().startswith("i don't know")


def _ensure_source_citation(text: str, sources: list[Source]) -> str:
    if not sources:
        return text

    entity_names = sorted({source.entity_name for source in sources})
    if any(f"{entity_name} article".casefold() in text.casefold() for entity_name in entity_names):
        return text
    text = re.sub(r"(?i)^according to \[\d+\],\s*", "", text)
    if len(entity_names) == 1:
        return f"According to the {entity_names[0]} article, {_citation_suffix(text, entity_names)}"
    return f"According to the retrieved articles for {', '.join(entity_names)}, {_citation_suffix(text, entity_names)}"


def _citation_suffix(text: str, entity_names: list[str]) -> str:
    if any(text.startswith(entity_name) for entity_name in entity_names):
        return text
    return f"{text[0].lower()}{text[1:]}"


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
