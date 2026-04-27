"""Line-based CLI fallback for localchat-rag."""

from __future__ import annotations

import argparse
from time import perf_counter

from app.runtime import (
    DEFAULT_K,
    create_runtime,
    elapsed_ms,
    footer_text,
    ordered_context_chunks,
    rebuild_index,
    stats_text,
)
from generation.llm import DEFAULT_MODEL
from retrieval.router import DEFAULT_ROSTER_PATH
from store.vector_store import DEFAULT_MANIFEST_PATH, DEFAULT_PERSIST_DIR, RetrievedChunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the localchat-rag CLI.")
    parser.add_argument("--roster", default=DEFAULT_ROSTER_PATH, help="Roster JSON path.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persistence directory.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Store manifest path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve.")
    args = parser.parse_args(argv)

    try:
        runtime = create_runtime(
            roster_path=args.roster,
            persist_dir=args.persist_dir,
            manifest_path=args.manifest,
            llm_model=args.model,
        )
    except RuntimeError as exc:
        print(exc)
        return 1
    show_sources = False
    turns: list[dict[str, object]] = []

    print('localchat-rag - type a question, ":sources" to toggle, ":reset" to clear, ":quit" to exit.')
    while True:
        try:
            raw = input("> ")
        except EOFError:
            print()
            return 0

        query = raw.strip()
        if not query:
            continue
        if query == ":quit":
            return 0
        if query == ":sources":
            show_sources = not show_sources
            print(f"sources {'on' if show_sources else 'off'}")
            continue
        if query == ":reset":
            turns.clear()
            print("chat reset")
            continue
        if query == ":stats":
            print(stats_text(runtime.store.stats()))
            continue
        if query == ":build":
            report = rebuild_index(runtime)
            print(f"indexed {report.get('chunks_built', 0)} chunks for {report.get('entities_built', 0)} entities")
            continue

        started = perf_counter()
        try:
            retrieval = runtime.retriever.retrieve(query, k=args.k)
            answer = runtime.answerer.answer(retrieval)
        except RuntimeError as exc:
            print(exc)
            print(f"Start Ollama and run `ollama pull {args.model}` before asking again.")
            continue

        latency_ms = elapsed_ms(started)
        print(f"[{footer_text(answer.intent, latency_ms, answer.refused)}]")
        print(answer.text)
        if show_sources:
            _print_sources(retrieval.chunks)
        turns.append(
            {
                "question": query,
                "answer": answer.text,
                "intent": answer.intent,
                "latency_ms": latency_ms,
                "refused": answer.refused,
            }
        )


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        print("sources: none")
        return
    print("sources:")
    for chunk in ordered_context_chunks(chunks):
        snippet = " ".join(chunk.text.split())
        if len(snippet) > 240:
            snippet = f"{snippet[:237]}..."
        print(f"- {chunk.entity_name} ({chunk.wikipedia_url}) pos={chunk.position}: {snippet}")


if __name__ == "__main__":
    raise SystemExit(main())
