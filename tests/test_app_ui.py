from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.runtime import (
    add_entity_to_roster,
    footer_text,
    intent_label,
    ordered_context_chunks,
    run_ingestion_pipeline,
    stats_text,
)
from chunking.splitter import ChunkingReport
from ingest.wikipedia import IngestReport
from store.vector_store import RetrievedChunk


def retrieved_chunk(chunk_id: str, entity_name: str, position: int, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        entity_name=entity_name,
        type="person",
        wikipedia_url=f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}",
        position=position,
        text=f"{entity_name} chunk {position}",
        distance=1.0 - similarity,
        similarity=similarity,
    )


class AppRuntimeTests(unittest.TestCase):
    def test_intent_label_matches_ui_contract(self) -> None:
        self.assertEqual(intent_label("person"), "people")
        self.assertEqual(intent_label("place"), "places")
        self.assertEqual(intent_label("both"), "both")

    def test_footer_includes_latency_and_refusal_flag(self) -> None:
        self.assertEqual(footer_text("person", 321, False), "searched in: people · 321ms")
        self.assertEqual(footer_text("place", 4, True), "searched in: places · 4ms · refused")

    def test_context_chunks_use_prompt_order(self) -> None:
        chunks = [
            retrieved_chunk("curie__0002", "Marie Curie", 2, 0.8),
            retrieved_chunk("curie__0001", "Marie Curie", 1, 0.9),
            retrieved_chunk("ada__0000", "Ada Lovelace", 0, 0.7),
        ]

        ordered = ordered_context_chunks(chunks)

        self.assertEqual([chunk.chunk_id for chunk in ordered], ["ada__0000", "curie__0001", "curie__0002"])

    def test_stats_text_summarizes_sidebar_values(self) -> None:
        text = stats_text(
            {
                "collection": "wikipedia_rag",
                "total_chunks": 10,
                "entities": 4,
                "by_type": {"person": 6, "place": 4},
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "dim": 384,
            }
        )

        self.assertIn("total_chunks: 10", text)
        self.assertIn("people_chunks: 6", text)
        self.assertIn("embedding_model: sentence-transformers/all-MiniLM-L6-v2", text)

    def test_add_entity_to_roster_appends_normalized_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roster_path = Path(temp_dir) / "roster.json"
            roster_path.write_text(
                json.dumps({"_comment": "test", "people": ["Ada Lovelace"], "places": ["Eiffel Tower"]}),
                encoding="utf-8",
            )

            changed = add_entity_to_roster(" Alan   Turing ", "person", roster_path=str(roster_path))
            duplicate = add_entity_to_roster("alan turing", "person", roster_path=str(roster_path))

            roster = json.loads(roster_path.read_text(encoding="utf-8"))
            self.assertTrue(changed)
            self.assertFalse(duplicate)
            self.assertEqual(roster["people"], ["Ada Lovelace", "Alan Turing"])
            self.assertEqual(roster["_comment"], "test")

    def test_add_entity_to_roster_rejects_cross_type_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roster_path = Path(temp_dir) / "roster.json"
            roster_path.write_text(
                json.dumps({"people": ["Ada Lovelace"], "places": ["Eiffel Tower"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "already exists under people"):
                add_entity_to_roster("Ada Lovelace", "place", roster_path=str(roster_path))

    def test_run_ingestion_pipeline_orders_ingest_chunk_and_index(self) -> None:
        runtime = object()
        ingest_report = IngestReport(fetched=1, cached=2, total=3, manifest_path="raw/_manifest.json")
        chunk_report = ChunkingReport(entities=3, chunks=9, output_dir="chunks")
        index_report = {"entities_built": 3, "chunks_built": 9}

        with (
            patch("app.runtime.ingest_roster", return_value=ingest_report) as ingest,
            patch("app.runtime.chunk_raw_documents", return_value=chunk_report) as chunk,
            patch("app.runtime.rebuild_index", return_value=index_report) as rebuild,
        ):
            report = run_ingestion_pipeline(
                runtime,  # type: ignore[arg-type]
                roster_path="roster.json",
                raw_dir="raw",
                chunks_dir="chunks",
                force=True,
            )

        ingest.assert_called_once_with("roster.json", "raw", force=True)
        chunk.assert_called_once_with("raw", "chunks")
        rebuild.assert_called_once_with(runtime, chunks_dir="chunks")
        self.assertEqual(report.ingest.fetched, 1)
        self.assertEqual(report.chunking.chunks, 9)
        self.assertEqual(report.index["chunks_built"], 9)


if __name__ == "__main__":
    unittest.main()
