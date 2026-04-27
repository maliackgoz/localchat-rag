from __future__ import annotations

import unittest

from app.runtime import footer_text, intent_label, ordered_context_chunks, stats_text
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


if __name__ == "__main__":
    unittest.main()
