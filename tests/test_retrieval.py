from __future__ import annotations

import unittest

from retrieval.retriever import Retriever
from retrieval.router import Roster, classify_intent
from store.vector_store import RetrievedChunk, TypeFilter


class FakeStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        text: str,
        *,
        k: int = 6,
        type_filter: TypeFilter = "any",
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"text": text, "k": k, "type_filter": type_filter, "entity_filter": entity_filter})
        results = self.chunks
        if type_filter != "any":
            results = [chunk for chunk in results if chunk.type == type_filter]
        if entity_filter is not None:
            results = [chunk for chunk in results if chunk.entity_name in entity_filter]
        return sorted(results, key=lambda chunk: -chunk.similarity)[:k]


def retrieved_chunk(chunk_id: str, entity_name: str, type_: str, position: int, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        entity_name=entity_name,
        type=type_,  # type: ignore[arg-type]
        wikipedia_url=f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}",
        position=position,
        text=f"{entity_name} chunk {position}",
        distance=1.0 - similarity,
        similarity=similarity,
    )


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roster = Roster(
            people=["Nikola Tesla", "Marie Curie", "Ada Lovelace"],
            places=["Eiffel Tower", "Statue of Liberty", "Mount Everest"],
        )

    def test_full_entity_match_wins_over_keyword_cues(self) -> None:
        intent, matched_entities = classify_intent("Who built the Eiffel Tower?", self.roster)

        self.assertEqual(intent, "place")
        self.assertEqual(matched_entities, ["Eiffel Tower"])

    def test_unique_entity_token_matches_one_word_query(self) -> None:
        intent, matched_entities = classify_intent("Tesla", self.roster)

        self.assertEqual(intent, "person")
        self.assertEqual(matched_entities, ["Nikola Tesla"])

    def test_keyword_cues_route_without_entity_match(self) -> None:
        self.assertEqual(classify_intent("Where is the famous mountain located?", self.roster), ("place", []))
        self.assertEqual(classify_intent("Who wrote important notes?", self.roster), ("person", []))
        self.assertEqual(classify_intent("Where was the composer born?", self.roster), ("both", []))

    def test_default_is_both(self) -> None:
        self.assertEqual(classify_intent("tell me something interesting", self.roster), ("both", []))


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roster = Roster(
            people=["Nikola Tesla", "Marie Curie"],
            places=["Eiffel Tower", "Statue of Liberty"],
        )

    def test_entity_pinned_search_preserves_each_matched_entity(self) -> None:
        store = FakeStore(
            [
                retrieved_chunk("eiffel__0000", "Eiffel Tower", "place", 0, 0.91),
                retrieved_chunk("eiffel__0001", "Eiffel Tower", "place", 1, 0.90),
                retrieved_chunk("eiffel__0002", "Eiffel Tower", "place", 2, 0.89),
                retrieved_chunk("eiffel__0003", "Eiffel Tower", "place", 3, 0.88),
                retrieved_chunk("statue__0000", "Statue of Liberty", "place", 0, 0.40),
            ]
        )
        retriever = Retriever(store, self.roster)

        result = retriever.retrieve("Compare Eiffel Tower and Statue of Liberty", k=2)

        self.assertEqual(result.intent, "place")
        self.assertEqual(set(result.matched_entities), {"Eiffel Tower", "Statue of Liberty"})
        self.assertEqual({chunk.entity_name for chunk in result.chunks}, {"Eiffel Tower", "Statue of Liberty"})
        self.assertTrue(any(call["entity_filter"] == ["Statue of Liberty"] for call in store.calls))

    def test_comparison_queries_prefer_overview_chunks_for_each_entity(self) -> None:
        store = FakeStore(
            [
                retrieved_chunk("tesla__0005", "Nikola Tesla", "person", 5, 0.95),
                retrieved_chunk("eiffel__0004", "Eiffel Tower", "place", 4, 0.90),
                retrieved_chunk("eiffel__0000", "Eiffel Tower", "place", 0, 0.36),
                retrieved_chunk("tesla__0000", "Nikola Tesla", "person", 0, 0.35),
            ]
        )
        retriever = Retriever(store, self.roster)

        result = retriever.retrieve("What is different between Nikola Tesla and the Eiffel Tower?", k=2)

        self.assertEqual({chunk.entity_name for chunk in result.chunks}, {"Nikola Tesla", "Eiffel Tower"})
        self.assertEqual({chunk.position for chunk in result.chunks}, {0})
        self.assertTrue(any(call["text"] == "Nikola Tesla" for call in store.calls))
        self.assertTrue(any(call["text"] == "Eiffel Tower" for call in store.calls))

    def test_one_word_entity_query_can_beat_higher_unpinned_score(self) -> None:
        store = FakeStore(
            [
                retrieved_chunk("curie__0000", "Marie Curie", "person", 0, 0.95),
                retrieved_chunk("tesla__0000", "Nikola Tesla", "person", 0, 0.55),
            ]
        )
        retriever = Retriever(store, self.roster)

        result = retriever.retrieve("Tesla", k=1)

        self.assertEqual(result.matched_entities, ["Nikola Tesla"])
        self.assertEqual([chunk.entity_name for chunk in result.chunks], ["Nikola Tesla"])

    def test_both_intent_runs_person_and_place_searches(self) -> None:
        store = FakeStore(
            [
                retrieved_chunk("curie__0000", "Marie Curie", "person", 0, 0.70),
                retrieved_chunk("eiffel__0000", "Eiffel Tower", "place", 0, 0.65),
            ]
        )
        retriever = Retriever(store, self.roster)

        result = retriever.retrieve("Where was the inventor born?", k=2)

        self.assertEqual(result.intent, "both")
        self.assertEqual([call["type_filter"] for call in store.calls], ["person", "place"])
        self.assertEqual({chunk.type for chunk in result.chunks}, {"person", "place"})

    def test_low_similarity_results_are_dropped(self) -> None:
        store = FakeStore([retrieved_chunk("curie__0000", "Marie Curie", "person", 0, 0.24)])
        retriever = Retriever(store, self.roster)

        result = retriever.retrieve("Who is the president of Mars?", k=3)

        self.assertEqual(result.intent, "person")
        self.assertEqual(result.chunks, [])


if __name__ == "__main__":
    unittest.main()
