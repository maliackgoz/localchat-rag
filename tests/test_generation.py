from __future__ import annotations

import unittest
from collections.abc import Iterator

from generation.answerer import IDK_TEXT, Answerer, render_prompt
from retrieval.retriever import RetrievalResult
from store.vector_store import RetrievedChunk


class FakeClient:
    model = "fake-llm"

    def __init__(self, response: str = "According to the Marie Curie article, she discovered polonium.") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, *, stream: bool = False) -> Iterator[str] | str:
        self.calls.append({"prompt": prompt, "stream": stream})
        if stream:
            return iter(["Marie Curie ", "discovered radium."])
        return self.response


def retrieved_chunk(
    chunk_id: str,
    entity_name: str,
    type_: str,
    position: int,
    similarity: float,
    text: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        entity_name=entity_name,
        type=type_,  # type: ignore[arg-type]
        wikipedia_url=f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}",
        position=position,
        text=text,
        distance=1.0 - similarity,
        similarity=similarity,
    )


def retrieval(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        query="What did Marie Curie discover?",
        intent="person",
        matched_entities=["Marie Curie"],
        chunks=chunks,
    )


class AnswererTests(unittest.TestCase):
    def test_empty_retrieval_returns_idk_without_calling_ollama(self) -> None:
        client = FakeClient()
        answer = Answerer(client).answer(retrieval([]))

        self.assertEqual(answer.text, IDK_TEXT)
        self.assertTrue(answer.refused)
        self.assertEqual(answer.sources, [])
        self.assertEqual(client.calls, [])

    def test_low_similarity_retrieval_returns_idk_without_calling_ollama(self) -> None:
        client = FakeClient()
        answer = Answerer(client).answer(
            retrieval(
                [
                    retrieved_chunk(
                        "marie-curie__0000",
                        "Marie Curie",
                        "person",
                        0,
                        0.24,
                        "Marie Curie studied radioactivity.",
                    )
                ]
            )
        )

        self.assertEqual(answer.text, IDK_TEXT)
        self.assertTrue(answer.refused)
        self.assertEqual(client.calls, [])

    def test_answer_renders_prompt_and_dedupes_sources(self) -> None:
        chunk = retrieved_chunk(
            "marie-curie__0000",
            "Marie Curie",
            "person",
            0,
            0.90,
            "Marie Curie discovered polonium and radium while studying radioactivity.",
        )
        client = FakeClient()
        answer = Answerer(client).answer(retrieval([chunk, chunk]))

        self.assertFalse(answer.refused)
        self.assertEqual(answer.model, "fake-llm")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].entity_name, "Marie Curie")
        self.assertEqual(answer.sources[0].chunk_position, 0)
        self.assertEqual(client.calls[0]["stream"], False)
        prompt = str(client.calls[0]["prompt"])
        self.assertIn("[1] (Marie Curie, person)", prompt)
        self.assertIn("Use ONLY the context below", prompt)
        self.assertIn("synthesize across chunks", prompt)
        self.assertIn("What did Marie Curie discover?", prompt)

    def test_refusal_detection_marks_llm_idk_response(self) -> None:
        chunk = retrieved_chunk(
            "marie-curie__0000",
            "Marie Curie",
            "person",
            0,
            0.90,
            "Marie Curie studied radioactivity.",
        )
        answer = Answerer(FakeClient("I don't know based on the indexed data.")).answer(retrieval([chunk]))

        self.assertTrue(answer.refused)

    def test_non_refusal_answer_is_prefixed_when_entity_citation_is_missing(self) -> None:
        chunk = retrieved_chunk(
            "marie-curie__0000",
            "Marie Curie",
            "person",
            0,
            0.90,
            "Marie Curie discovered polonium and radium.",
        )
        answer = Answerer(FakeClient("She discovered polonium and radium.")).answer(retrieval([chunk]))

        self.assertEqual(
            answer.text,
            "According to the Marie Curie article, she discovered polonium and radium.",
        )

    def test_numbered_citation_intro_is_replaced_with_entity_article(self) -> None:
        chunk = retrieved_chunk(
            "marie-curie__0000",
            "Marie Curie",
            "person",
            0,
            0.90,
            "Marie Curie discovered polonium and radium.",
        )
        answer = Answerer(FakeClient("According to [1], Marie Curie discovered polonium and radium.")).answer(
            retrieval([chunk])
        )

        self.assertEqual(
            answer.text,
            "According to the Marie Curie article, Marie Curie discovered polonium and radium.",
        )

    def test_stream_yields_deltas_and_final_chunk(self) -> None:
        chunk = retrieved_chunk(
            "marie-curie__0000",
            "Marie Curie",
            "person",
            0,
            0.90,
            "Marie Curie studied radioactivity.",
        )
        client = FakeClient()
        chunks = list(Answerer(client).stream(retrieval([chunk])))

        self.assertEqual([chunk.delta for chunk in chunks], ["Marie Curie ", "discovered radium.", ""])
        self.assertEqual(chunks[-1].text, "According to the Marie Curie article, Marie Curie discovered radium.")
        self.assertTrue(chunks[-1].done)
        self.assertFalse(chunks[-1].refused)
        self.assertEqual(client.calls[0]["stream"], True)

    def test_render_prompt_matches_locked_chunk_format(self) -> None:
        prompt = render_prompt(
            retrieval(
                [
                    retrieved_chunk(
                        "marie-curie__0000",
                        "Marie Curie",
                        "person",
                        0,
                        0.90,
                        "Marie Curie studied radioactivity.",
                    )
                ]
            )
        )

        self.assertIn("[1] (Marie Curie, person) Marie Curie studied radioactivity.", prompt)
        self.assertTrue(prompt.endswith("ANSWER:"))


if __name__ == "__main__":
    unittest.main()
