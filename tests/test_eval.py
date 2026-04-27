from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.run_eval import CaseChecks, CaseResult, EvalReport, GoldenCase, load_golden, score_case, write_reports
from generation.answerer import Answer, Source
from retrieval.retriever import RetrievalResult
from store.vector_store import RetrievedChunk


def retrieved_chunk(entity_name: str, type_: str = "person") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{entity_name.lower().replace(' ', '-')}__0000",
        entity_name=entity_name,
        type=type_,  # type: ignore[arg-type]
        wikipedia_url=f"https://en.wikipedia.org/wiki/{entity_name.replace(' ', '_')}",
        position=0,
        text=f"{entity_name} context",
        distance=0.1,
        similarity=0.9,
    )


class EvalHarnessTests(unittest.TestCase):
    def test_load_golden_and_score_case_accepts_both_as_superset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            golden_path = Path(tmp_dir) / "golden.jsonl"
            golden_path.write_text(
                json.dumps(
                    {
                        "id": "einstein_known_for",
                        "question": "Who was Albert Einstein?",
                        "expected_intent": "person",
                        "expected_entities": ["Albert Einstein"],
                        "must_contain_any": ["relativity"],
                        "must_not_refuse": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            case = load_golden(str(golden_path))[0]

        chunk = retrieved_chunk("Albert Einstein")
        result = score_case(
            case,
            RetrievalResult(
                query=case.question,
                intent="both",
                matched_entities=["Albert Einstein"],
                chunks=[chunk],
            ),
            Answer(
                text="According to the Albert Einstein article, he developed relativity.",
                sources=[
                    Source(
                        entity_name="Albert Einstein",
                        wikipedia_url=chunk.wikipedia_url,
                        chunk_position=chunk.position,
                    )
                ],
                intent="both",
                latency_ms=12,
                model="fake",
                refused=False,
            ),
            t_route_ms=1,
            t_retrieve_ms=2,
            t_generate_ms=3,
            t_total_ms=6,
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.checks.intent)
        self.assertEqual(result.source_entities, ["Albert Einstein"])

    def test_write_reports_includes_latency_and_failure_details(self) -> None:
        failed = CaseResult(
            id="missing_content",
            question="What did Marie Curie discover?",
            expected_intent="person",
            actual_intent="person",
            expected_entities=["Marie Curie"],
            must_not_refuse=True,
            source_entities=["Marie Curie"],
            retrieved_entities=["Marie Curie"],
            matched_entities=["Marie Curie"],
            answer="According to the Marie Curie article, she studied science.",
            refused=False,
            passed=False,
            checks=CaseChecks(intent=True, entities=True, content=False, refusal=True),
            t_route_ms=1,
            t_retrieve_ms=10,
            t_generate_ms=100,
            t_total_ms=111,
        )
        report = EvalReport(
            generated_at="2026-04-27T19:50:00+00:00",
            golden_path="eval/golden.jsonl",
            model="fake",
            k=6,
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            latency_ms={
                "route": {"p50": 1, "p95": 1},
                "retrieve": {"p50": 10, "p95": 10},
                "generate": {"p50": 100, "p95": 100},
                "total": {"p50": 111, "p95": 111},
            },
            cases=[failed],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path, md_path = write_reports(report, tmp_dir)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")

        self.assertEqual(payload["failed_cases"], 1)
        self.assertIn("| total | 111 | 111 |", markdown)
        self.assertIn("missing_content", markdown)
        self.assertIn("Retrieved entities: Marie Curie", markdown)


if __name__ == "__main__":
    unittest.main()
