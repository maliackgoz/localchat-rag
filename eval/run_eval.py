"""Run golden-question evaluation for the local Wikipedia RAG pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from generation.answerer import Answer, Answerer
from generation.llm import DEFAULT_MODEL, OllamaClient
from retrieval.retriever import DEFAULT_K, DEFAULT_MIN_SIM, RetrievalResult, Retriever
from retrieval.router import DEFAULT_ROSTER_PATH, Intent, Roster, classify_intent, load_roster
from store.vector_store import DEFAULT_MANIFEST_PATH, DEFAULT_PERSIST_DIR, RetrievedChunk, TypeFilter, VectorStore


DEFAULT_GOLDEN_PATH = "eval/golden.jsonl"
DEFAULT_REPORT_DIR = "eval/results"
DEFAULT_PASS_THRESHOLD = 18


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_intent: Intent
    expected_entities: list[str]
    must_contain_any: list[str]
    must_not_refuse: bool


@dataclass(frozen=True)
class CaseChecks:
    intent: bool
    entities: bool
    content: bool
    refusal: bool


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    expected_intent: Intent
    actual_intent: Intent
    expected_entities: list[str]
    must_not_refuse: bool
    source_entities: list[str]
    retrieved_entities: list[str]
    matched_entities: list[str]
    answer: str
    refused: bool
    passed: bool
    checks: CaseChecks
    t_route_ms: int
    t_retrieve_ms: int
    t_generate_ms: int
    t_total_ms: int


@dataclass(frozen=True)
class EvalReport:
    generated_at: str
    golden_path: str
    model: str
    k: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    latency_ms: dict[str, dict[str, int]]
    cases: list[CaseResult]


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


class TimedStore:
    def __init__(self, store: QueryableStore) -> None:
        self.store = store
        self.query_ms = 0

    def reset(self) -> None:
        self.query_ms = 0

    def query(
        self,
        text: str,
        *,
        k: int = DEFAULT_K,
        type_filter: TypeFilter = "any",
        entity_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        started = perf_counter()
        try:
            return self.store.query(text, k=k, type_filter=type_filter, entity_filter=entity_filter)
        finally:
            self.query_ms += _elapsed_ms(started)


def evaluate(golden_path: str, *, k: int = DEFAULT_K, model: str = DEFAULT_MODEL) -> EvalReport:
    roster = load_roster(DEFAULT_ROSTER_PATH)
    store = TimedStore(VectorStore(DEFAULT_PERSIST_DIR, manifest_path=DEFAULT_MANIFEST_PATH))
    retriever = Retriever(store, roster, min_sim=DEFAULT_MIN_SIM)
    answerer = Answerer(OllamaClient(model=model), min_sim=DEFAULT_MIN_SIM)
    return evaluate_cases(
        load_golden(golden_path),
        retriever,
        answerer,
        roster,
        golden_path=golden_path,
        k=k,
        model=model,
        timed_store=store,
    )


def evaluate_cases(
    cases: list[GoldenCase],
    retriever: Retriever,
    answerer: Answerer,
    roster: Roster,
    *,
    golden_path: str,
    k: int = DEFAULT_K,
    model: str,
    timed_store: TimedStore | None = None,
) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        if timed_store is not None:
            timed_store.reset()

        total_started = perf_counter()
        route_started = perf_counter()
        classify_intent(case.question, roster)
        t_route_ms = _elapsed_ms(route_started)

        retrieval = retriever.retrieve(case.question, k=k)
        t_retrieve_ms = timed_store.query_ms if timed_store is not None else 0

        generate_started = perf_counter()
        answer = answerer.answer(retrieval)
        t_generate_ms = 0 if answer.refused and not retrieval.chunks else _elapsed_ms(generate_started)

        results.append(
            score_case(
                case,
                retrieval,
                answer,
                t_route_ms=t_route_ms,
                t_retrieve_ms=t_retrieve_ms,
                t_generate_ms=t_generate_ms,
                t_total_ms=_elapsed_ms(total_started),
            )
        )

    passed = sum(1 for result in results if result.passed)
    return EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        golden_path=golden_path,
        model=model,
        k=k,
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        latency_ms=_latency_summary(results),
        cases=results,
    )


def score_case(
    case: GoldenCase,
    retrieval: RetrievalResult,
    answer: Answer,
    *,
    t_route_ms: int,
    t_retrieve_ms: int,
    t_generate_ms: int,
    t_total_ms: int,
) -> CaseResult:
    source_entities = _unique(source.entity_name for source in answer.sources)
    retrieved_entities = _unique(chunk.entity_name for chunk in retrieval.chunks)
    checks = CaseChecks(
        intent=_intent_matches(case.expected_intent, retrieval.intent),
        entities=all(entity in source_entities for entity in case.expected_entities),
        content=_contains_any(answer.text, case.must_contain_any),
        refusal=answer.refused == (not case.must_not_refuse),
    )
    return CaseResult(
        id=case.id,
        question=case.question,
        expected_intent=case.expected_intent,
        actual_intent=retrieval.intent,
        expected_entities=case.expected_entities,
        must_not_refuse=case.must_not_refuse,
        source_entities=source_entities,
        retrieved_entities=retrieved_entities,
        matched_entities=retrieval.matched_entities,
        answer=answer.text,
        refused=answer.refused,
        passed=all(asdict(checks).values()),
        checks=checks,
        t_route_ms=t_route_ms,
        t_retrieve_ms=t_retrieve_ms,
        t_generate_ms=t_generate_ms,
        t_total_ms=t_total_ms,
    )


def load_golden(path: str) -> list[GoldenCase]:
    golden_path = Path(path)
    cases: list[GoldenCase] = []
    with golden_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{golden_path}:{line_number} is not valid JSON") from exc
            cases.append(_golden_case(data, golden_path, line_number))
    if not cases:
        raise ValueError(f"{golden_path} has no cases")
    return cases


def write_reports(report: EvalReport, report_dir: str) -> tuple[Path, Path]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _report_timestamp(report.generated_at)
    json_path = output_dir / f"{timestamp}.json"
    md_path = output_dir / f"{timestamp}.md"

    json_path.write_text(json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the localchat-rag golden-question eval.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH, help="Golden JSONL path.")
    parser.add_argument("--report", default=DEFAULT_REPORT_DIR, help="Directory for JSON and Markdown reports.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve per question.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--min-pass", type=int, default=DEFAULT_PASS_THRESHOLD, help="Minimum passing cases.")
    args = parser.parse_args(argv)

    try:
        report = evaluate(args.golden, k=args.k, model=args.model)
        json_path, md_path = write_reports(report, args.report)
    except RuntimeError as exc:
        print(exc)
        return 1

    failure_cases_passed = all(result.passed for result in report.cases if not result.must_not_refuse)
    ok = report.passed_cases >= args.min_pass and failure_cases_passed
    print(f"passed {report.passed_cases}/{report.total_cases} cases")
    print(f"reports: {json_path} and {md_path}")
    if not ok:
        print(f"eval failed: need >= {args.min_pass}/{report.total_cases} and all refusal cases passing")
    return 0 if ok else 1


def _golden_case(data: Any, path: Path, line_number: int) -> GoldenCase:
    if not isinstance(data, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object")
    expected_intent = _required_intent(data, "expected_intent", path, line_number)
    return GoldenCase(
        id=_required_string(data, "id", path, line_number),
        question=_required_string(data, "question", path, line_number),
        expected_intent=expected_intent,
        expected_entities=_required_string_list(data, "expected_entities", path, line_number),
        must_contain_any=_required_string_list(data, "must_contain_any", path, line_number),
        must_not_refuse=_required_bool(data, "must_not_refuse", path, line_number),
    )


def _latency_summary(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    metrics = {
        "route": [result.t_route_ms for result in results],
        "retrieve": [result.t_retrieve_ms for result in results],
        "generate": [result.t_generate_ms for result in results],
        "total": [result.t_total_ms for result in results],
    }
    return {name: {"p50": _percentile(values, 50), "p95": _percentile(values, 95)} for name, values in metrics.items()}


def _markdown_report(report: EvalReport) -> str:
    lines = [
        "# localchat-rag Eval Report",
        "",
        f"- Generated: {report.generated_at}",
        f"- Golden set: `{report.golden_path}`",
        f"- Model: `{report.model}`",
        f"- k: {report.k}",
        f"- Passed: {report.passed_cases}/{report.total_cases}",
        "",
        "## Latency",
        "",
        "| Metric | p50 ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for metric in ("route", "retrieve", "generate", "total"):
        values = report.latency_ms[metric]
        lines.append(f"| {metric} | {values['p50']} | {values['p95']} |")

    failures = [result for result in report.cases if not result.passed]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None.")
    else:
        for result in failures:
            failed_checks = [name for name, passed in asdict(result.checks).items() if not passed]
            lines.extend(
                [
                    f"### {result.id}",
                    "",
                    f"- Question: {result.question}",
                    f"- Failed checks: {', '.join(failed_checks)}",
                    f"- Expected intent: {result.expected_intent}; actual intent: {result.actual_intent}",
                    f"- Expected entities: {', '.join(result.expected_entities) or 'none'}",
                    f"- Source entities: {', '.join(result.source_entities) or 'none'}",
                    f"- Retrieved entities: {', '.join(result.retrieved_entities) or 'none'}",
                    f"- Refused: {result.refused}",
                    f"- Answer: {result.answer}",
                    "",
                ]
            )

    lines.extend(["", "## Cases", ""])
    for result in report.cases:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"- {status} `{result.id}` ({result.t_total_ms} ms): "
            f"{', '.join(result.retrieved_entities) or 'no retrieved entities'}"
        )
    return "\n".join(lines) + "\n"


def _report_payload(report: EvalReport) -> dict[str, Any]:
    return {
        "generated_at": report.generated_at,
        "golden_path": report.golden_path,
        "model": report.model,
        "k": report.k,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "latency_ms": report.latency_ms,
        "cases": [asdict(result) for result in report.cases],
    }


def _report_timestamp(generated_at: str) -> str:
    return generated_at.replace(":", "").replace("-", "").replace("+0000", "Z").replace("+00:00", "Z")


def _intent_matches(expected: Intent, actual: Intent) -> bool:
    if expected == actual:
        return True
    return expected in ("person", "place") and actual == "both"


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.casefold()
    return any(phrase.casefold() in lowered for phrase in phrases)


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[index]


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _required_string(data: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number} missing non-empty {key!r}")
    return value.strip()


def _required_string_list(data: dict[str, Any], key: str, path: Path, line_number: int) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{path}:{line_number} missing string-list {key!r}")
    return [item.strip() for item in value]


def _required_bool(data: dict[str, Any], key: str, path: Path, line_number: int) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}:{line_number} missing boolean {key!r}")
    return value


def _required_intent(data: dict[str, Any], key: str, path: Path, line_number: int) -> Intent:
    value = data.get(key)
    if value not in ("person", "place", "both"):
        raise ValueError(f"{path}:{line_number} has invalid {key!r}: {value!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
