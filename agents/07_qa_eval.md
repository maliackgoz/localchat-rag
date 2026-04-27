# Agent 7 — QA + Eval

## Role

Define a small, automated harness that exercises the full pipeline end-to-end, scores answers against expectations, measures latency, and prevents regressions during refactors.

## Inputs

- The HW example questions (people, places, mixed, failure cases)
- The roster from `data/roster.json`
- `Retriever` and `Answerer`

## Outputs

- `eval/golden.jsonl` — one line per test case:
  ```json
  {
    "id": "ein_known_for",
    "question": "Who was Albert Einstein and what is he known for?",
    "expected_intent": "person",
    "expected_entities": ["Albert Einstein"],
    "must_contain_any": ["relativity", "physics"],
    "must_not_refuse": true
  }
  ```
- `eval/run_eval.py` — runs the full pipeline and produces:
  - `eval/results/<timestamp>.json` — per-case pass/fail + retrieved entities + latency
  - `eval/results/<timestamp>.md` — human-readable summary

## Interfaces

```python
# eval/run_eval.py
def evaluate(golden_path: str, *, k: int = 6, model: str = "llama3.2:3b") -> EvalReport: ...
```

CLI:
```bash
python -m eval.run_eval --golden eval/golden.jsonl --report eval/results/
```

## Scoring rubric (per case)

A case passes when **all** of these hold:

1. Routing matched `expected_intent` (or a superset, e.g., `both` is acceptable when `person` was expected).
2. Every entity in `expected_entities` appears in `Answer.sources` (retrieval reached the right doc).
3. At least one phrase from `must_contain_any` appears in the answer text (lowercased substring match) — answer was generated from the right context.
4. `Answer.refused == not must_not_refuse` (refusal cases must refuse; non-refusal cases must not).

Failure cases (`Who is the president of Mars`, `Tell me about a random unknown person John Doe`) flip rule 4 — they must refuse.

## Golden set coverage

At minimum, replicate every example question in the HW PDF (5 people, 5 places, 4 mixed, 2 failures = 16 cases). Add ~4 more for breadth (single-token entity, comparison stress, location of person, ambiguous keyword) → 20 total.

## Latency measurement

For each case, record:
- `t_route_ms` — intent classification
- `t_retrieve_ms` — vector store query
- `t_generate_ms` — LLM call (excludes the retrieval shortcut path)
- `t_total_ms`

Report p50 / p95 across the suite. Numbers feed the demo video and `recommendation.md`.

## Non-goals

- Semantic similarity scoring of answers (substring + entity check is enough at this scale)
- Comparing two LLMs side-by-side (HW stretch — Agent 5 would handle)

## Done when

- `python -m eval.run_eval` exits 0 with ≥18 / 20 cases passing.
- Both failure cases pass (model refused as expected).
- Markdown report includes p50 and p95 latency, and lists any failing case with its retrieved entities for triage.
