# Agent 5 — Generation

## Role

Take a query plus retrieved chunks and produce a grounded answer using a local LLM via Ollama. Refuse to answer when the data does not support a response.

## Inputs

- `RetrievalResult` from Agent 4
- LLM config: `model = "llama3.2:3b"`, temperature 0.1, max_tokens 512
- Ollama running at `http://127.0.0.1:11434`

## Outputs

```python
@dataclass
class Answer:
    text: str
    sources: list[Source]            # entity_name, wikipedia_url, chunk_position
    intent: Intent
    latency_ms: int
    model: str
    refused: bool                    # True when the model returned the IDK fallback
```

## Interfaces

```python
# generation/llm.py
class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434"): ...
    def generate(self, prompt: str, *, stream: bool = False) -> Iterator[str] | str: ...
    def health(self) -> bool: ...

# generation/answerer.py
class Answerer:
    def __init__(self, client: OllamaClient): ...
    def answer(self, retrieval: RetrievalResult) -> Answer: ...
    def stream(self, retrieval: RetrievalResult) -> Iterator[AnswerChunk]: ...
```

## Prompt template

```
You are a careful assistant answering questions strictly from the provided context.

RULES:
- Use ONLY the context below. Do NOT use outside knowledge.
- If the answer is not present, reply exactly: "I don't know based on the indexed data."
- Be concise (3–6 sentences) unless the user asks to compare; comparisons may be longer.
- When citing a fact, name the entity it came from (e.g., "According to the Albert Einstein article, ...").

CONTEXT:
{numbered_chunks}

QUESTION:
{query}

ANSWER:
```

`numbered_chunks` is rendered as:
```
[1] (Albert Einstein, person) <chunk text>
[2] (Eiffel Tower, place) <chunk text>
...
```

## Hallucination guards

1. **Empty / weak retrieval shortcut.** If `len(retrieval.chunks) == 0` or top similarity < `min_sim`, skip the LLM and return the IDK message directly. Saves latency and prevents the model from making things up off whitespace.
2. **Refusal detection.** Treat any answer that starts with "I don't know" as `refused=True` so the UI can render it differently and Agent 7 can score it.
3. **Source coverage.** `Answer.sources` is the deduped set of entity names actually present in the chunks fed to the model — used by the UI to display citations and by Agent 7 to verify the right entity was reached.

## Streaming

Streaming is mandatory for the Streamlit UI's UX (HW optional extension, but it materially helps the demo). `OllamaClient.generate(stream=True)` yields token deltas; the answerer wraps them with periodic source-line updates.

## Non-goals

- Multi-turn conversation memory (stretch — Agent 6 owns chat state if added)
- Tool use / function calling (out of scope)
- Comparing across two LLMs in parallel (stretch — HW optional)

## Done when

- `answer(retrieval)` returns the IDK fallback for empty context without calling Ollama.
- For "What did Marie Curie discover?" the answer cites Marie Curie and mentions radioactivity / polonium / radium (verified by Agent 7).
- Streaming yields tokens within ~1s of submit on the dev machine (Agent 7 records latency).
