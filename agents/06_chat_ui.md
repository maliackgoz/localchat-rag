# Agent 6 — Chat UI

## Role

Provide a chat experience over the RAG pipeline. Streamlit is the primary surface (best for the demo video); a CLI is the fallback so the grader can run the system without a browser.

## Inputs

- `Retriever` (Agent 4) and `Answerer` (Agent 5)
- `VectorStore.stats()` for the sidebar

## Outputs

- `app/streamlit_app.py` — chat UI
- `app/cli.py` — line-based REPL

## Streamlit interface

Layout:

- **Sidebar:** model name, embedding model, total chunks, "Reset chat" button, "Show retrieved context" toggle.
- **Main pane:** chat history (user + assistant turns) using `st.chat_message`.
- **Input:** `st.chat_input` at the bottom.
- **Per-answer expander:** "Sources used" listing entity name, Wikipedia URL, and the chunk text actually shown to the model.
- **Per-answer footer:** intent badge ("searched in: people / places / both"), latency in ms, refusal flag.

Streaming: tokens appear as they arrive; sources are filled in once the answer completes.

```bash
streamlit run app/streamlit_app.py
```

## CLI interface

```
$ python -m app.cli
localchat-rag — type a question, ":sources" to toggle, ":reset" to clear, ":quit" to exit.
> What did Marie Curie discover?
[searched in: people · 320ms]
According to the Marie Curie article, ...
sources: Marie Curie (https://...)
>
```

Flags:
- `:sources` — toggle source display
- `:reset` — clear chat history
- `:stats` — print vector store stats
- `:quit` — exit

## State

In-memory only for MVP. Each session is independent. (Stretch: persist chat history to `data/chat_log.jsonl` so demos can be replayed.)

## Reset behavior

"Reset chat" clears the in-memory turn list. It does **not** touch the vector store. A separate "Clear index" button (sidebar, behind a confirm) calls `VectorStore.reset()` for evaluators who want to demonstrate a fresh ingest in the demo.

## Non-goals

- Auth / multi-user (single local user)
- Persistent chat history across runs (stretch)
- Conversation memory injected into retrieval (stretch — risks of contaminating context outweigh benefit at this scale)

## Done when

- `streamlit run app/streamlit_app.py` opens a chat that answers all 14 example questions from the HW PDF.
- The two failure-case questions render with refusal styling.
- CLI handles the same questions and prints sources when `:sources` is on.
