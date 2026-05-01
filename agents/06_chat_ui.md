# Agent 6 — Chat UI

## Role

Provide a chat experience over the RAG pipeline. Streamlit is the primary surface (best for the demo video); a CLI is the fallback so the grader can run the system without a browser.

## Inputs

- `Retriever` (Agent 4) and `Answerer` (Agent 5)
- `VectorStore.stats()` for the sidebar

## Outputs

- `app/localchat_rag.py` — Streamlit hub (`st.navigation`); sidebar labels: **localchat-rag** (landing), **Chat**, **Ingestion**
- `app/landing_page.py` — system overview and locked technical decisions
- `app/chat_page.py` — chat UI
- `app/ingestion_page.py` — ingest + chunk + Chroma rebuild pipeline
- `app/cli.py` — line-based REPL

## Streamlit interface

Layout:

- **Landing (localchat-rag):** README-style overview; table of locked embedding, LLM, chunk, store, and retrieval settings.
- **Chat:** same behavior as MVP chat: sidebar model / embedding / chunk count / reset / context toggle / clear-index confirm; main history via `st.chat_message`; bottom `st.chat_input`; expandable sources showing entity name, Wikipedia URL, and chunk text fed to the model; footer intent + latency ms + refusal. Streaming renders tokens incrementally until the final chunk.

- **Ingestion:** roster updates, Wikipedia fetch, chunk regeneration, vector upsert + stats.

```bash
streamlit run app/localchat_rag.py
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

- `streamlit run app/localchat_rag.py` opens the UI; **Chat** answers all 14 example questions from the HW PDF.
- The two failure-case questions render with refusal styling.
- CLI handles the same questions and prints sources when `:sources` is on.
