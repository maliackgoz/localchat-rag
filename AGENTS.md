# AGENTS — localchat-rag

This project is built as a **multi-agent development effort**: separate developer roles, each with a clear scope, interface, and handoff. Agents do **not** talk to each other at runtime — the running system is a single Python process. "Agent" here means **a developer role with a defined contract**.

The intent mirrors the prior `google-in-a-day` repo: clear ownership per area, shared constraints in the PRD, and prompts that keep implementations aligned across roles.

## Why this structure

Project 3 combines retrieval (Project 1) with AI-driven workflows (Project 2) into a full RAG application. There are too many moving parts (ingestion, chunking, embeddings, vector store, routing, generation, UI, eval) for one undifferentiated session — splitting them by role keeps each prompt focused and each contract explicit.

## Agents

| # | Agent | File | Owns |
|---|-------|------|------|
| 0 | Orchestrator | [agents/00_orchestrator.md](agents/00_orchestrator.md) | PRD, roadmap, cross-cutting decisions, integration |
| 1 | Ingestion | [agents/01_ingestion.md](agents/01_ingestion.md) | Wikipedia fetch, raw document store |
| 2 | Chunking + Embedding | [agents/02_chunking_embedding.md](agents/02_chunking_embedding.md) | Chunk strategy, local embedding model |
| 3 | Vector Store | [agents/03_vector_store.md](agents/03_vector_store.md) | Chroma persistence, schema, metadata |
| 4 | Retrieval | [agents/04_retrieval.md](agents/04_retrieval.md) | Query router, filtered similarity search |
| 5 | Generation | [agents/05_generation.md](agents/05_generation.md) | Ollama integration, grounded prompts |
| 6 | Chat UI | [agents/06_chat_ui.md](agents/06_chat_ui.md) | Streamlit chat, CLI fallback |
| 7 | QA + Eval | [agents/07_qa_eval.md](agents/07_qa_eval.md) | Golden questions, failure cases, latency |
| 8 | Docs | [agents/08_docs.md](agents/08_docs.md) | README, product_prd, recommendation, demo |

## Shared contract

Every agent must respect:

1. **Local-only.** No external LLM API. Embedding and generation run on the user's machine (Ollama or sentence-transformers).
2. **Native first.** Prefer language-native functionality over heavyweight frameworks (per HW spec). Use Chroma for the vector store and `sentence-transformers` (or Ollama) for embeddings — but write the surrounding glue ourselves.
3. **Deterministic, inspectable storage.** Raw docs, chunks, and metadata live under `data/` in a documented format. Re-running ingestion must be idempotent.
4. **Grounded answers only.** The generation layer must refuse to answer outside retrieved context with `"I don't know based on the indexed data."`
5. **Two roles in retrieval.** Every chunk is tagged `type ∈ {person, place}` plus `entity_name`. Routing decides which subset to search.
6. **Single source of truth.** PRD ([product_prd.md](product_prd.md)) defines required behavior; if an agent disagrees, it updates the PRD before changing code.

## Handoff conventions

- Each agent's file states **inputs**, **outputs**, **interfaces** (function signatures or file paths), and **non-goals**.
- When an agent finishes a milestone, it updates the relevant section of [roadmap.md](roadmap.md) (status flag) and notes the integration point for the next agent.
- Cross-cutting changes (anything touching the PRD, schema, or two+ agent areas) are routed through the Orchestrator.

## Lifecycle of a milestone

1. **Orchestrator (Agent 0) opens a milestone** in [`roadmap.md`](roadmap.md) with exit criteria.
2. **The owning agent reads the PRD and its agent file**, drafts the deliverable, and runs its smoke test.
3. **Agent reports back**: deliverable path(s), how to run it, known limitations, suggested PRD edits.
4. **Orchestrator integrates**: merges the work, updates roadmap status, opens the next milestone.
5. **Agent 7 (QA)** runs the full eval at the end of each integration. A regression blocks the next milestone.

## Prompt template per agent

When kicking off an agent's work, use this template:

```
You are Agent N — <role>. Your scope is in agents/0N_<file>.md.
Locked decisions are in product_prd.md (do not change without an Orchestrator handoff).

Today's milestone: <M-id and exit criteria from roadmap.md>.
Inputs you should read first: <list>
Deliverables: <files / interfaces / CLI>
Out of scope: <bullets>

When done, report:
1. Files added / changed.
2. The exact command(s) to verify.
3. Known limitations.
4. Any PRD edit you want.
```

**In Cursor**, prefix the prompt with explicit `@`-mentions of the contract files so the model loads them deterministically (the `.cursor/rules/*.mdc` glob fires only on files already in the edit context — for net-new work, the `@`-mention is what guarantees the contract is read):

```
@agents/0N_<role>.md @product_prd.md @roadmap.md
<prompt template body above>
```

## Conflict resolution

When two agents disagree (e.g., chunking owner wants 600 tokens, retrieval owner wants 300):

1. Each writes a one-paragraph rationale in the relevant agent file.
2. Orchestrator picks the default in the PRD with a one-line "Why" note.
3. The losing rationale stays in the agent file as a documented alternative — useful for the demo's "trade-offs" beat.
