# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An autonomous multi-agent research assistant built with LangGraph, FastAPI, and a RAG pipeline backed by Pinecone. A supervisor agent decomposes queries into sub-tasks; worker agents retrieve context via RAG and Tavily web search; a writer synthesises a cited report; a critic quality-gates the output and triggers re-search if confidence is low. Responses stream token-by-token to the client.

## Commands

```bash
uv sync
PYTHONPATH=. uv run uvicorn api.routes:app --reload
PYTHONPATH=. uv run pytest tests/ -v
PYTHONPATH=. uv run pytest tests/test_agent.py::test_name -v
uv run ruff format . && uv run ruff check .
uv run python -m rag.ingest docs/
uv add <package-name>
```

## Environment variables

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=research-agent
TAVILY_API_KEY=...
APP_ENV=development
```

Pinecone index must be created with dimension `1536` (matching `text-embedding-3-small`) and cosine similarity.

## Architecture

**Node flow:** `supervisor → researcher → writer → critic → (END or back to researcher)`

- `supervisor` — decomposes query, sets `state["next"]`
- `researcher` — RAG + Tavily search, populates `state["retrieved_docs"]`
- `writer` — assembles cited report from retrieved docs
- `critic` — scores quality; routes back to researcher if score < threshold

Graph compiled in `agent/graph.py`. State in `agent/state.py`. Nodes in `agent/nodes.py`. Tools in `agent/tools.py`. Prompts in `prompts/templates.py`. Pinecone singleton in `vectorstore/client.py`.

RAG is split: `rag/ingest.py` (offline, writes to Pinecone) and `rag/retriever.py` (online, reads per request). Chunking: `chunk_size=512`, `chunk_overlap=64`.

## Teaching reference

**Reference repo:** `/Users/havanthien/AI Agentic Learning/agents/`

When the user asks for step-by-step explanations, always point to the specific reference file and section first, explain the concept, then let the user write.

| File to implement | Reference to read first |
|---|---|
| `agent/state.py` ✅ | `4_langgraph/sidekick.py` lines 1–30 |
| `agent/tools.py` ✅ | `4_langgraph/sidekick_tools.py` (full file) |
| `agent/nodes.py` ✅ | `4_langgraph/sidekick.py` lines 57–165 |
| `agent/graph.py` | `4_langgraph/sidekick.py` lines 167–220 + `4_langgraph/4_lab4.ipynb` |
| `rag/embedder.py` + `rag/retriever.py` | `1_foundations/4_lab4.ipynb` + `2_openai/deep_research/search_agent.py` |
| `rag/ingest.py` | `1_foundations/4_lab4.ipynb` |
| `api/schemas.py` + `api/routes.py` | `4_langgraph/app.py` + `2_openai/deep_research/deep_research.py` |
| `main.py` | `4_langgraph/app.py` |
| `prompts/templates.py` | `4_langgraph/sidekick.py` lines 35–55 |
