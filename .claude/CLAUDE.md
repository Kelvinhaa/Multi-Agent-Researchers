# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An autonomous multi-agent research assistant built with LangGraph, FastAPI, and a RAG pipeline backed by Pinecone. A supervisor agent decomposes queries into sub-tasks; worker agents retrieve context via RAG and Tavily web search; a writer synthesises a cited report; a critic quality-gates the output and triggers re-search if confidence is low. Responses stream token-by-token to the client.

## Commands

```bash
# Install dependencies (manages venv and Python version automatically)
uv sync

# Start the API server (with hot reload)
PYTHONPATH=. uv run uvicorn api.routes:app --reload

# Run tests
PYTHONPATH=. uv run pytest tests/ -v

# Run a single test file or test
PYTHONPATH=. uv run pytest tests/test_agent.py::test_name -v

# Lint and format
uv run ruff format .
uv run ruff check .

# Ingest documents into Pinecone
uv run python -m rag.ingest docs/

# Add a dependency
uv add <package-name>
```

## Environment variables

Copy `.env.example` to `.env` and populate:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=research-agent
TAVILY_API_KEY=...
APP_ENV=development
```

Pinecone index must be created with dimension `1536` (matching `text-embedding-3-small`) and cosine similarity.

## Architecture

### Agent graph (`agent/`)

The LangGraph `StateGraph` is compiled in `agent/graph.py`. All nodes share a single `AgentState` TypedDict (`agent/state.py`). Conditional routing reads `state["next"]` — the supervisor writes this field to direct the graph at each step.

**Node flow:**
```
supervisor → researcher → writer → critic → (END or back to researcher)
```

- `supervisor` — decomposes the query, sets `state["next"]`
- `researcher` — runs RAG retrieval + Tavily web search, populates `state["retrieved_docs"]`
- `writer` — assembles retrieved context into a cited report
- `critic` — scores output quality; routes back to researcher if score < threshold

Node functions live in `agent/nodes.py`. Tool definitions decorated with `@tool` are in `agent/tools.py`. System prompt constants per node are in `prompts/templates.py`.

### RAG pipeline (`rag/`)

Ingestion and retrieval are fully decoupled — `ingest.py` runs independently; `retriever.py` only reads. Chunking uses `RecursiveCharacterTextSplitter` with `chunk_size=512`, `chunk_overlap=64`. Chunks carry source path and page number metadata for citations.

### API (`api/`)

FastAPI app and streaming `/chat` endpoint in `api/routes.py`. Pydantic request/response models in `api/schemas.py`. Docs available at `http://localhost:8000/docs`.

### Vector store (`vectorstore/`)

`vectorstore/client.py` holds the Pinecone client singleton. Used by both `rag/ingest.py` and `rag/retriever.py`.

### MCP (`mcp/config.json`)

Declares external MCP server connections (filesystem, Gmail, etc.) used as tools by the agent.
