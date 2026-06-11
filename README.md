# AI Agent Researcher

An autonomous multi-agent research assistant that decomposes a query into sub-tasks, retrieves relevant context from a vector knowledge base, searches the live web, and synthesises a cited report. Built with LangGraph, FastAPI, and a RAG pipeline backed by Pinecone.

---

## How it works

A supervisor agent receives the user's query and breaks it into sub-tasks. Worker agents retrieve context from the internal knowledge base (via RAG) and the live web (via Tavily). A synthesis agent assembles the retrieved context into a structured report. A critique agent checks the output for gaps and triggers a re-search loop if confidence is low. The final answer streams back to the client token by token.

```
User query
    └── Supervisor (decomposes + routes)
            ├── Researcher (RAG + web search)
            ├── Writer (synthesis)
            └── Critic (quality gate + re-search loop)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 0.2.x |
| LLM | OpenAI GPT-4o via OpenAI SDK |
| Retrieval | RAG over Pinecone vector DB |
| Embeddings | OpenAI text-embedding-3-small |
| Web search | Tavily API |
| External tools | Model Context Protocol (MCP) |
| API server | FastAPI + uvicorn |
| Package manager | uv |
| Python | 3.12 |

---

## Project structure

```
ai-agent-researcher/
├── main.py                  # Entry point — starts FastAPI, compiles agent graph on startup
├── pyproject.toml           # Dependencies and project metadata
├── uv.lock                  # Pinned dependency versions (commit this)
├── .env                     # Secret keys — never commit
├── .python-version          # Pins Python 3.12
│
├── agent/
│   ├── __init__.py
│   ├── graph.py             # LangGraph StateGraph definition and compile()
│   ├── state.py             # AgentState TypedDict — shared memory across nodes
│   ├── nodes.py             # Node functions: supervisor, researcher, writer, critic
│   └── tools.py             # @tool definitions callable by the LLM
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py            # Load documents, chunk, embed, upsert to Pinecone
│   ├── embedder.py          # Wrapper around OpenAI embeddings API
│   └── retriever.py         # Semantic search against the vector index
│
├── vectorstore/
│   ├── __init__.py
│   └── client.py            # Pinecone client singleton
│
├── api/
│   ├── __init__.py
│   ├── routes.py            # FastAPI app and streaming /chat endpoint
│   └── schemas.py           # Pydantic request/response models
│
├── prompts/
│   ├── __init__.py
│   └── templates.py         # System prompt constants per agent node
│
├── mcp/
│   └── config.json          # MCP server declarations (filesystem, Gmail, etc.)
│
└── tests/
    └── test_agent.py        # Integration tests for the compiled graph
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- OpenAI API key — [platform.openai.com](https://platform.openai.com)
- Pinecone API key — [pinecone.io](https://pinecone.io)
- Tavily API key — [tavily.com](https://tavily.com)

---

## Setup

**1. Clone and enter the project**

```bash
git clone https://github.com/your-username/ai-agent-researcher.git
cd ai-agent-researcher
```

**2. Install dependencies**

uv handles the virtual environment and Python version automatically.

```bash
uv sync
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=research-agent
TAVILY_API_KEY=...
APP_ENV=development
```

**4. Create your Pinecone index**

Create a Pinecone index named `research-agent` with dimension `1536` (matching `text-embedding-3-small`) and cosine similarity metric.

**5. Ingest your knowledge base** (optional — skip to use web search only)

Place documents in `docs/` then run:

```bash
uv run python -m rag.ingest docs/
```

**6. Start the server**

```bash
PYTHONPATH=. uv run uvicorn api.routes:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Usage

**Send a research query**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest advances in long-context LLMs?"}'
```

**Stream the response**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "Compare transformer and state space model architectures"}' \
  --no-buffer
```

**Ingest a new document at runtime**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@path/to/document.pdf"
```


## Agent architecture

The graph is built in `agent/graph.py` using LangGraph's `StateGraph`. All nodes communicate via a shared `AgentState` TypedDict defined in `agent/state.py`.

```
Entry
  └── supervisor         Decomposes query, sets routing decision in state["next"]
        ├── researcher   Runs RAG retrieval + Tavily web search, populates state["retrieved_docs"]
        └── writer       Assembles retrieved context into a cited report
              └── critic Scores output quality; routes back to researcher if score < threshold
                    └── END
```

Conditional routing is handled by a single `route()` function that reads `state["next"]` — the supervisor writes this field to direct the graph at each step.

---

## RAG pipeline

Ingestion and retrieval are fully decoupled. Run `ingest.py` independently on a schedule or on demand; the agent's `retriever.py` only ever reads from the index.

**Chunking strategy:** `RecursiveCharacterTextSplitter` with `chunk_size=512` and `chunk_overlap=64`. Chunks are labelled with source path and page number in metadata so the writer node can cite them accurately.

**Retrieval:** Semantic similarity search using the embedded user query against the Pinecone index. Top-k results are returned with source metadata and passed directly into the writer node's context window.

---

## Development

**Add a dependency**

```bash
uv add <package-name>
```

**Run tests**

```bash
PYTHONPATH=. uv run pytest tests/ -v
```

**Format and lint**

```bash
uv run ruff format .
uv run ruff check .
```

---

## Deployment

The project is containerised with Docker. The image targets Python 3.12-slim and copies only the application source, keeping the image small.

```bash
docker build -t ai-agent-researcher .
docker run -p 8000:8000 --env-file .env ai-agent-researcher
```

For production, set `APP_ENV=production` and mount secrets via your cloud provider's secret manager rather than a `.env` file.

---

## Acknowledgements

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [FastAPI](https://fastapi.tiangolo.com), [Pinecone](https://pinecone.io), and [Tavily](https://tavily.com).