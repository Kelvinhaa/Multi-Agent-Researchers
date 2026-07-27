# Live demo frontend design

**Date:** 2026-07-27
**Status:** Approved, pending implementation

## Context

The project has no user-facing surface — only `curl`-able endpoints. For a portfolio/recruiter audience the multi-agent architecture is the differentiator, but nothing makes it *visible*: a plain chat UI over `/research/stream` would be indistinguishable from a single-model wrapper.

This spec adds a deployed single-page demo that renders the LangGraph run itself — node-by-node execution, timings, visit counts, and the critic's re-search loop — as the primary content, with the streamed report as its output.

The visual design is imported from the Claude Design project **"Multi-agent RAG pipeline UI"** (`976c5afc-8232-42b8-9e90-5704f4d2d329`), file `Research Console.dc.html`, option **1b — graph instrument**. Its design system is Nocturne (`_ds/nocturne-a0cff356-.../styles.css`).

## Decisions

| Decision | Choice |
|---|---|
| Layout | Design option **1b**: thin project header → live graph strip → 3-column grid (report / retrieved / eval) |
| Frontend stack | One self-contained `static/index.html`, inline CSS+JS, no framework, no build step |
| Serving | FastAPI serves it at `GET /` — one deploy unit, no CORS |
| Design tokens | Nocturne palette/typography copied inline (accent `#9184d9`, surface `#232532`, bg `#161826`, Inter + JetBrains Mono) |
| Prototype runtime | `support.js` / `<x-dc>` / `<sc-if>` are canvas-only — **not** ported. Real state comes from SSE events |
| Trace transport | Extend existing SSE with a new `stage` event; all four current event types unchanged |
| Node timings | Measured client-side as deltas between event arrivals (real observed elapsed time) |
| Deployment | Vercel Hobby, deployed by Claude from the session |
| Rate limiting | In-memory per-IP + global daily cap; best-effort on serverless |
| Eval numbers | Real measured values only, from a committed snapshot. Unmeasured metrics render as pending |
| Eval strengthening | Corpus expansion with distractor docs + re-measurement. **No `eval/*` code written by Claude** (teaching mode) |
| Cost metric | Dropped — token usage is not instrumented, so it cannot be shown honestly |

## Architecture

```
Vercel (one project, public URL)
┌──────────────────────────────────────────────┐
│ FastAPI (api/routes.py)                      │
│   GET  /            → static/index.html      │
│   GET  /static/*    → StaticFiles mount      │
│                       (incl. eval_summary.json)│
│   GET  /healthz     → liveness               │
│   POST /research         (rate-limited)      │
│   POST /research/stream  (rate-limited, SSE) │
└──────────────────────────────────────────────┘
```

`agent/*`, `rag/*`, and `eval/*` are **not modified**. All new code is in `api/`, `static/`, and deploy config.

## Components

### 1. `stage` events — `api/streaming.py`

`graph.astream(stream_mode=["messages","updates"])` already yields each node's output on completion. Today only `critic` and `researcher` updates are consumed. Add one new event type emitted per node completion:

| Node | Event payload |
|---|---|
| supervisor | `{type:"stage", node:"supervisor", data:{sub_queries:[...]}}` |
| researcher | `{type:"stage", node:"researcher", data:{docs:N, web_search:bool, step:n}}` |
| tools | `{type:"stage", node:"tools", data:{results:N}}` |
| writer | `{type:"stage", node:"writer"}` |
| critic | `{type:"stage", node:"critic", data:{score:float, passed:bool, feedback:str}}` |

`web_search` is derived from whether the researcher's emitted message carries tool calls. `step` comes from `state["steps"]`.

Existing `token`, `retry`, `sources`, and `done` events keep their exact current shape and ordering semantics. The two route tests asserting full event sequences are updated to include the new `stage` events.

**Running states are not on the wire** — LangGraph `updates` fires on completion. The frontend infers "running" by advancing to the next node when the previous one completes, and marks `writer` running on the first `token`.

### 2. `static/index.html` — the 1b layout

Sections, top to bottom:

**Header.** Project name, one-sentence description, tech tags (`LangGraph 0.6`, `FastAPI · SSE`, `Pinecone`, `Tavily`, `Docker`). Static.

**Query row.** Text input + Research button + 3 example queries drawn from the golden set.

**Graph strip.** Inline SVG DAG matching `agent/graph.py`: `supervisor → researcher → writer → critic → END`, with `tools` above `researcher` (bidirectional edges), and the critic→researcher loop-back edge. Per-node card shows name, elapsed time, visit count (`×N`), and a live detail line:

| Node | Detail line source |
|---|---|
| supervisor | `N sub-queries` from `stage.data.sub_queries.length` |
| researcher | `N chunks · gather×N` from `stage.data.docs` and sub-query count |
| writer | `N tok streamed` from counted `token` events |
| critic | score history, e.g. `0.62 → 0.91` |

Node states: pending (dim) → active (accent border + `pulseRing`) → done. The loop-back edge is dim until a `retry` event fires, then highlights and stays highlighted.

**Report column.** Streamed tokens with a blinking caret, minimal markdown rendering (headings, bold, lists, inline code), followed by a stat row of **measured** values: latency (wall-clock), TTFT (first `token`), tokens (count), steps (`n/6`).

**Retrieved column.** Chunk cards from the `sources` event — source name, similarity score, text snippet, accent-graded left border by rank.

**Eval column.** Bars fed from `static/eval_summary.json`. Measured metrics render with values; unmeasured ones render `—` with a `pending` label and a caption naming the corpus size and `n`.

Errors and 429s render inline. Responsive: below ~900px the 3-column grid stacks and the graph strip scrolls horizontally in its own container.

### 3. `api/ratelimit.py`

FastAPI dependency applied to both POST endpoints:

- Per-IP sliding window, default 5/hour, env `RATE_LIMIT_PER_IP_HOURLY`
- Global daily cap, default 40/day UTC, env `RATE_LIMIT_DAILY_GLOBAL`
- 429 + JSON `{detail: "..."}` the frontend surfaces inline
- Client IP from the left-most `X-Forwarded-For` entry, falling back to the socket peer

In-memory state; on serverless each warm instance counts independently, so limits are **best-effort**, not exact. The README notes that an OpenAI account usage limit is the real spend ceiling.

### 4. `static/eval_summary.json`

Committed snapshot of the last real eval run, read by the frontend:

```json
{
  "measured_at": "2026-07-27",
  "golden_set_n": 18,
  "corpus_docs": 41,
  "corpus_chunks": 512,
  "metrics": {
    "recall_at_5": 0.83,
    "mrr": 0.71,
    "retrieval_p50_ms": 1046,
    "faithfulness": null,
    "answer_relevancy": null,
    "context_precision": null
  }
}
```

`null` renders as pending. Values above are illustrative of the file's *shape*; the committed file carries whatever the re-measurement actually produces.

### 5. Corpus expansion + re-measurement

Current corpus is 3 documents / 34 chunks, so `recall@5 = 1.00` and `MRR = 1.00` are uninformative — retrieval picks the right source from 3 candidates.

Add ~20–40 **distractor documents** to `docs/corpus/` on adjacent-but-distinct topics (statistics, ML, systems). The existing 18-query `eval/golden_set.json` stays valid unchanged — its `expected_sources` still name the same three documents, but retrieval must now surface them against 40+ competitors. Same metric definition, materially harder test.

Then re-run `rag/ingest.py` and score with the user's existing `eval/metrics.py` via a scratchpad harness. Whatever numbers result are what ship.

**Constraint:** Claude writes no files under `eval/`. `golden_set.json`, `metrics.py`, and `run.py` remain the user's per the teaching-mode rule in CLAUDE.md and the 2026-06-29 eval spec. RAGAS metrics stay pending until the user writes `run.py`.

### 6. Deployment

- `main.py`: bind `port=int(os.getenv("PORT", 8000))` and set `forwarded_allow_ips="*"`. Remains the local-dev entry point.
- New adapter entrypoint + `vercel.json` using Vercel's Python/FastAPI preset. The repo's `api/` package name collides with Vercel's legacy functions-directory convention; resolve via explicit config, verified against current Vercel docs during implementation.
- Move `ragas` to a dev dependency group — it is imported only by `eval/`, never by the running app, and drags in `pyarrow`/`pandas`/`datasets`. Local `uv sync` behaviour for eval work is unchanged.
- `Dockerfile` and `render.yaml`-style container deployment are left intact as the containerization story; Vercel is the live demo.
- README gains a Live Demo section with the URL and a GIF recorded from the deployed page.

## Data flow

```
POST /research/stream
   │
   ├─ stage(supervisor) ──→ supervisor card done, sub-queries listed, researcher → active
   ├─ stage(researcher) ──→ researcher card done (chunks, ×N); next is tools if the
   │                        message carried tool calls, else writer
   ├─ stage(tools) ───────→ tools card done, researcher re-activates (×N increments)
   ├─ token ×N ───────────→ report streams; first token stops the TTFT clock
   ├─ stage(writer) ──────→ writer card done (tok count), critic → active
   ├─ stage(critic) ──────→ critic card done, score appended to history
   ├─ retry ──────────────→ loop-back edge highlights, researcher ×N increments
   ├─ sources ────────────→ retrieved column populates
   └─ done ───────────────→ clocks stop, stat row finalises
```

## Testing strategy

- **`api/streaming.py`** — extend the existing mocked route tests to assert the new `stage` events appear in the right order alongside unchanged `token`/`retry`/`sources`/`done`. No live calls.
- **`api/ratelimit.py`** — new unit tests: under limit passes, over limit 429s, window expiry resets, global cap independent of per-IP, `X-Forwarded-For` parsing.
- **`static/index.html`** — no unit tests (no build step, no framework). Verified by driving the real page in a browser via Playwright against a locally running server and confirming every node card, the loop-back edge, streaming report, sources, and eval bars render from real events.
- Full `pytest tests/` stays fast and fully mocked.

## Non-goals

- No chat history, multi-turn conversation, or persisted sessions.
- No auth or user accounts.
- No `agent/*`, `rag/*`, or `eval/*` modifications.
- No RAGAS metrics until the user writes `eval/run.py`.
- No per-query cost display — token usage is not instrumented.
- No exact distributed rate limiting (no Redis); best-effort in-memory is accepted.

## Verification

1. `PYTHONPATH=. uv run pytest tests/ -v` — all pass, including new stage-event and rate-limit tests.
2. Local server + Playwright: run a real query end to end, confirm each graph node transitions pending→active→done with live timings, report streams, sources populate.
3. Force a critic retry (low-threshold query or temporarily raised threshold) and confirm the loop-back edge highlights and `researcher ×N` increments.
4. Exceed the per-IP limit and confirm a 429 renders inline rather than as an unhandled error.
5. Re-ingest the expanded corpus, re-measure, and confirm `static/eval_summary.json` matches the printed run output.
6. Deploy to Vercel, run a real query against the public URL, confirm parity with local.
