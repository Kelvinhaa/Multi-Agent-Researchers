# RAG eval suite + synthetic corpus design

**Date:** 2026-07-28
**Status:** Approved, pending implementation
**Supersedes:** `2026-06-29-rag-eval-design.md`

## Context

The earlier eval spec was approved but only partially built: `eval/metrics.py` and `eval/golden_set.json` exist, while `eval/run.py` and `tests/test_eval_metrics.py` were never written. No quality number has ever been produced for this pipeline.

Two findings invalidate parts of that spec and motivate this one.

**RAGAS cannot be imported in this project.** `ragas` 0.4.3 (the latest release) imports `langchain_community.chat_models.vertexai`, a module deleted in `langchain-community` 0.4.2. This project runs the LangChain v1 line (`langchain-core` 1.4.6, `langchain` 1.3.9), and `langchain-community` is formally sunset. Pinning back far enough to restore that module would drag `langchain-core` below 0.4 and break LangGraph. `import ragas` fails outright, so the entire `eval/run.py` import block in the prior spec is unrunnable.

**The corpus makes any metric meaningless.** The index holds 34 chunks across 3 documents, two of which are the project's own `README.md` and `ENGINEERING_LOG.md`. With `top_k=5` against 3 documents, retrieval cannot fail. Worse, the third document (`probability_basics.md`) covers material `gpt-4o-mini` knows cold from pretraining, so the writer can answer correctly with zero retrieved context — end-to-end scores would measure the model's memory rather than the RAG pipeline.

## Decisions

| Decision | Choice |
|---|---|
| Judge implementation | Hand-rolled faithfulness / answer relevancy / context precision. `ragas` removed from dependencies |
| Judge LLM | `gpt-4o` — deliberately *not* the agent's `gpt-4o-mini`, to avoid self-preference bias |
| Corpus | ~20 synthetic PDFs: internal documents of a fictional company, "Harbourline Freight Systems" |
| Why synthetic | Fictional company facts are unknowable from pretraining, so retrieval is load-bearing; and ground truth is correct by construction rather than asserted |
| Corpus format | PDF, with page headers/footers and tables — realistic extraction artefacts |
| Corpus structure | 4 clusters × 5 documents; deliberately adjacent within cluster, separable across clusters |
| Retrieval metrics | `hit_rate_at_k` (binary), `recall_at_k` (fractional — new), `mrr` |
| Reporting | Aggregate *and* per-cluster breakdown |
| Unanswerable items | 3–4 golden items with no answer in the corpus, scored on abstention |
| Implementer | Claude writes; user reviews. This overrides the prior spec's teaching-mode decision for `eval/`, and extends to `rag/ingest.py` for this task only |
| Invocation | Standalone CLI (`python -m eval.run`), outside the mocked `pytest tests/` suite |

## Corpus design

### Company

**Harbourline Freight Systems** — a fictional mid-size Australian B2B logistics-software company. The Australian setting supplies natural factual density (GST, superannuation, state jurisdictions) while every figure is invented.

Where a policy touches a real statutory floor, it must sit *above* it with a company-specific number. "Harbourline provides 25 days annual leave" is unknowable to any model; the statutory 20 is not. No document may state a real-world legal entitlement as a bare fact.

### Documents

| Cluster | Documents |
|---|---|
| `people` | Leave policy, parental leave, remote work, performance & promotion, code of conduct |
| `finance` | Expense reimbursement, travel policy, corporate card, procurement approval, invoicing & payment terms |
| `engineering` | Incident response runbook, on-call rotation, release & deployment, access control, data retention |
| `commercial` | Discount approval matrix, SLA & service credits, refund policy, data processing, partner program |

Within a cluster, documents are deliberately adjacent — expense reimbursement, travel policy, and corporate card all specify spending limits — so a query about meal allowances must select the right one of three plausible neighbours. Across clusters they are trivially separable. Reporting both aggregate and per-cluster retrieval metrics exposes this gap, which a single aggregate number would hide.

### Authoring rules

1. **Canonical facts sheet first.** Every number lives in one source of truth at `docs/corpus_facts.md`, from which all 20 documents are derived. Without this the expense policy says a $75 meal cap while the travel policy says $80, and golden items become unanswerable. This file is internal scaffolding: it must live *outside* `docs/corpus/` so it is never ingested.
2. **Planted near-misses.** Contractor expense limits differ from employee limits and live in different documents. P1/P2 response times differ between the incident runbook and the SLA. These are what a weak retriever confuses, and what makes `recall_at_k` move.
3. **Facts stated in prose, not tables alone.** Tables are included for realism, but any fact a golden item depends on must also appear in prose, so a table-extraction failure degrades rather than silently corrupts ground truth.

### Generation

Rendered with `reportlab` (pure Python, no system dependencies): page headers, footers carrying document ID and version date, section headings, and tables. Generation script lives at `eval/generate_corpus.py`, is run once, and its PDF output is committed.

## Ingestion changes

`rag/ingest.py` needs three changes:

- **PDF support.** Add `*.pdf` to `DOC_GLOBS` and parse with `pdfplumber`. `pypdf` is rejected: it flattens tables into scrambled text, which would corrupt ground truth invisibly.
- **`--reset` flag.** Record IDs are `{stem}-{i}`, so re-ingesting a changed corpus leaves orphaned records forever — the existing 34 records would survive and be retrieved as stale context. `--reset` clears the namespace before upserting.
- **Batching.** `upsert_records` is currently called with an entire document's chunks at once. Pinecone caps batch size, and 20 PDFs will produce far more chunks than the current 3-document corpus. Batch at 96 records.

Ingest target is `docs/corpus/`, not `docs/`. `iter_doc_paths` uses `rglob`, so the command documented in `CLAUDE.md` (`python -m rag.ingest docs/`) would now also ingest `docs/superpowers/specs/*.md` — polluting the index with design documents. `CLAUDE.md` must be updated to `python -m rag.ingest docs/corpus/`.

## Eval architecture

```
docs/
  corpus_facts.md        # canonical figures — sibling of corpus/, never ingested
  corpus/                # 20 generated PDFs
eval/
  generate_corpus.py     # one-shot PDF generator
  golden_set.json        # ~20 items, rewritten against the new corpus
  metrics.py             # pure retrieval metrics, no I/O
  judges.py              # LLM-judged answer metrics
  run.py                 # CLI: both passes, aggregation, report
  results/               # gitignored, timestamped JSON
tests/
  test_eval_metrics.py   # pure-function + mocked-judge tests, joins the fast suite
```

Two passes per golden item, sharing one dataset:

- **Retrieval pass** — `rag.retriever.retrieve(query, top_k=5)` directly, isolating RAG from agent orchestration. Extract `source` per hit, score against `expected_sources`.
- **End-to-end pass** — `agent.graph.graph.ainvoke({"query": ...})`, exercising supervisor → researcher → writer → critic. `final_state["report"]` is the answer; `final_state["retrieved_docs"]` supplies contexts to the judges.

### `eval/golden_set.json`

```json
{
  "id": "fin-003",
  "query": "What is the daily meal allowance for interstate travel?",
  "cluster": "finance",
  "expected_sources": ["travel_policy"],
  "reference_answer": "...",
  "answerable": true
}
```

`expected_sources` matches the `source` field written by `rag/ingest.py` — the filename stem — so it survives re-chunking. Unanswerable items set `"answerable": false` and `"expected_sources": []`.

Composition: ~20 items, majority single-source; 2–3 multi-source; several near-miss items targeting within-cluster discrimination; 3–4 unanswerable.

### `eval/metrics.py`

Pure functions, no I/O:

```python
def hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1.0 if any retrieved source is expected, else 0.0. Empty -> 0.0."""

def recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """|set(retrieved) & set(expected)| / |set(expected)|. Empty expected -> 0.0."""

def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1 / (1-indexed rank of first match). 0.0 if none."""
```

`hit_rate_at_k` is the existing `recall_at_k` renamed. The old name was inaccurate — it takes no `k` and computes a binary hit, not recall. The new fractional `recall_at_k` is what makes multi-source items meaningful: today, finding one of two expected sources scores a perfect 1.0.

### `eval/judges.py`

All judges use `ChatOpenAI(model="gpt-4o").with_structured_output(...)`, matching the pattern already working in `agent/nodes.py`. No new LLM dependency.

**Faithfulness** — one structured call decomposes the answer into atomic claims and marks each supported/unsupported against the retrieved contexts. Score = supported / total. Zero extracted claims yields `None` and is excluded from the aggregate rather than scored 0.0.

**Answer relevancy** — generate 3 questions from the answer alone, embed them and the original query with `text-embedding-3-small`, and take the mean cosine similarity. This is harder to game than a direct rating prompt, and it is the metric that demonstrates understanding of what is being measured.

**Context precision** — judge each retrieved chunk relevant/not against the query and reference answer, in rank order, then compute average precision:

```
AP@k = sum(precision@i for i where chunk i is relevant) / total_relevant
precision@i = (relevant chunks in first i) / i
```

Zero relevant chunks yields 0.0. Rank-aware, so burying a good chunk at position 5 costs score.

**Abstention** (unanswerable items only) — a structured call classifies whether the answer declines to answer or asserts one. Correct refusal scores 1.0. Faithfulness, relevancy, and precision are skipped for these items; scoring a correct "I don't know" on answer relevancy would penalise the right behaviour.

### `eval/run.py`

CLI entry point. Fail-fast guard before constructing any client: if `golden_set.json` is missing or empty, exit immediately rather than burning API calls.

Both passes run per item under an `asyncio.Semaphore(4)` so ~20 items × ~5 judge calls do not trip rate limits. Each item is wrapped in try/except: record the error, continue, never let one bad item kill the run.

Output: a table of aggregates plus a per-cluster retrieval breakdown, and a JSON report at `eval/results/eval_<UTC-timestamp>.json` containing per-item records, aggregates, and failure count. `Path("eval/results").mkdir(parents=True, exist_ok=True)` at startup. Exit code 1 if any item errored.

## Error handling

| Failure | Handling |
|---|---|
| Missing/empty golden set | Exit before any API call |
| Single item raises in either pass | Record `error`, continue, count toward exit code 1 |
| Judge returns malformed output | `with_structured_output` retries; persistent failure is an item error |
| Zero claims extracted | `None`, excluded from aggregate |
| Zero relevant contexts | Context precision 0.0 — a real result, not an error |

## Testing strategy

- `eval/metrics.py` — real unit tests: hit on any expected source, miss on disjoint sets, empty retrieved, duplicates counted once, fractional recall on multi-source, MRR at first/third position, MRR no match.
- `eval/judges.py` — mocked-LLM tests for the aggregation maths, particularly AP@k, which is the most likely thing to be subtly wrong. Ranking sensitivity is asserted explicitly: the same relevant chunks at better ranks must score higher.
- `eval/run.py` — not unit tested. Orchestration meant to be run for real; verified manually.
- `eval/generate_corpus.py` — not unit tested. One-shot script; its output is committed and inspected.

## Dependencies

Added: `reportlab`, `pdfplumber`. Removed: `ragas` (and its transitive `datasets`, `instructor`, etc.).

## Non-goals

- Not CI-gated; no pass/fail threshold enforcement this iteration.
- Not fixing chunking. `chunk_size=512` / `chunk_overlap=50` stays as-is, even though PDF documents with headings and tables will likely expose its weaknesses. This spec establishes the measurement first; any chunking change is a follow-on with a before/after number.
- Not fixing the global-ranking bug in `agent/nodes.py`, where per-sub-query results are concatenated in sub-query order rather than sorted by score. Recorded here as the highest-value follow-on once a baseline exists.
- Not adding idempotent content-hash ingestion. `--reset` is the minimum needed to make re-ingest correct.
- Not hybrid or reranked retrieval.

## Verification

1. `uv sync` after dependency changes; confirm `uv run python -c "import reportlab, pdfplumber"` succeeds and `ragas` is gone.
2. `uv run python -m eval.generate_corpus` — 20 PDFs appear in `docs/corpus/`; spot-check two by eye for headers, tables, and readable text.
3. `PYTHONPATH=. uv run pytest tests/ -v` — full suite green, including new metric and judge tests, with no live calls.
4. `PYTHONPATH=. uv run python -m rag.ingest docs/corpus/ --reset` — reports chunks ingested; Pinecone `describe-index-stats` shows only the new records and no stale `README-*` or `probability_basics-*` IDs.
5. `PYTHONPATH=. uv run python -m eval.run` — table prints with aggregate and per-cluster rows, JSON lands in `eval/results/`, exit code 0 (`echo $?`).
6. Sanity-check the baseline: within-cluster recall should be visibly below cross-cluster recall. If both are 1.0, the near-misses were not planted aggressively enough and the corpus needs another pass.
