"""Standalone RAG evaluation CLI.

Runs two independent passes per golden item — retrieval-only, and full
end-to-end through the agent graph — then scores both. Needs live Pinecone and
live OpenAI, so it deliberately sits outside the mocked pytest suite.

Usage: PYTHONPATH=. uv run python -m eval.run
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler

from agent.graph import graph
from eval.judges import (
    score_abstention,
    score_answer_relevancy,
    score_context_precision,
    score_faithfulness,
)
from eval.metrics import hit_rate_at_k, mrr, recall_at_k
from rag.retriever import retrieve

GOLDEN_SET = Path("eval/golden_set.json")
RESULTS_DIR = Path("eval/results")
TOP_K = 5
CONCURRENCY = 4

# USD per 1M tokens, (input, output). Verified 2026-07-28 against
# https://developers.openai.com/api/docs/pricing — re-check before quoting
# these figures anywhere, model pricing moves.
MODEL_PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}

METRIC_KEYS = (
    "hit_rate",
    "recall",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "abstention",
)

OPERATIONAL_KEYS = ("steps", "latency_s", "tokens", "cost_usd")


def token_cost(usage: dict) -> float:
    """USD cost from a UsageMetadataCallbackHandler.usage_metadata mapping.

    Keys are dated model IDs like "gpt-4o-mini-2024-07-18", which start with
    both "gpt-4o" and "gpt-4o-mini". Matching the LONGEST prefix is mandatory —
    the shorter match would bill mini traffic at 16x its real rate.
    Unknown models contribute nothing rather than guessing a price.
    """
    total = 0.0
    for model, counts in usage.items():
        candidates = [p for p in MODEL_PRICES if model.startswith(p)]
        if not candidates:
            continue
        input_price, output_price = MODEL_PRICES[max(candidates, key=len)]
        total += counts.get("input_tokens", 0) / 1_000_000 * input_price
        total += counts.get("output_tokens", 0) / 1_000_000 * output_price
    return total


def aggregate(records: list[dict]) -> dict:
    """Mean of each metric over scored items, overall and per cluster.

    None scores are excluded rather than counted as zero — a metric that could
    not be computed is absent data, not a failure.
    """
    scored = [r for r in records if not r.get("error")]

    def means(rows: list[dict]) -> dict:
        out = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                out[key] = sum(values) / len(values)
        return out

    def op_means(rows: list[dict]) -> dict:
        out = {}
        for key in OPERATIONAL_KEYS:
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                out[key] = sum(values) / len(values)
        return out

    clusters = sorted({r["cluster"] for r in scored})
    return {
        "overall": means(scored),
        "by_cluster": {
            c: means([r for r in scored if r["cluster"] == c]) for c in clusters
        },
        "operational": op_means(scored),
        "total_cost_usd": sum(r.get("cost_usd") or 0.0 for r in scored),
        "scored": len(scored),
        "failed": len(records) - len(scored),
    }


async def evaluate_item(item: dict, semaphore: asyncio.Semaphore) -> dict:
    """Run both passes for one golden item. Never raises — errors are recorded."""
    record = {
        "id": item["id"],
        "cluster": item["cluster"],
        "query": item["query"],
        "error": None,
    }

    async with semaphore:
        try:
            hits = await asyncio.to_thread(retrieve, item["query"], TOP_K)
            sources = [h["source"] for h in hits]
            record["retrieved_sources"] = sources

            if item["answerable"]:
                expected = item["expected_sources"]
                record["hit_rate"] = hit_rate_at_k(sources, expected)
                record["recall"] = recall_at_k(sources, expected)
                record["mrr"] = mrr(sources, expected)

            # Only the agent run is instrumented — judge calls construct their
            # own clients without this handler, so grading cost never inflates
            # the cost of serving the query.
            usage = UsageMetadataCallbackHandler()
            started = time.perf_counter()
            state = await graph.ainvoke(
                {"query": item["query"]}, config={"callbacks": [usage]}
            )
            record["latency_s"] = time.perf_counter() - started
            record["steps"] = state.get("steps", 0)
            record["cost_usd"] = token_cost(usage.usage_metadata)
            record["tokens"] = sum(
                c.get("total_tokens", 0) for c in usage.usage_metadata.values()
            )

            answer = state.get("report") or ""
            contexts = [d["text"] for d in (state.get("retrieved_docs") or [])]
            record["answer"] = answer

            if item["answerable"]:
                record["faithfulness"] = await score_faithfulness(answer, contexts)
                record["answer_relevancy"] = await score_answer_relevancy(
                    item["query"], answer
                )
                record["context_precision"] = await score_context_precision(
                    item["query"], item["reference_answer"], contexts
                )
            else:
                record["abstention"] = await score_abstention(answer)

        except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
            record["error"] = f"{type(e).__name__}: {e}"

    status = "!" if record["error"] else "."
    print(status, end="", flush=True)
    return record


def _print_report(summary: dict) -> None:
    def row(label: str, metrics: dict) -> str:
        cells = "  ".join(f"{k}={metrics[k]:.3f}" for k in METRIC_KEYS if k in metrics)
        return f"  {label:<14} {cells}"

    print("\n\n=== RAG eval ===")
    print(f"scored {summary['scored']}, failed {summary['failed']}\n")
    print(row("OVERALL", summary["overall"]))
    print()
    for cluster, metrics in summary["by_cluster"].items():
        print(row(cluster, metrics))

    ops = summary["operational"]
    if ops:
        print(f"\n  OPERATIONAL  (concurrency={summary['concurrency']})")
        print(f"    steps/run     {ops.get('steps', 0):.2f}")
        print(f"    latency/run   {ops.get('latency_s', 0):.2f}s")
        print(f"    tokens/run    {ops.get('tokens', 0):.0f}")
        print(f"    cost/run      ${ops.get('cost_usd', 0):.4f}")
        print(f"    total cost    ${summary['total_cost_usd']:.4f}")
        if summary["concurrency"] > 1:
            print("    NOTE: latency is contended at this concurrency.")
            print("          Use --sequential for comparable figures.")


async def main(concurrency: int = CONCURRENCY) -> int:
    if not GOLDEN_SET.exists():
        raise SystemExit(f"Golden set not found at {GOLDEN_SET}")

    items = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    if not items:
        raise SystemExit("Golden set is empty — nothing to evaluate")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    print(f"Evaluating {len(items)} items at concurrency {concurrency}", flush=True)
    records = await asyncio.gather(*(evaluate_item(i, semaphore) for i in items))

    summary = aggregate(list(records))
    # Latency is only comparable across runs at the same concurrency, so the
    # setting travels with the numbers rather than living in someone's memory.
    summary["concurrency"] = concurrency
    _print_report(summary)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"eval_{stamp}.json"
    out.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written to {out}")

    for record in records:
        if record["error"]:
            print(f"  ERROR {record['id']}: {record['error']}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite.")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run items one at a time. Latency figures are only comparable "
        "between runs at the same concurrency; use this for clean numbers.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(1 if args.sequential else CONCURRENCY)))
