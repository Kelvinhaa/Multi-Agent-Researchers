import json
from typing import Any, AsyncIterator

from agent.graph import graph

SCORE_RETRY_THRESHOLD = 0.7  # must match agent/graph.py route_after_critic


def format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def format_stage(node: str, data: dict[str, Any]) -> str:
    return format_sse({"type": "stage", "node": node, "data": data})


def _researcher_stage_data(update: dict) -> dict[str, Any]:
    messages = update.get("messages") or []
    last = messages[-1] if messages else None
    return {
        "docs": len(update.get("retrieved_docs") or []),
        "web_search": bool(getattr(last, "tool_calls", None)),
        "step": update.get("steps"),
    }


async def stream_research_events(query: str) -> AsyncIterator[str]:
    latest_retrieved_docs: list[dict] = []

    async for mode, payload in graph.astream(
        {"query": query}, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            message_chunk, metadata = payload
            if metadata.get("langgraph_node") == "writer" and message_chunk.content:
                yield format_sse({"type": "token", "content": message_chunk.content})

        elif mode == "updates":
            if "supervisor" in payload:
                yield format_stage(
                    "supervisor",
                    {"sub_queries": payload["supervisor"].get("sub_queries") or []},
                )

            if "researcher" in payload:
                update = payload["researcher"]
                yield format_stage("researcher", _researcher_stage_data(update))
                docs = update.get("retrieved_docs")
                if docs:
                    # Publish chunks the moment they exist rather than holding
                    # them until the run ends, so the UI can fill in mid-run.
                    latest_retrieved_docs = docs
                    yield format_sse({"type": "sources", "sources": docs})

            if "tools" in payload:
                results = payload["tools"].get("messages") or []
                yield format_stage("tools", {"results": len(results)})

            if "writer" in payload:
                yield format_stage("writer", {})

            if "critic" in payload:
                score = payload["critic"].get("score")
                feedback = payload["critic"].get("feedback")
                yield format_stage(
                    "critic",
                    {
                        "score": score,
                        "passed": score is not None and score >= SCORE_RETRY_THRESHOLD,
                        "feedback": feedback,
                    },
                )
                if score is not None and score < SCORE_RETRY_THRESHOLD:
                    yield format_sse({"type": "retry", "feedback": feedback})

    yield format_sse({"type": "sources", "sources": latest_retrieved_docs})
    yield format_sse({"type": "done"})
