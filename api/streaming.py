import json
from typing import Any, AsyncIterator

from agent.graph import graph

SCORE_RETRY_THRESHOLD = 0.7  # must match agent/graph.py route_after_critic


def format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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
            if "researcher" in payload:
                latest_retrieved_docs = (
                    payload["researcher"].get("retrieved_docs") or latest_retrieved_docs
                )
            if "critic" in payload:
                score = payload["critic"].get("score")
                if score is not None and score < SCORE_RETRY_THRESHOLD:
                    yield format_sse(
                        {"type": "retry", "feedback": payload["critic"].get("feedback")}
                    )

    yield format_sse({"type": "sources", "sources": latest_retrieved_docs})
    yield format_sse({"type": "done"})
