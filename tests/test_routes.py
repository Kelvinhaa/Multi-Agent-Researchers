import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.routes import app, graph


def make_fake_astream(events):
    async def fake_astream(*args, **kwargs):
        for event in events:
            yield event

    return fake_astream


def parse_sse_events(text: str) -> list[dict]:
    return [
        json.loads(block[len("data: ") :])
        for block in text.split("\n\n")
        if block.strip().startswith("data: ")
    ]


def test_research_endpoint_returns_report_and_sources():
    fake_result = {
        "report": "Alpha is a concept. [doc1-0]",
        "retrieved_docs": [
            {"id": "doc1-0", "score": 0.92, "text": "alpha content", "source": "doc1"}
        ],
    }

    with patch.object(
        graph, "ainvoke", new=AsyncMock(return_value=fake_result)
    ) as mock_ainvoke:
        client = TestClient(app)
        response = client.post("/research", json={"query": "what is alpha?"})

    assert response.status_code == 200
    assert response.json() == {
        "report": "Alpha is a concept. [doc1-0]",
        "sources": [
            {"id": "doc1-0", "score": 0.92, "text": "alpha content", "source": "doc1"}
        ],
    }
    mock_ainvoke.assert_awaited_once_with({"query": "what is alpha?"})


def test_research_endpoint_rejects_empty_query():
    client = TestClient(app)
    response = client.post("/research", json={"query": ""})

    assert response.status_code == 422


def test_research_stream_happy_path_streams_tokens_then_sources_and_done():
    fake_events = [
        ("messages", (SimpleNamespace(content="Hello"), {"langgraph_node": "writer"})),
        ("messages", (SimpleNamespace(content=" world"), {"langgraph_node": "writer"})),
        (
            "updates",
            {
                "researcher": {
                    "retrieved_docs": [
                        {
                            "id": "doc1-0",
                            "score": 0.92,
                            "text": "alpha content",
                            "source": "doc1",
                        }
                    ]
                }
            },
        ),
        ("updates", {"critic": {"score": 0.9, "feedback": "Looks good"}}),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events == [
        {"type": "token", "content": "Hello"},
        {"type": "token", "content": " world"},
        {
            "type": "stage",
            "node": "researcher",
            "data": {"docs": 1, "web_search": False, "step": None},
        },
        {
            "type": "sources",
            "sources": [
                {
                    "id": "doc1-0",
                    "score": 0.92,
                    "text": "alpha content",
                    "source": "doc1",
                }
            ],
        },
        {
            "type": "stage",
            "node": "critic",
            "data": {"score": 0.9, "passed": True, "feedback": "Looks good"},
        },
        {
            "type": "sources",
            "sources": [
                {
                    "id": "doc1-0",
                    "score": 0.92,
                    "text": "alpha content",
                    "source": "doc1",
                }
            ],
        },
        {"type": "done"},
    ]


def test_research_stream_emits_two_retry_events_for_two_rejections_with_same_score():
    fake_events = [
        (
            "messages",
            (SimpleNamespace(content="Draft one"), {"langgraph_node": "writer"}),
        ),
        ("updates", {"critic": {"score": 0.4, "feedback": "fix it"}}),
        (
            "messages",
            (SimpleNamespace(content="Draft two"), {"langgraph_node": "writer"}),
        ),
        ("updates", {"critic": {"score": 0.4, "feedback": "still not enough"}}),
        (
            "messages",
            (SimpleNamespace(content="Draft three"), {"langgraph_node": "writer"}),
        ),
        ("updates", {"critic": {"score": 0.9, "feedback": "good"}}),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    retry_events = [e for e in events if e["type"] == "retry"]
    assert len(retry_events) == 2


def test_research_stream_endpoint_rejects_empty_query():
    client = TestClient(app)
    response = client.post("/research/stream", json={"query": ""})

    assert response.status_code == 422


def test_index_route_serves_the_demo_page():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ai-agent-researcher" in response.text


def test_stream_emits_supervisor_stage_with_sub_queries():
    fake_events = [
        ("updates", {"supervisor": {"sub_queries": ["what is alpha", "why alpha"]}}),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0] == {
        "type": "stage",
        "node": "supervisor",
        "data": {"sub_queries": ["what is alpha", "why alpha"]},
    }


def test_stream_emits_researcher_stage_with_doc_count_and_web_search_flag():
    message_with_tool_call = SimpleNamespace(tool_calls=[{"name": "web_search"}])
    fake_events = [
        (
            "updates",
            {
                "researcher": {
                    "retrieved_docs": [
                        {"id": "d0", "score": 0.9, "text": "x", "source": "doc1"},
                        {"id": "d1", "score": 0.8, "text": "y", "source": "doc2"},
                    ],
                    "messages": [message_with_tool_call],
                    "steps": 1,
                }
            },
        ),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0] == {
        "type": "stage",
        "node": "researcher",
        "data": {"docs": 2, "web_search": True, "step": 1},
    }


def test_researcher_stage_reports_no_web_search_when_message_has_no_tool_calls():
    message_without_tool_calls = SimpleNamespace(tool_calls=[])
    fake_events = [
        (
            "updates",
            {
                "researcher": {
                    "retrieved_docs": [],
                    "messages": [message_without_tool_calls],
                    "steps": 2,
                }
            },
        ),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0]["data"] == {"docs": 0, "web_search": False, "step": 2}


def test_stream_emits_sources_as_soon_as_the_researcher_finishes():
    """Sources used to arrive only in the final event, so the UI showed an
    empty 'retrieving' panel for the whole run even after chunks were in."""
    docs = [{"id": "d0", "score": 0.9, "text": "x", "source": "doc1"}]
    fake_events = [
        ("updates", {"researcher": {"retrieved_docs": docs, "steps": 1}}),
        ("messages", (SimpleNamespace(content="tok"), {"langgraph_node": "writer"})),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    # a sources event lands before the first token, not just at the end
    first_token = next(i for i, e in enumerate(events) if e["type"] == "token")
    early = [e for e in events[:first_token] if e["type"] == "sources"]
    assert early == [{"type": "sources", "sources": docs}]


def test_stream_emits_tools_stage_with_result_count():
    fake_events = [
        ("updates", {"tools": {"messages": [SimpleNamespace(content="r1")]}}),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0] == {"type": "stage", "node": "tools", "data": {"results": 1}}


def test_stream_emits_writer_stage():
    fake_events = [("updates", {"writer": {"report": "the report"}})]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0] == {"type": "stage", "node": "writer", "data": {}}


def test_critic_stage_reports_passed_true_when_score_meets_threshold():
    fake_events = [("updates", {"critic": {"score": 0.9, "feedback": "good"}})]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    stage_events = [e for e in parse_sse_events(response.text) if e["type"] == "stage"]
    assert stage_events[0] == {
        "type": "stage",
        "node": "critic",
        "data": {"score": 0.9, "passed": True, "feedback": "good"},
    }


def test_critic_stage_precedes_retry_event_when_score_below_threshold():
    fake_events = [("updates", {"critic": {"score": 0.4, "feedback": "thin"}})]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events[0] == {
        "type": "stage",
        "node": "critic",
        "data": {"score": 0.4, "passed": False, "feedback": "thin"},
    }
    assert events[1] == {"type": "retry", "feedback": "thin"}


def test_research_stream_emits_retry_event_when_critic_rejects():
    fake_events = [
        (
            "messages",
            (SimpleNamespace(content="Draft one"), {"langgraph_node": "writer"}),
        ),
        ("updates", {"critic": {"score": 0.4, "feedback": "Needs more detail"}}),
        (
            "messages",
            (SimpleNamespace(content="Draft two"), {"langgraph_node": "writer"}),
        ),
        (
            "updates",
            {
                "researcher": {
                    "retrieved_docs": [
                        {"id": "doc1-0", "score": 0.9, "text": "x", "source": "doc1"}
                    ]
                }
            },
        ),
        ("updates", {"critic": {"score": 0.85, "feedback": "Good now"}}),
    ]

    with patch.object(graph, "astream", new=make_fake_astream(fake_events)):
        client = TestClient(app)
        response = client.post("/research/stream", json={"query": "what is alpha?"})

    events = parse_sse_events(response.text)
    assert events == [
        {"type": "token", "content": "Draft one"},
        {
            "type": "stage",
            "node": "critic",
            "data": {"score": 0.4, "passed": False, "feedback": "Needs more detail"},
        },
        {"type": "retry", "feedback": "Needs more detail"},
        {"type": "token", "content": "Draft two"},
        {
            "type": "stage",
            "node": "researcher",
            "data": {"docs": 1, "web_search": False, "step": None},
        },
        {
            "type": "sources",
            "sources": [{"id": "doc1-0", "score": 0.9, "text": "x", "source": "doc1"}],
        },
        {
            "type": "stage",
            "node": "critic",
            "data": {"score": 0.85, "passed": True, "feedback": "Good now"},
        },
        {
            "type": "sources",
            "sources": [{"id": "doc1-0", "score": 0.9, "text": "x", "source": "doc1"}],
        },
        {"type": "done"},
    ]
