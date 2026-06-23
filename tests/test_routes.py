from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.routes import app, graph


def test_research_endpoint_returns_report_and_sources():
    fake_result = {
        "report": "Alpha is a concept. [doc1-0]",
        "retrieved_docs": [
            {"id": "doc1-0", "score": 0.92, "text": "alpha content", "source": "doc1"}
        ],
    }

    with patch.object(graph, "ainvoke", new=AsyncMock(return_value=fake_result)) as mock_ainvoke:
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
