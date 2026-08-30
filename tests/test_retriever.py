from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag.embedder import TEXT_FIELD
from rag.retriever import retrieve


def _fake_response(hits):
    return SimpleNamespace(result=SimpleNamespace(hits=hits))


def test_retrieve_returns_parsed_hits():
    fake_hits = [
        SimpleNamespace(
            id="doc1-0",
            score=0.92,
            fields={TEXT_FIELD: "alpha content", "source": "doc1"},
        ),
        SimpleNamespace(
            id="doc2-0",
            score=0.81,
            fields={TEXT_FIELD: "beta content", "source": "doc2"},
        ),
    ]
    fake_index = MagicMock()
    fake_index.search.return_value = _fake_response(fake_hits)

    with patch("rag.retriever.get_index", return_value=fake_index):
        results = retrieve("what is alpha?", top_k=2)

    fake_index.search.assert_called_once_with(
        namespace="default",
        top_k=2,
        inputs={"text": "what is alpha?"},
        fields=[TEXT_FIELD, "source"],
    )
    assert results == [
        {"id": "doc1-0", "score": 0.92, "text": "alpha content", "source": "doc1"},
        {"id": "doc2-0", "score": 0.81, "text": "beta content", "source": "doc2"},
    ]
