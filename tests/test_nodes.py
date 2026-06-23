from types import SimpleNamespace
from unittest.mock import call, patch

from agent.nodes import researcher, supervisor, writer


def test_researcher_retrieves_docs_for_each_sub_query_and_dedupes():
    state = {
        "query": "remote work effects",
        "sub_queries": ["health effects", "economic effects"],
    }
    docs_a = [{"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"}]
    docs_b = [
        {"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"},
        {"id": "doc2-0", "score": 0.8, "text": "economic content", "source": "doc2"},
    ]
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with patch("agent.nodes.retrieve", side_effect=[docs_a, docs_b]) as mock_retrieve, \
         patch("agent.nodes.llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = researcher(state)

    assert mock_retrieve.call_args_list == [call("health effects"), call("economic effects")]
    assert result["retrieved_docs"] == [
        {"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"},
        {"id": "doc2-0", "score": 0.8, "text": "economic content", "source": "doc2"},
    ]


def test_supervisor_decomposes_query_and_sets_next():
    state = {"query": "What are the health and economic effects of remote work?"}
    fake_result = SimpleNamespace(
        sub_queries=["health effects of remote work", "economic effects of remote work"]
    )

    with patch("agent.nodes.supervisor_llm") as mock_llm:
        mock_llm.invoke.return_value = fake_result
        result = supervisor(state)

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "What are the health and economic effects of remote work?" in prompt_used
    assert result == {
        "sub_queries": fake_result.sub_queries,
        "next": "researcher",
    }


def test_researcher_retrieves_docs_and_populates_state():
    state = {"query": "what is alpha?"}
    fake_docs = [{"id": "doc1-0", "score": 0.9, "text": "alpha content", "source": "doc1"}]
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with patch("agent.nodes.retrieve", return_value=fake_docs) as mock_retrieve, \
         patch("agent.nodes.llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = researcher(state)

    mock_retrieve.assert_called_once_with("what is alpha?")
    assert result["retrieved_docs"] == fake_docs
    assert result["messages"] == [fake_response]

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "alpha content" in prompt_used


def test_writer_includes_retrieved_docs_in_prompt():
    state = {
        "retrieved_docs": [{"text": "alpha content", "source": "doc1"}],
        "report": None,
    }
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = writer(state)

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "alpha content" in prompt_used
    assert result == {"report": "generated report"}


def test_writer_does_not_crash_on_first_pass_with_no_prior_report():
    state = {"retrieved_docs": [{"text": "alpha content", "source": "doc1"}]}
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = writer(state)

    assert result == {"report": "generated report"}
