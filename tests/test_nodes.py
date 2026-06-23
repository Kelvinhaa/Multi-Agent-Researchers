from types import SimpleNamespace
from unittest.mock import patch

from agent.nodes import researcher, writer


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
