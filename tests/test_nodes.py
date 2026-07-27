import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import ToolMessage

from agent.nodes import critic, researcher, supervisor, writer


def test_researcher_retrieves_docs_for_each_sub_query_and_dedupes():
    state = {
        "query": "remote work effects",
        "sub_queries": ["health effects", "economic effects"],
    }
    docs_a = [
        {"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"}
    ]
    docs_b = [
        {"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"},
        {"id": "doc2-0", "score": 0.8, "text": "economic content", "source": "doc2"},
    ]
    docs_by_query = {"health effects": docs_a, "economic effects": docs_b}
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with (
        patch(
            "agent.nodes.retrieve", side_effect=lambda q: docs_by_query[q]
        ) as mock_retrieve,
        patch("agent.nodes.llm_with_tools") as mock_llm,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        result = asyncio.run(researcher(state))

    assert mock_retrieve.call_count == 2
    mock_retrieve.assert_any_call("health effects")
    mock_retrieve.assert_any_call("economic effects")
    assert result["retrieved_docs"] == [
        {"id": "doc1-0", "score": 0.9, "text": "health content", "source": "doc1"},
        {"id": "doc2-0", "score": 0.8, "text": "economic content", "source": "doc2"},
    ]


def test_researcher_retrieves_sub_queries_concurrently():
    state = {
        "query": "remote work effects",
        "sub_queries": ["health effects", "economic effects", "social effects"],
    }
    fake_response = SimpleNamespace(content="", tool_calls=[])

    def slow_retrieve(sub_query):
        time.sleep(0.2)
        return [
            {"id": f"{sub_query}-0", "score": 0.9, "text": sub_query, "source": "doc"}
        ]

    with (
        patch("agent.nodes.retrieve", side_effect=slow_retrieve),
        patch("agent.nodes.llm_with_tools") as mock_llm,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        start = time.monotonic()
        asyncio.run(researcher(state))
        elapsed = time.monotonic() - start

    assert elapsed < 0.35


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
    fake_docs = [
        {"id": "doc1-0", "score": 0.9, "text": "alpha content", "source": "doc1"}
    ]
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with (
        patch("agent.nodes.retrieve", return_value=fake_docs) as mock_retrieve,
        patch("agent.nodes.llm_with_tools") as mock_llm,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        result = asyncio.run(researcher(state))

    mock_retrieve.assert_called_once_with("what is alpha?")
    assert result["retrieved_docs"] == fake_docs
    assert result["messages"] == [fake_response]

    messages_sent = mock_llm.ainvoke.call_args[0][0]
    assert "alpha content" in messages_sent[0].content


def test_researcher_passes_message_history_to_llm_so_it_sees_prior_tool_results():
    prior_tool_message = ToolMessage(
        content="Tavily result: alpha is a concept", tool_call_id="call_1"
    )
    state = {
        "query": "what is alpha?",
        "sub_queries": ["what is alpha?"],
        "retrieved_docs": [
            {"id": "doc1-0", "score": 0.9, "text": "alpha content", "source": "doc1"}
        ],
        "messages": [prior_tool_message],
    }
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with patch("agent.nodes.llm_with_tools") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        asyncio.run(researcher(state))

    messages_sent = mock_llm.ainvoke.call_args[0][0]
    assert prior_tool_message in messages_sent


def test_researcher_skips_retrieval_when_docs_already_retrieved():
    existing_docs = [
        {"id": "doc1-0", "score": 0.9, "text": "alpha content", "source": "doc1"}
    ]
    state = {
        "query": "what is alpha?",
        "sub_queries": ["what is alpha?"],
        "retrieved_docs": existing_docs,
    }
    fake_response = SimpleNamespace(content="", tool_calls=[])

    with (
        patch("agent.nodes.retrieve") as mock_retrieve,
        patch("agent.nodes.llm_with_tools") as mock_llm,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)
        result = asyncio.run(researcher(state))

    mock_retrieve.assert_not_called()
    assert result["retrieved_docs"] == existing_docs


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


def test_writer_includes_web_search_results_from_message_history():
    tool_message = ToolMessage(
        content="Tavily result: alpha is a concept", tool_call_id="call_1"
    )
    state = {
        "retrieved_docs": [],
        "report": None,
        "messages": [tool_message],
    }
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = writer(state)

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "Tavily result: alpha is a concept" in prompt_used
    assert result == {"report": "generated report"}


def test_writer_does_not_crash_on_first_pass_with_no_prior_report():
    state = {"retrieved_docs": [{"text": "alpha content", "source": "doc1"}]}
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        result = writer(state)

    assert result == {"report": "generated report"}


def test_writer_is_given_the_user_query_so_it_answers_the_right_question():
    """Without the query the writer summarises whatever context it receives —
    which is how off-topic retrieved chunks ended up half the report."""
    state = {
        "query": "What are the newest AI chip announcements?",
        "retrieved_docs": [{"text": "how to run docker build", "source": "README"}],
    }
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        writer(state)

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "What are the newest AI chip announcements?" in prompt_used


def test_writer_is_told_to_ignore_context_that_does_not_serve_the_query():
    state = {
        "query": "what is alpha?",
        "retrieved_docs": [{"text": "x", "source": "d"}],
    }
    fake_response = SimpleNamespace(content="generated report")

    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_response
        writer(state)

    prompt_used = mock_llm.invoke.call_args[0][0].lower()
    assert "ignore" in prompt_used


def test_critic_is_given_the_user_query_so_it_can_judge_relevance():
    """The critic scored an off-topic report 0.85 because it was never shown
    the question the report was supposed to answer."""
    state = {
        "query": "What are the newest AI chip announcements?",
        "report": "Here is how to install the project with docker.",
    }
    fake_result = SimpleNamespace(score=0.2, feedback="off topic")

    with patch("agent.nodes.critic_llm") as mock_llm:
        mock_llm.invoke.return_value = fake_result
        result = critic(state)

    prompt_used = mock_llm.invoke.call_args[0][0]
    assert "What are the newest AI chip announcements?" in prompt_used
    assert result == {"score": 0.2, "feedback": "off topic"}
