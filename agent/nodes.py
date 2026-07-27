import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from agent.state import AgentState
from agent.tools import tools
from rag.retriever import retrieve
from dotenv import load_dotenv

load_dotenv(override=True)


# Pydantic model for the critic's output
class CriticOutput(BaseModel):
    feedback: str = Field(description="...")
    score: float = Field(description="0.0 to 1.0")


# Pydantic model for the supervisor's output
class SupervisorOutput(BaseModel):
    sub_queries: list[str] = Field(
        description="2-4 focused sub-questions that together cover what's needed to answer the user's query"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_with_tools = llm.bind_tools(tools)
critic_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(CriticOutput)
supervisor_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(
    SupervisorOutput
)


async def researcher(state: AgentState) -> dict:
    sub_queries = state.get("sub_queries") or [state["query"]]
    docs = state.get("retrieved_docs") or []

    if not docs:
        results = await asyncio.gather(
            *(asyncio.to_thread(retrieve, sub_query) for sub_query in sub_queries)
        )

        seen_ids = set()
        for sub_query_docs in results:
            for doc in sub_query_docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    docs.append(doc)

    prompt = f"""You are a researcher agent. You are responsible for retrieving the context based on the user's query.
              The user's query is: {state["query"]}
              It was broken down into these sub-questions: {sub_queries}
              The retrieved docs are: {docs}
              """
    if state.get("feedback"):
        prompt += f"\nPrevious attempt was rejected. Feedback: {state['feedback']}"

    response = await llm_with_tools.ainvoke(
        [SystemMessage(content=prompt)] + (state.get("messages") or [])
    )
    return {
        "retrieved_docs": docs,
        "messages": [response],
        "steps": state.get("steps", 0) + 1,
    }


def supervisor(state: AgentState) -> dict:
    prompt = f"""You are a supervisor agent. Break the user's query into 2-4 focused,
              self-contained sub-questions that together cover what's needed to answer it.
              If the query is already narrow, return a single sub-question (a cleaned-up
              restatement of the query).
              The user's query is: {state["query"]}
              """

    result = supervisor_llm.invoke(prompt)
    return {"sub_queries": result.sub_queries, "next": "researcher"}


def writer(state: AgentState) -> dict:
    web_results = [
        m.content for m in (state.get("messages") or []) if isinstance(m, ToolMessage)
    ]

    prompt = f"""You are a writer agent. You write a cited report that answers the
              user's question.
              The user's question is: {state.get("query")}
              Answer that question and only that question. The retrieved context is
              ranked by similarity, and low-ranked context is often unrelated to the
              question — ignore anything that does not help answer it rather than
              summarising it.
              The retrieved context is: {state["retrieved_docs"]}
              """
    if web_results:
        prompt += f"\nWeb search results: {web_results}"
    if state.get("report"):
        prompt += f"\nPrevious draft: {state['report']}"

    response = llm.invoke(prompt)
    return {"report": response.content}


def critic(state: AgentState) -> dict:
    prompt = f"""You are a critic agent. You critique the report and decide if it is
              good enough.
              The user's question is: {state.get("query")}
              Score how well the report answers that question. A fluent, well-formed
              report that answers a different question, or pads itself with material
              unrelated to the question, is a low score.
              The report is: {state["report"]}
              """
    result = critic_llm.invoke(prompt)
    return {"score": result.score, "feedback": result.feedback}
