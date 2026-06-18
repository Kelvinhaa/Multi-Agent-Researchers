from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.tools import tools
from dotenv import load_dotenv

load_dotenv(override=True)


# Pydantic model for the critic's output
class CriticOutput(BaseModel):
    feedback: str = Field(description="...")
    score: float = Field(description="0.0 to 1.0")
    should_retry: bool = Field(description="...")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_with_tools = llm.bind_tools(tools)
critic_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(CriticOutput)

def researcher(state: AgentState) -> dict:
    prompt = f"""You are a researcher agent. You are responsible for retrieving the context based on the user's query.
              The user's query is: {state['query']}
              The retrieved docs are: {state['retrieved_docs']}
              """
    if state.get("feedback"):
        prompt += f"\nPrevious attempt was rejected. Feedback: {state['feedback']}"

    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

def supervisor(state: AgentState) -> dict:
    prompt = """You are a supervisor agent. You are responsible for decomposing the user's query into sub-tasks and routing the agent to the appropriate node.
              You decide what to do first. """
    
    return {"next": "researcher"}

def writer(state: AgentState) -> dict:
    prompt = f"""You are a writer agent. You are responsible for writing the report based on the retrieved context.
              The retrieved context is: {state['retrieved_context']}
              The report is: {state['report']}
              """
    response = llm.invoke(prompt)
    return {"report": response.content}

def critic(state: AgentState) -> dict:
    prompt = f"""You are a critic agent. You are responsible for critiquing the report and deciding if it is good enough.
              The report is: {state['report']}
              """
    result = critic_llm.invoke(prompt)
    return {"score": result.score, "feedback": result.feedback}
    