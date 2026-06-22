from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.tools import tools
from agent.state import AgentState
from agent.nodes import supervisor, researcher, writer, critic


def route_after_critic(state: AgentState) -> str:
    if state.get("score") and state["score"] >= 0.7:
        return "END"
    return "researcher"

def route_after_supervisor(state: AgentState) -> str:
    return state["next"]

def route_after_researcher(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    else:
        return "writer"

def build_graph():

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("writer", writer)
    graph_builder.add_node("critic", critic)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    # Unconditional edges, always go to next node
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges("supervisor", route_after_supervisor, {"researcher": "researcher"})
    graph_builder.add_conditional_edges("researcher", route_after_researcher, {"tools": "tools", "writer": "writer"})
    graph_builder.add_edge("writer", "critic")
    graph_builder.add_edge("tools", "researcher")
   
    graph_builder.add_conditional_edges("critic", route_after_critic, {"researcher": "researcher", "END": END})

    return graph_builder.compile()

graph = build_graph()

    