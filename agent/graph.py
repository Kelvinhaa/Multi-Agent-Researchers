from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import supervisor, researcher, writer, critic

def route_after_critic(state: AgentState) ->str:
    if state.get("score") and state["score"] >= 0.7:
        return "END"
    return "reseacher"

def build_graph():

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("writer", writer)
    graph_builder.add_node("critic", critic)

    # Unconditional edges, always go to next node
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("supervisor", "researcher")
    graph_builder.add_edge("researcher", "writer")
    graph_builder.add_edge("writer", "critic")
   
    graph_builder.add_conditional_edges("critic", route_after_critic, {"researcher": "researcher", "END": END})

    return graph_builder.compile()

graph = build_graph()

    