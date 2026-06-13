from typing_extensions import TypedDict
from typing import Annotated, Optional, List, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]

    query: str
    next: Optional[str]
    retrieved_docs: Optional[List[dict]]
    report: Optional[str]
    feedback: Optional[str]
    score: Optional[float]
    

