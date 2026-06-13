from langchain.agents import tool

@tool
def web_search(query: str) -> str:
    """
    Search the live web for recent information on a given topic.
    """
    return "I'm sorry, I don't know how to do that."

@tool