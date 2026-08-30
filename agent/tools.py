import os
from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """Search the live web for current, time-sensitive information.

    Use this only when the user's request needs information that is recent
    or likely outside an internal knowledge base — news, prices, releases,
    or anything that may have changed after this model's training cutoff.
    Do not use it to answer questions already covered by retrieved internal
    documents; prefer those when they're sufficient.

    Args:
        query: A short, keyword-style search query (roughly 3-8 words), not
            a full question. Strip filler words and keep only the core
            entities/topic — e.g. "Pinecone integrated inference pricing",
            not "how much does Pinecone's integrated inference cost".

    Returns:
        The top results as plain text, one per result, each with its title,
        source URL, and a short content snippet, so the result can be cited.
    """
    try:
        response = client.search(
            query=query,
            include_answer="basic",
            search_depth="basic",
        )
    except Exception as e:
        return f"Web search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found for that query."

    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['url']}\n   {r['content']}"
        for i, r in enumerate(results, start=1)
    )


tools = [web_search]
