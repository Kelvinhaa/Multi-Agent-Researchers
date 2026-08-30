from pinecone.exceptions import NotFoundException

from rag.embedder import TEXT_FIELD, NAMESPACE
from vectorstore.client import get_index


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    try:
        # Pinecone resolves the index host when the handle is created, so a
        # missing index 404s here rather than on search().
        index = get_index()
        response = index.search(
            namespace=NAMESPACE,
            top_k=top_k,
            inputs={"text": query},
            fields=[TEXT_FIELD, "source"],
        )
    except NotFoundException:
        # The index only exists once rag.ingest has run. Skipping ingestion is a
        # supported setup (web search only), so treat a missing index as an
        # empty knowledge base rather than failing the whole request.
        return []

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.fields[TEXT_FIELD],
            "source": hit.fields["source"],
        }
        for hit in response.result.hits
    ]
