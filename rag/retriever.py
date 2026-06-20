from rag.embedder import TEXT_FIELD, NAMESPACE
from vectorstore.client import get_index

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    index = get_index()

    response = index.search(
        namespace=NAMESPACE,
        top_k=top_k,
        inputs={"text":query},
        fields=[TEXT_FIELD, "source"],
    )
    
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.fields[TEXT_FIELD],
            "source": hit.fields["source"],
        }
        for hit in response.result.hits
    ]
