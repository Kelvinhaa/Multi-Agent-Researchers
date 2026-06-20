import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embedder import NAMESPACE, TEXT_FIELD
from vectorstore.client import ensure_index

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
DOC_GLOBS = ("*.txt", "*.md")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_records(source: str, chunks: list[str]) -> list[dict]:
    return [
        {"_id": f"{source}-{i}", TEXT_FIELD: chunk, "source": source}
        for i, chunk in enumerate(chunks)
    ]


def iter_doc_paths(docs_dir: Path):
    for pattern in DOC_GLOBS:
        yield from docs_dir.rglob(pattern)


def ingest_path(docs_dir: str) -> int:
    index = ensure_index()
    total = 0
    for path in iter_doc_paths(Path(docs_dir)):
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        records = build_records(path.stem, chunks)
        if records:
            index.upsert_records(namespace=NAMESPACE, records=records)
            total += len(records)
    return total


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m rag.ingest <docs_dir>")
    print(f"Ingested {ingest_path(sys.argv[1])} chunks")
