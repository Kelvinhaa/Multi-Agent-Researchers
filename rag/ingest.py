import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embedder import NAMESPACE, TEXT_FIELD
from vectorstore.client import ensure_index

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
DOC_GLOBS = ("*.txt", "*.md", "*.pdf")
BATCH_SIZE = 96


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def build_records(source: str, chunks: list[str]) -> list[dict]:
    return [
        {"_id": f"{source}-{i}", TEXT_FIELD: chunk, "source": source}
        for i, chunk in enumerate(chunks)
    ]


def iter_doc_paths(docs_dir: Path):
    for pattern in DOC_GLOBS:
        yield from docs_dir.rglob(pattern)


def read_document(path: Path) -> str:
    """Read a document to plain text, dispatching on file extension.

    PDFs go through pdfplumber rather than pypdf: pypdf flattens tables into
    scrambled text, which would silently corrupt retrieved context.

    Ingest runs unattended over a growing corpus, so a single malformed PDF
    must not abort the whole batch. If pdfplumber can't open or read the
    file, the failure is reported to stderr with the offending path and an
    empty string is returned — the document contributes zero chunks and is
    effectively skipped, rather than the run dying with an uncaught
    low-level pdfminer/pdfplumber exception.
    """
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    parts.append(text)
                for table in page.extract_tables() or []:
                    for row in table:
                        cells = [c.strip() for c in row if c and c.strip()]
                        if cells:
                            parts.append(" | ".join(cells))
    except Exception as exc:
        print(f"Skipping unreadable PDF {path}: {exc}", file=sys.stderr)
        return ""

    return "\n".join(parts)


def batched(records: list[dict], size: int = BATCH_SIZE) -> Iterator[list[dict]]:
    """Yield records in batches no larger than `size` — Pinecone caps upsert size."""
    for start in range(0, len(records), size):
        yield records[start : start + size]


def reset_namespace(index) -> None:
    """Delete every record in the namespace.

    Record IDs are `{stem}-{i}`, so a shrunk or renamed document would otherwise
    leave orphaned chunks that get retrieved as stale context forever.
    """
    index.delete(delete_all=True, namespace=NAMESPACE)


def ingest_path(docs_dir: str, reset: bool = False) -> int:
    index = ensure_index()
    if reset:
        reset_namespace(index)

    total = 0
    for path in iter_doc_paths(Path(docs_dir)):
        chunks = chunk_text(read_document(path))
        records = build_records(path.stem, chunks)
        for batch in batched(records):
            index.upsert_records(namespace=NAMESPACE, records=batch)
            total += len(batch)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument("docs_dir", help="Directory to ingest, e.g. docs/corpus/")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing records in the namespace before ingesting.",
    )
    args = parser.parse_args()
    print(f"Ingested {ingest_path(args.docs_dir, reset=args.reset)} chunks")
