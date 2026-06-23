from unittest.mock import MagicMock, patch

from rag.embedder import TEXT_FIELD
from rag.ingest import build_records, chunk_text, ingest_path


def test_chunk_text_splits_long_text_into_overlapping_chunks():
    text = "word " * 300  # ~1500 chars, well over chunk_size=512
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

    assert len(chunks) > 1
    assert all(len(chunk) <= 512 for chunk in chunks)


def test_chunk_text_returns_single_chunk_for_short_text():
    chunks = chunk_text("short document", chunk_size=512, chunk_overlap=64)

    assert chunks == ["short document"]


def test_build_records_assigns_ids_and_text_field():
    records = build_records("doc1", ["chunk one", "chunk two"])

    assert records == [
        {"_id": "doc1-0", TEXT_FIELD: "chunk one", "source": "doc1"},
        {"_id": "doc1-1", TEXT_FIELD: "chunk two", "source": "doc1"},
    ]


def test_ingest_path_upserts_chunks_for_each_doc(tmp_path):
    (tmp_path / "a.md").write_text("alpha content")
    (tmp_path / "b.txt").write_text("beta content")

    fake_index = MagicMock()
    with patch("rag.ingest.ensure_index", return_value=fake_index):
        total = ingest_path(str(tmp_path))

    assert total == 2
    assert fake_index.upsert_records.call_count == 2
