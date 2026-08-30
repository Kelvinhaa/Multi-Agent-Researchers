from pathlib import Path
from unittest.mock import MagicMock, patch

from eval.generate_corpus import render_pdf
from rag.embedder import TEXT_FIELD
from rag.ingest import (
    batched,
    build_records,
    chunk_text,
    ingest_path,
    read_document,
    reset_namespace,
)


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


def _write_pdf(tmp_path: Path, name: str, body: str) -> Path:
    src = tmp_path / f"{name}.md"
    src.write_text(
        f"""---
doc_id: HFS-TST-001
title: {name}
version: "1.0"
effective: 2026-01-01
---

# {name}

{body}
""",
        encoding="utf-8",
    )
    out = tmp_path / f"{name}.pdf"
    render_pdf(src, out)
    return out


def test_read_document_extracts_text_from_pdf(tmp_path):
    pdf = _write_pdf(tmp_path, "travel_policy", "The daily meal allowance is $85.")

    text = read_document(pdf)

    assert "meal allowance" in text
    assert "$85" in text


def test_read_document_reads_markdown_unchanged(tmp_path):
    md = tmp_path / "note.md"
    md.write_text("plain markdown body", encoding="utf-8")

    assert read_document(md) == "plain markdown body"


def test_batched_splits_records_into_size_capped_batches():
    records = [{"_id": str(i)} for i in range(200)]

    batches = list(batched(records, 96))

    assert [len(b) for b in batches] == [96, 96, 8]


def test_reset_namespace_deletes_all_records():
    fake_index = MagicMock()

    reset_namespace(fake_index)

    fake_index.delete.assert_called_once_with(delete_all=True, namespace="default")


def test_ingest_path_resets_namespace_when_requested(tmp_path):
    (tmp_path / "a.md").write_text("alpha content")
    fake_index = MagicMock()

    with patch("rag.ingest.ensure_index", return_value=fake_index):
        ingest_path(str(tmp_path), reset=True)

    fake_index.delete.assert_called_once_with(delete_all=True, namespace="default")


def test_ingest_path_does_not_reset_by_default(tmp_path):
    (tmp_path / "a.md").write_text("alpha content")
    fake_index = MagicMock()

    with patch("rag.ingest.ensure_index", return_value=fake_index):
        ingest_path(str(tmp_path))

    fake_index.delete.assert_not_called()


def test_ingest_path_ingests_pdfs(tmp_path):
    _write_pdf(tmp_path, "travel_policy", "The daily meal allowance is $85.")
    fake_index = MagicMock()

    with patch("rag.ingest.ensure_index", return_value=fake_index):
        total = ingest_path(str(tmp_path))

    assert total >= 1
    records = fake_index.upsert_records.call_args.kwargs["records"]
    assert records[0]["source"] == "travel_policy"


def test_read_document_extracts_table_cells_from_pdf(tmp_path):
    pdf = _write_pdf(
        tmp_path,
        "sla_policy",
        "## SLA Tiers\n\n"
        "| Tier | Response Time |\n"
        "|---|---|\n"
        "| Gold | 2 hours |\n"
        "| Silver | 8 hours |\n",
    )

    text = read_document(pdf)

    # " | "-joined cell text only comes from the extract_tables() branch —
    # extract_text() alone would never insert that separator between cells.
    # A scrambled join (wrong row, swapped columns, dropped cell) fails this.
    assert "Gold | 2 hours" in text
    assert "Silver | 8 hours" in text


def test_read_document_skips_unreadable_pdf_without_raising(tmp_path, capsys):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 this is not a real pdf body" * 5)

    text = read_document(corrupt)

    assert text == ""
    stderr = capsys.readouterr().err
    assert "corrupt.pdf" in stderr


def test_ingest_path_skips_corrupt_pdf_and_completes(tmp_path):
    (tmp_path / "good.md").write_text("alpha content")
    (tmp_path / "corrupt.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf body" * 5)
    fake_index = MagicMock()

    with patch("rag.ingest.ensure_index", return_value=fake_index):
        total = ingest_path(str(tmp_path))

    # The corrupt PDF contributes zero chunks and does not abort the run —
    # only the one valid document's chunk gets upserted.
    assert total == 1
    assert fake_index.upsert_records.call_count == 1
