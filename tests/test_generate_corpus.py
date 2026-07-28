from pathlib import Path

from eval.generate_corpus import parse_front_matter, render_pdf


def test_parse_front_matter_extracts_metadata_and_body():
    text = """---
doc_id: HFS-PEO-001
title: Annual Leave Policy
version: "3.2"
effective: 2026-03-01
---

# Annual Leave Policy

Employees accrue 25 days.
"""
    meta, body = parse_front_matter(text)

    assert meta["doc_id"] == "HFS-PEO-001"
    assert meta["title"] == "Annual Leave Policy"
    assert meta["version"] == "3.2"
    assert body.lstrip().startswith("# Annual Leave Policy")


def test_parse_front_matter_raises_when_missing():
    try:
        parse_front_matter("# No front matter here\n")
    except ValueError as e:
        assert "front matter" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_render_pdf_writes_a_real_pdf(tmp_path: Path):
    src = tmp_path / "sample.md"
    src.write_text(
        """---
doc_id: HFS-TST-001
title: Sample Policy
version: "1.0"
effective: 2026-01-01
---

# Sample Policy

## Scope

This policy applies to all staff.

- First bullet
- Second bullet

| Tier | Limit |
|---|---|
| Standard | $5,000 |
""",
        encoding="utf-8",
    )
    out = tmp_path / "sample.pdf"

    render_pdf(src, out)

    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
    assert out.stat().st_size > 1000
