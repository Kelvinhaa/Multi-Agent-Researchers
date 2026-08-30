"""Render corpus markdown sources to PDFs with realistic headers and footers.

One-shot script. Holds no content — all content lives in eval/corpus_src/,
all figures trace to docs/corpus_facts.md.
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SRC_DIR = Path("eval/corpus_src")
OUT_DIR = Path("docs/corpus")

FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split `---`-delimited key: value front matter from the markdown body."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        raise ValueError("Document is missing front matter")

    meta = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')

    return meta, text[match.end() :]


def _draw_header_footer(canvas, doc, meta: dict) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)

    canvas.drawString(20 * mm, A4[1] - 12 * mm, "Harbourline Freight Systems")
    canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 12 * mm, meta["title"])
    canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)

    footer = (
        f"{meta['doc_id']}  |  v{meta['version']}  |  effective {meta['effective']}"
    )
    canvas.drawString(20 * mm, 12 * mm, footer)
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _table_from_block(rows: list[str], styles) -> Table:
    cells = [
        [c.strip() for c in row.strip().strip("|").split("|")]
        for row in rows
        if not set(row.replace("|", "").strip()) <= {"-", " "}
    ]
    wrapped = [[Paragraph(c, styles["BodyText"]) for c in row] for row in cells]
    table = Table(wrapped, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _flowables(body: str, styles) -> list:
    story: list = []
    bullets: list[str] = []
    table_rows: list[str] = []

    def flush() -> None:
        nonlocal bullets, table_rows
        if bullets:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(b, styles["BodyText"])) for b in bullets],
                    bulletType="bullet",
                    leftIndent=12,
                )
            )
            bullets = []
        if table_rows:
            story.append(_table_from_block(table_rows, styles))
            story.append(Spacer(1, 6))
            table_rows = []

    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            bullets and flush()
            table_rows.append(line)
        elif line.startswith("- "):
            table_rows and flush()
            bullets.append(line[2:])
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(line[3:], styles["H2"]))
        elif line.startswith("# "):
            flush()
            story.append(Paragraph(line[2:], styles["H1"]))
        elif not line:
            flush()
        else:
            flush()
            story.append(Paragraph(line, styles["BodyText"]))

    flush()
    return story


def render_pdf(md_path: Path, out_path: Path) -> None:
    """Render one markdown source to a PDF at out_path."""
    meta, body = parse_front_matter(md_path.read_text(encoding="utf-8"))

    base = getSampleStyleSheet()
    styles = {
        "BodyText": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=9.5, leading=13, spaceAfter=5
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=15, spaceAfter=9, spaceBefore=2
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=11.5, spaceAfter=6, spaceBefore=10
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=meta["title"],
    )

    def on_page(canvas, doc_):
        _draw_header_footer(canvas, doc_, meta)

    doc.build(_flowables(body, styles), onFirstPage=on_page, onLaterPages=on_page)


def main() -> int:
    sources = sorted(SRC_DIR.glob("*.md"))
    if not sources:
        raise SystemExit(f"No corpus sources found in {SRC_DIR}")

    for src in sources:
        out = OUT_DIR / f"{src.stem}.pdf"
        render_pdf(src, out)
        print(f"{src.name} -> {out}")

    print(f"Rendered {len(sources)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
