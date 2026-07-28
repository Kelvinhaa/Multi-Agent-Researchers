# RAG Eval Suite + Synthetic Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the project's first real RAG quality numbers by building a 20-document synthetic corpus, ingesting it, and scoring both retrieval and end-to-end answer quality.

**Architecture:** Corpus content is authored as markdown under `eval/corpus_src/`, rendered to realistic PDFs in `docs/corpus/` by a one-shot generator, and ingested through an extended `rag/ingest.py`. The eval splits into pure retrieval metrics (`eval/metrics.py`), LLM-judged answer metrics (`eval/judges.py`), and a CLI orchestrator (`eval/run.py`) that runs both passes against `eval/golden_set.json`.

**Tech Stack:** Python 3.12, `reportlab` (PDF generation), `pdfplumber` (PDF extraction), `langchain-openai` (judge LLM + embeddings), Pinecone integrated inference, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-28-rag-eval-corpus-design.md`. Read it before starting.
- Judge LLM is `gpt-4o` — deliberately NOT the agent's `gpt-4o-mini` (self-preference bias). Embeddings are `text-embedding-3-small`.
- Async is tested with plain `asyncio.run(...)` inside sync test functions. Do NOT add `pytest-asyncio` — the existing suite (`tests/test_nodes.py`) uses this pattern.
- All test commands run with `PYTHONPATH=.` — e.g. `PYTHONPATH=. uv run pytest tests/ -v`.
- Every figure in the corpus comes from `docs/corpus_facts.md`. Never invent a number in a corpus document; if it is not in the facts sheet, add it there first.
- Corpus documents must state facts in **prose**, not tables alone. Tables are for realism; a table-extraction failure must degrade, never silently corrupt ground truth.
- Where a policy touches a real Australian statutory floor, it must sit *above* it with a company-specific figure. Never state a real-world legal entitlement as a bare fact.
- `docs/corpus_facts.md` lives OUTSIDE `docs/corpus/` and must never be ingested.
- Ingest target is `docs/corpus/`, never `docs/` — `iter_doc_paths` uses `rglob` and would otherwise sweep in `docs/superpowers/specs/*.md`.
- Run `uv run ruff format . && uv run ruff check .` before each commit.
- `eval/judges.py` builds its `ChatOpenAI` clients at module level, so importing it (including from tests) requires `OPENAI_API_KEY` to be present. It is, via `.env` and `load_dotenv`. The judge tests still make no network calls — they patch the client objects — but the import itself is not key-free. If this ever needs to run in a keyless CI, move the client construction behind a lazy accessor.

---

## File Structure

| File | Responsibility |
|---|---|
| `eval/metrics.py` (modify) | Pure retrieval metrics. No I/O. |
| `docs/corpus_facts.md` (create) | Canonical figures. Single source of truth for all corpus content. |
| `eval/corpus_src/*.md` (create, 20) | Corpus content as reviewable markdown with front-matter. |
| `eval/generate_corpus.py` (create) | Renders `corpus_src/*.md` → `docs/corpus/*.pdf`. Generic; holds no content. |
| `rag/ingest.py` (modify) | Adds PDF reading, `--reset`, batched upserts. |
| `eval/golden_set.json` (rewrite) | ~20 scored items against the new corpus. |
| `eval/judges.py` (create) | LLM-judged answer metrics + pure `average_precision`. |
| `eval/run.py` (create) | CLI orchestrator, aggregation, report writing. |
| `tests/test_eval_metrics.py` (create) | Metrics + `average_precision` + aggregation tests. |
| `tests/test_generate_corpus.py` (create) | Front-matter parsing + PDF smoke test. |
| `tests/test_ingest.py` (modify) | PDF reading, reset, batching. |

---

### Task 1: Retrieval metrics

Fixes a misnamed function and adds the fractional recall that makes multi-source golden items meaningful. Pure functions, no dependencies — safe first task.

**Files:**
- Modify: `eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float`, `recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float`, `mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float`. All used by `eval/run.py` in Task 7.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_metrics.py`:

```python
from eval.metrics import hit_rate_at_k, mrr, recall_at_k


def test_hit_rate_is_one_when_any_expected_source_retrieved():
    assert hit_rate_at_k(["travel_policy", "corporate_card"], ["travel_policy"]) == 1.0


def test_hit_rate_is_zero_on_disjoint_sets():
    assert hit_rate_at_k(["leave_policy"], ["travel_policy"]) == 0.0


def test_hit_rate_is_zero_when_nothing_retrieved():
    assert hit_rate_at_k([], ["travel_policy"]) == 0.0


def test_hit_rate_counts_duplicate_sources_once():
    assert hit_rate_at_k(["travel_policy", "travel_policy"], ["travel_policy"]) == 1.0


def test_recall_is_fractional_when_only_some_expected_sources_found():
    assert recall_at_k(["travel_policy"], ["travel_policy", "corporate_card"]) == 0.5


def test_recall_is_one_when_all_expected_sources_found():
    assert recall_at_k(
        ["corporate_card", "travel_policy"], ["travel_policy", "corporate_card"]
    ) == 1.0


def test_recall_ignores_duplicate_retrievals():
    assert recall_at_k(["travel_policy", "travel_policy"], ["travel_policy"]) == 1.0


def test_recall_is_zero_when_expected_sources_empty():
    assert recall_at_k(["travel_policy"], []) == 0.0


def test_mrr_is_one_when_first_result_matches():
    assert mrr(["travel_policy", "leave_policy"], ["travel_policy"]) == 1.0


def test_mrr_uses_reciprocal_of_first_matching_rank():
    assert mrr(["a", "b", "travel_policy"], ["travel_policy"]) == 1 / 3


def test_mrr_is_zero_when_no_result_matches():
    assert mrr(["a", "b"], ["travel_policy"]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'hit_rate_at_k'`

- [ ] **Step 3: Rewrite `eval/metrics.py`**

```python
"""Pure retrieval metrics. Source-document granularity, no I/O."""


def hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1.0 if any retrieved source is expected, else 0.0.

    Binary hit rate — deliberately not fractional. Use recall_at_k when the
    question genuinely requires more than one source.
    """
    return 1.0 if set(retrieved_sources) & set(expected_sources) else 0.0


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of expected sources that were retrieved.

    Empty expected_sources -> 0.0 (unanswerable items are scored on abstention
    instead, and never reach this function).
    """
    expected = set(expected_sources)
    if not expected:
        return 0.0
    return len(expected & set(retrieved_sources)) / len(expected)


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal of the 1-indexed rank of the first expected source. 0.0 if none."""
    expected = set(expected_sources)
    for i, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / i
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_eval_metrics.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat: add fractional recall and rename the binary hit-rate metric"
```

---

### Task 2: Facts sheet, PDF generator, and the `people` cluster

Establishes the content pipeline and proves it end to end on one cluster before scaling to twenty documents.

**Files:**
- Create: `docs/corpus_facts.md`
- Create: `eval/generate_corpus.py`
- Create: `eval/corpus_src/leave_policy.md`, `parental_leave.md`, `remote_work.md`, `performance_review.md`, `code_of_conduct.md`
- Test: `tests/test_generate_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_front_matter(text: str) -> tuple[dict, str]`, `render_pdf(md_path: Path, out_path: Path) -> None`, `main() -> int`. Task 4 adds source files consumed by the same generator; Task 3 ingests its PDF output.

- [ ] **Step 1: Add the reportlab dependency**

```bash
uv add reportlab
```

- [ ] **Step 2: Write `docs/corpus_facts.md`**

This is the single source of truth. Every number in every corpus document must trace to a line here.

```markdown
# Harbourline Freight Systems — canonical facts

Internal scaffolding for the eval corpus. NOT ingested — lives outside `docs/corpus/`.
Every figure in `eval/corpus_src/` must trace to a line in this file.

## Company

Harbourline Freight Systems. B2B logistics software. Founded 2014.
HQ Melbourne; offices Sydney, Perth, Auckland. 340 staff.

## People

| Fact | Value | Document |
|---|---|---|
| Annual leave | 25 days per year | leave_policy |
| Personal/carer's leave | 12 days per year | leave_policy |
| Leave request notice | 10 business days for 5+ consecutive days | leave_policy |
| Leave balance cap | 40 days, excess forfeited each 30 June | leave_policy |
| Parental leave, primary carer | 18 weeks at full pay | parental_leave |
| Parental leave, secondary carer | 6 weeks at full pay | parental_leave |
| Parental leave eligibility | 12 months continuous service | parental_leave |
| Parental leave, superannuation | Paid on unpaid portion for up to 12 months | parental_leave |
| Remote days | Up to 3 days per week | remote_work |
| Core hours | 10:00–15:00 AEST | remote_work |
| Fully remote | Requires Executive Leadership Team approval | remote_work |
| Home office allowance | $650 once per 24 months | remote_work |
| Review cycle | Twice yearly, March and September | performance_review |
| Promotion nominations close | 14 February and 14 August | performance_review |
| Rating scale | 1–5, where 3 is "meets expectations" | performance_review |
| Calibration | Panel of 3 department heads | performance_review |
| Gift declaration threshold | $200 | code_of_conduct |
| Conflict declaration window | 5 business days | code_of_conduct |
| Secondary employment | Written approval required | code_of_conduct |

## Finance

| Fact | Value | Document |
|---|---|---|
| Expense submission window | 30 days from spend | expense_reimbursement |
| Receipt required above | $75 | expense_reimbursement |
| Reimbursement pay run | 15th of each month | expense_reimbursement |
| Domestic meal allowance | $85 per day | travel_policy |
| International meal allowance | $120 per day | travel_policy |
| Hotel cap, Sydney/Melbourne | $280 per night | travel_policy |
| Hotel cap, elsewhere domestic | $220 per night | travel_policy |
| Business class | Flights over 4 hours, ELT only | travel_policy |
| Corporate card limit, standard | $5,000 per month | corporate_card |
| Corporate card limit, manager | $15,000 per month | corporate_card |
| Card reconciliation window | 10 business days | corporate_card |
| Card suspension | After 2 missed reconciliations | corporate_card |
| Procurement, manager approval | Under $5,000 | procurement_approval |
| Procurement, department head | $5,000–$25,000 | procurement_approval |
| Procurement, CFO | $25,000–$100,000 | procurement_approval |
| Procurement, board | Above $100,000 | procurement_approval |
| New vendor security review | Required if handling customer data | procurement_approval |
| Standard payment terms | 30 days | invoicing_payment_terms |
| Enterprise payment terms | 45 days | invoicing_payment_terms |
| Late payment fee | 1.5% per month | invoicing_payment_terms |
| Purchase order required above | $2,500 | invoicing_payment_terms |

## Engineering

| Fact | Value | Document |
|---|---|---|
| P1 internal acknowledgement | 15 minutes | incident_response |
| P1 resolution target | 4 hours | incident_response |
| P2 acknowledgement | 1 hour | incident_response |
| P2 resolution target | 12 hours | incident_response |
| P3 acknowledgement | 4 hours | incident_response |
| P3 resolution target | 5 business days | incident_response |
| P1 status page update | Within 30 minutes | incident_response |
| P1 incident commander | Mandatory | incident_response |
| On-call rotation length | 7 days | on_call_rotation |
| On-call handover | Wednesday 10:00 AEST | on_call_rotation |
| On-call allowance | $900 per week | on_call_rotation |
| On-call frequency cap | 1 rotation per 4 weeks | on_call_rotation |
| Deploy window | Monday–Thursday, no Friday deploys | release_deployment |
| Canary stage | 10% of traffic for 30 minutes | release_deployment |
| Automatic rollback | Error rate above 2% | release_deployment |
| Production deploy approval | Release manager | release_deployment |
| Production access | Time-boxed 8-hour elevation | access_control |
| Access review cadence | Quarterly | access_control |
| Offboarding revocation | Within 2 hours | access_control |
| SSO | Mandatory, no local passwords | access_control |
| Customer shipment data retention | 7 years | data_retention |
| Application log retention | 90 days | data_retention |
| Audit log retention | 3 years | data_retention |
| Backup retention | 35 days | data_retention |
| Deletion request fulfilment | Within 30 days | data_retention |

## Commercial

| Fact | Value | Document |
|---|---|---|
| Discount, account executive | Up to 10% | discount_approval |
| Discount, sales manager | 10–20% | discount_approval |
| Discount, VP Sales | 20–30% | discount_approval |
| Discount, CFO and CEO | Above 30% | discount_approval |
| Uptime, standard tier | 99.9% | sla_service_credits |
| Uptime, enterprise tier | 99.95% | sla_service_credits |
| Service credit, below 99.9% | 10% of monthly fee | sla_service_credits |
| Service credit, below 99.0% | 25% of monthly fee | sla_service_credits |
| Service credit, below 95% | 50% of monthly fee | sla_service_credits |
| Contractual P1 response | 30 minutes | sla_service_credits |
| Credit claim window | 30 days from incident | sla_service_credits |
| Money-back window | 30 days, new annual contracts | refund_policy |
| Pro-rata cancellation notice | 90 days | refund_policy |
| Usage overages | Non-refundable | refund_policy |
| Breach notification | Within 72 hours | data_processing |
| Default data residency | ap-southeast-2 | data_processing |
| EU data residency option | eu-west-1 | data_processing |
| Sub-processor change notice | 30 days | data_processing |
| Partner tiers | Registered, Silver, Gold | partner_program |
| Gold tier requirement | $500,000 referred ARR | partner_program |
| Partner margin | 15% / 22% / 30% by tier | partner_program |
| Deal registration validity | 90 days | partner_program |

## Planted near-misses

Deliberate traps. A weak retriever confuses these; they are what makes
`recall_at_k` move.

| # | Collision | Documents |
|---|---|---|
| 1 | P1 response: 15 min internal ack vs 30 min contractual | incident_response vs sla_service_credits |
| 2 | "30 days" means four different things | expense_reimbursement, invoicing_payment_terms, refund_policy, data_retention |
| 3 | "90 days" means two different things | refund_policy vs partner_program |
| 4 | Spending limits appear in three finance documents | expense_reimbursement, travel_policy, corporate_card |
| 5 | Retention periods differ by data class within one document | data_retention |

## Deliberately absent

Never appears in any corpus document. Used for unanswerable golden items.

- Employee stock option or equity policy
- Annual professional development / training budget
- Bereavement leave entitlement
- Preferred airline or travel booking vendor
```

- [ ] **Step 3: Write the failing generator tests**

Create `tests/test_generate_corpus.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_generate_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.generate_corpus'`

- [ ] **Step 5: Write `eval/generate_corpus.py`**

```python
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

    footer = f"{meta['doc_id']}  |  v{meta['version']}  |  effective {meta['effective']}"
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_generate_corpus.py -v`
Expected: 3 passed

- [ ] **Step 7: Write the five `people` cluster sources**

Create each file under `eval/corpus_src/`. Every figure must come from the People table in `docs/corpus_facts.md`. Each document needs front matter, an H1, 3–5 `##` sections, at least one table, and **every fact restated in prose**.

Use this as the pattern — `eval/corpus_src/leave_policy.md`:

```markdown
---
doc_id: HFS-PEO-001
title: Annual and Personal Leave Policy
version: "3.2"
effective: 2026-03-01
---

# Annual and Personal Leave Policy

## Purpose

This policy sets out how Harbourline Freight Systems staff accrue, request and
take annual leave and personal leave. It applies to all permanent employees
across our Melbourne, Sydney, Perth and Auckland offices.

## Annual leave entitlement

Permanent full-time employees accrue 25 days of paid annual leave per year,
accruing progressively from the first day of employment. Part-time employees
accrue on a pro-rata basis according to their contracted hours.

Accrued annual leave is capped at 40 days. Any balance above 40 days is
forfeited on 30 June each year, so staff carrying large balances should plan
leave with their manager well before the end of the financial year.

| Leave type | Entitlement | Accrual |
|---|---|---|
| Annual leave | 25 days per year | Progressive |
| Personal/carer's leave | 12 days per year | Progressive |

## Personal and carer's leave

Employees receive 12 days of paid personal and carer's leave per year. This
covers personal illness or injury, and caring responsibilities for an immediate
family or household member.

A medical certificate is required for absences of three or more consecutive
days, and for any absence immediately before or after a public holiday.

## Requesting leave

Leave requests are submitted through the People Portal. Requests of five or
more consecutive days require at least 10 business days notice so that team
coverage can be arranged. Shorter absences require manager approval but no
fixed notice period.

Managers respond to leave requests within five business days. Where a request
is declined, the manager must record the operational reason in the portal.
```

Then write the remaining four to the same standard:

- **`parental_leave.md`** (`HFS-PEO-002`, v2.4, effective 2026-01-15) — 18 weeks full pay for primary carers, 6 weeks full pay for secondary carers, 12 months continuous service to qualify, superannuation paid on the unpaid portion for up to 12 months.
- **`remote_work.md`** (`HFS-PEO-003`, v4.1, effective 2026-05-01) — up to 3 remote days per week, core hours 10:00–15:00 AEST, fully remote arrangements need Executive Leadership Team approval, $650 home office allowance once per 24 months.
- **`performance_review.md`** (`HFS-PEO-004`, v2.0, effective 2026-02-01) — reviews twice yearly in March and September, promotion nominations close 14 February and 14 August, 1–5 rating scale where 3 means "meets expectations", calibration by a panel of 3 department heads.
- **`code_of_conduct.md`** (`HFS-PEO-005`, v5.3, effective 2026-04-01) — gifts above $200 must be declared, conflicts of interest declared within 5 business days, secondary employment requires written approval.

- [ ] **Step 8: Generate the PDFs and inspect them**

Run: `PYTHONPATH=. uv run python -m eval.generate_corpus`
Expected: five `-> docs/corpus/*.pdf` lines and `Rendered 5 documents`

Open `docs/corpus/leave_policy.pdf` and confirm the header, footer with document ID and version, section headings, and the table all render.

- [ ] **Step 9: Remove the obsolete corpus**

```bash
git rm docs/corpus/README.md docs/corpus/ENGINEERING_LOG.md docs/corpus/probability_basics.md
```

If those files are untracked rather than tracked, delete them with `rm` instead.

- [ ] **Step 10: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add docs/corpus_facts.md eval/generate_corpus.py eval/corpus_src tests/test_generate_corpus.py docs/corpus pyproject.toml uv.lock
git commit -m "feat: add corpus facts sheet, PDF generator, and people cluster"
```

---

### Task 3: PDF ingestion, namespace reset, and batching

Without `--reset` the 34 existing records survive re-ingest and get retrieved as stale context, silently corrupting every eval result.

**Files:**
- Modify: `rag/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `eval.generate_corpus.render_pdf` (test fixtures only).
- Produces: `read_document(path: Path) -> str`, `reset_namespace(index) -> None`, `batched(records: list[dict], size: int) -> Iterator[list[dict]]`, `ingest_path(docs_dir: str, reset: bool = False) -> int`.

- [ ] **Step 1: Add the pdfplumber dependency**

```bash
uv add pdfplumber
```

`pypdf` is explicitly rejected — it flattens tables into scrambled text, which would corrupt ground truth invisibly.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from eval.generate_corpus import render_pdf
from rag.ingest import batched, ingest_path, read_document, reset_namespace


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
```

Note: `test_ingest_path_upserts_chunks_for_each_doc` already exists and must keep passing unchanged.

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_ingest.py -v`
Expected: FAIL with `ImportError: cannot import name 'read_document'`

- [ ] **Step 4: Update `rag/ingest.py`**

Replace the imports, constants and `ingest_path`; leave `chunk_text` and `build_records` untouched.

```python
import argparse
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
```

Add these functions:

```python
def read_document(path: Path) -> str:
    """Read a document to plain text, dispatching on file extension.

    PDFs go through pdfplumber rather than pypdf: pypdf flattens tables into
    scrambled text, which would silently corrupt retrieved context.
    """
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    parts: list[str] = []
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
```

Replace `ingest_path` and the `__main__` block:

```python
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
```

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. uv run pytest tests/ -v`
Expected: all pass, including the pre-existing ingest tests

- [ ] **Step 6: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add rag/ingest.py tests/test_ingest.py pyproject.toml uv.lock
git commit -m "feat: ingest PDFs, batch upserts, and add a namespace reset flag"
```

---

### Task 4: The remaining three clusters

Fifteen documents completing the corpus. Same pattern as Task 2 Step 7 — front matter, H1, 3–5 `##` sections, at least one table, every fact also stated in prose, every figure traced to `docs/corpus_facts.md`.

**Files:**
- Create: 15 files under `eval/corpus_src/`

**Interfaces:**
- Consumes: `eval.generate_corpus.main`.
- Produces: 20 PDFs in `docs/corpus/`, whose filename stems become the `source` values in `eval/golden_set.json`.

- [ ] **Step 1: Write the `finance` cluster**

- **`expense_reimbursement.md`** (`HFS-FIN-001`, v3.0, 2026-02-01) — claims submitted within 30 days of spend, receipts required above $75, reimbursement paid in the pay run on the 15th of each month.
- **`travel_policy.md`** (`HFS-FIN-002`, v4.2, 2026-04-01) — $85 domestic and $120 international daily meal allowance, hotel caps of $280 in Sydney and Melbourne and $220 elsewhere domestically, business class only on flights over 4 hours and only for the Executive Leadership Team.
- **`corporate_card.md`** (`HFS-FIN-003`, v2.1, 2026-03-15) — $5,000 standard and $15,000 manager monthly limits, reconciliation within 10 business days, card suspended after 2 missed reconciliations.
- **`procurement_approval.md`** (`HFS-FIN-004`, v3.4, 2026-01-10) — under $5,000 manager, $5,000–$25,000 department head, $25,000–$100,000 CFO, above $100,000 board; new vendors handling customer data require a security review.
- **`invoicing_payment_terms.md`** (`HFS-FIN-005`, v2.2, 2026-05-01) — 30-day standard and 45-day enterprise payment terms, 1.5% monthly late fee, purchase order required above $2,500.

- [ ] **Step 2: Write the `engineering` cluster**

- **`incident_response.md`** (`HFS-ENG-001`, v6.0, 2026-06-01) — P1 acknowledged within 15 minutes and targeted for resolution in 4 hours, P2 within 1 hour and 12 hours, P3 within 4 hours and 5 business days; P1 requires a named incident commander and a status page update within 30 minutes.
- **`on_call_rotation.md`** (`HFS-ENG-002`, v3.1, 2026-04-15) — 7-day rotations handing over Wednesday 10:00 AEST, primary and secondary responders, $900 weekly allowance, no more than 1 rotation in any 4 weeks.
- **`release_deployment.md`** (`HFS-ENG-003`, v5.2, 2026-05-20) — deploys Monday to Thursday with no Friday deploys, canary at 10% of traffic for 30 minutes, automatic rollback above a 2% error rate, release manager approval for production.
- **`access_control.md`** (`HFS-ENG-004`, v4.0, 2026-03-01) — SSO mandatory with no local passwords, production access through time-boxed 8-hour elevation, quarterly access reviews, offboarding revocation within 2 hours.
- **`data_retention.md`** (`HFS-ENG-005`, v3.3, 2026-02-15) — customer shipment data 7 years, application logs 90 days, audit logs 3 years, backups 35 days, deletion requests fulfilled within 30 days.

- [ ] **Step 3: Write the `commercial` cluster**

- **`discount_approval.md`** (`HFS-COM-001`, v2.5, 2026-01-20) — up to 10% account executive, 10–20% sales manager, 20–30% VP Sales, above 30% joint CFO and CEO approval.
- **`sla_service_credits.md`** (`HFS-COM-002`, v4.1, 2026-04-01) — 99.9% standard and 99.95% enterprise uptime; service credits of 10%, 25% and 50% of monthly fees below 99.9%, 99.0% and 95%; contractual P1 response within 30 minutes; credits claimed within 30 days of the incident.
- **`refund_policy.md`** (`HFS-COM-003`, v1.8, 2026-02-10) — 30-day money-back guarantee on new annual contracts, pro-rata refunds with 90 days notice, usage overages non-refundable.
- **`data_processing.md`** (`HFS-COM-004`, v3.0, 2026-05-15) — breach notification within 72 hours, default residency ap-southeast-2 with an eu-west-1 option, 30 days notice of sub-processor changes.
- **`partner_program.md`** (`HFS-COM-005`, v2.3, 2026-03-20) — Registered, Silver and Gold tiers; Gold requires $500,000 referred ARR; margins of 15%, 22% and 30%; deal registration valid 90 days.

- [ ] **Step 4: Confirm the near-misses landed**

Re-read `docs/corpus_facts.md`'s "Planted near-misses" table and confirm each collision is genuinely present in the written documents. In particular, `incident_response.md` must say 15 minutes and `sla_service_credits.md` must say 30 minutes for P1 — that contrast is load-bearing for the eval.

- [ ] **Step 5: Generate all twenty PDFs**

Run: `PYTHONPATH=. uv run python -m eval.generate_corpus`
Expected: `Rendered 20 documents`

- [ ] **Step 6: Ingest the corpus**

```bash
PYTHONPATH=. uv run python -m rag.ingest docs/corpus/ --reset
```
Expected: a chunk count well above 34.

Verify no stale records survive — the index must contain no IDs beginning `README-`, `ENGINEERING_LOG-` or `probability_basics-`.

- [ ] **Step 7: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add eval/corpus_src docs/corpus
git commit -m "feat: complete the corpus with finance, engineering and commercial clusters"
```

---

### Task 5: Golden set

**Files:**
- Rewrite: `eval/golden_set.json`

**Interfaces:**
- Consumes: corpus filename stems from Task 4.
- Produces: items with keys `id`, `query`, `cluster`, `expected_sources`, `reference_answer`, `answerable` — consumed by `eval/run.py` in Task 7.

- [ ] **Step 1: Replace `eval/golden_set.json`**

Every `expected_sources` value must exactly match a PDF filename stem in `docs/corpus/`. Every `reference_answer` must be checkable against `docs/corpus_facts.md`.

```json
[
  {
    "id": "peo-001",
    "query": "How many days of annual leave do permanent full-time employees get?",
    "cluster": "people",
    "expected_sources": ["leave_policy"],
    "reference_answer": "Permanent full-time employees accrue 25 days of paid annual leave per year, accruing progressively. Balances are capped at 40 days, with any excess forfeited on 30 June.",
    "answerable": true
  },
  {
    "id": "peo-002",
    "query": "How much paid parental leave does a primary carer receive?",
    "cluster": "people",
    "expected_sources": ["parental_leave"],
    "reference_answer": "Primary carers receive 18 weeks at full pay, after 12 months of continuous service. Secondary carers receive 6 weeks at full pay.",
    "answerable": true
  },
  {
    "id": "peo-003",
    "query": "What are the core hours staff must be available during?",
    "cluster": "people",
    "expected_sources": ["remote_work"],
    "reference_answer": "Core hours are 10:00 to 15:00 AEST. Staff may work remotely up to 3 days per week; fully remote arrangements require Executive Leadership Team approval.",
    "answerable": true
  },
  {
    "id": "peo-004",
    "query": "When do promotion nominations close?",
    "cluster": "people",
    "expected_sources": ["performance_review"],
    "reference_answer": "Promotion nominations close on 14 February and 14 August, aligning with the twice-yearly review cycle in March and September.",
    "answerable": true
  },
  {
    "id": "peo-005",
    "query": "At what value does a gift need to be declared?",
    "cluster": "people",
    "expected_sources": ["code_of_conduct"],
    "reference_answer": "Gifts valued above $200 must be declared. Conflicts of interest must be declared within 5 business days.",
    "answerable": true
  },
  {
    "id": "fin-001",
    "query": "What is the daily meal allowance for domestic travel?",
    "cluster": "finance",
    "expected_sources": ["travel_policy"],
    "reference_answer": "The domestic daily meal allowance is $85. International travel is $120 per day.",
    "answerable": true
  },
  {
    "id": "fin-002",
    "query": "Above what amount do I need to keep a receipt for an expense claim?",
    "cluster": "finance",
    "expected_sources": ["expense_reimbursement"],
    "reference_answer": "Receipts are required for any expense above $75. Claims must be submitted within 30 days of the spend and are paid in the pay run on the 15th.",
    "answerable": true
  },
  {
    "id": "fin-003",
    "query": "What is the monthly corporate card limit for a manager?",
    "cluster": "finance",
    "expected_sources": ["corporate_card"],
    "reference_answer": "Managers have a $15,000 monthly corporate card limit, against a $5,000 standard limit. Cards must be reconciled within 10 business days.",
    "answerable": true
  },
  {
    "id": "fin-004",
    "query": "Who approves a purchase of $40,000?",
    "cluster": "finance",
    "expected_sources": ["procurement_approval"],
    "reference_answer": "The CFO approves purchases between $25,000 and $100,000. Above $100,000 requires board approval.",
    "answerable": true
  },
  {
    "id": "fin-005",
    "query": "What are the standard payment terms on a customer invoice?",
    "cluster": "finance",
    "expected_sources": ["invoicing_payment_terms"],
    "reference_answer": "Standard payment terms are 30 days; enterprise customers get 45 days. Late payments attract a 1.5% monthly fee.",
    "answerable": true
  },
  {
    "id": "eng-001",
    "query": "How quickly must a P1 incident be acknowledged internally?",
    "cluster": "engineering",
    "expected_sources": ["incident_response"],
    "reference_answer": "A P1 must be acknowledged within 15 minutes internally, with a resolution target of 4 hours, a named incident commander, and a status page update within 30 minutes.",
    "answerable": true
  },
  {
    "id": "eng-002",
    "query": "What is the on-call allowance and how long is a rotation?",
    "cluster": "engineering",
    "expected_sources": ["on_call_rotation"],
    "reference_answer": "Rotations run 7 days, handing over Wednesday at 10:00 AEST, and the allowance is $900 per week. No engineer does more than one rotation in any 4 weeks.",
    "answerable": true
  },
  {
    "id": "eng-003",
    "query": "What error rate triggers an automatic rollback during a deploy?",
    "cluster": "engineering",
    "expected_sources": ["release_deployment"],
    "reference_answer": "An error rate above 2% triggers automatic rollback. Releases canary at 10% of traffic for 30 minutes, and deploys run Monday to Thursday only.",
    "answerable": true
  },
  {
    "id": "eng-004",
    "query": "How long is production access granted for when elevated?",
    "cluster": "engineering",
    "expected_sources": ["access_control"],
    "reference_answer": "Production access is granted as a time-boxed 8-hour elevation. Access is reviewed quarterly and revoked within 2 hours of offboarding.",
    "answerable": true
  },
  {
    "id": "eng-005",
    "query": "How long is customer shipment data retained?",
    "cluster": "engineering",
    "expected_sources": ["data_retention"],
    "reference_answer": "Customer shipment data is retained for 7 years. Application logs are kept 90 days, audit logs 3 years, and backups 35 days.",
    "answerable": true
  },
  {
    "id": "com-001",
    "query": "Who needs to approve a 25% discount?",
    "cluster": "commercial",
    "expected_sources": ["discount_approval"],
    "reference_answer": "The VP Sales approves discounts between 20% and 30%. Above 30% requires joint CFO and CEO approval.",
    "answerable": true
  },
  {
    "id": "com-002",
    "query": "What service credit applies if uptime falls below 99%?",
    "cluster": "commercial",
    "expected_sources": ["sla_service_credits"],
    "reference_answer": "A 25% credit of the monthly fee applies below 99.0% uptime, rising to 50% below 95%. Below the 99.9% standard target the credit is 10%.",
    "answerable": true
  },
  {
    "id": "com-003",
    "query": "How much notice is needed to cancel and get a pro-rata refund?",
    "cluster": "commercial",
    "expected_sources": ["refund_policy"],
    "reference_answer": "Pro-rata refunds require 90 days notice. New annual contracts also carry a 30-day money-back guarantee, and usage overages are non-refundable.",
    "answerable": true
  },
  {
    "id": "com-004",
    "query": "Where is customer data stored by default, and what is the alternative?",
    "cluster": "commercial",
    "expected_sources": ["data_processing"],
    "reference_answer": "Data resides in ap-southeast-2 by default, with eu-west-1 available for customers requiring EU residency. Breaches are notified within 72 hours.",
    "answerable": true
  },
  {
    "id": "xcl-001",
    "query": "What is the P1 response time we commit to contractually, and how does it compare to our internal acknowledgement target?",
    "cluster": "cross",
    "expected_sources": ["sla_service_credits", "incident_response"],
    "reference_answer": "The contractual P1 response commitment is 30 minutes, while the internal acknowledgement target is stricter at 15 minutes, giving the on-call team headroom against the customer-facing commitment.",
    "answerable": true
  },
  {
    "id": "xcl-002",
    "query": "A partner registers a deal and the customer later cancels. How long is deal registration valid, and what notice does the customer need to give?",
    "cluster": "cross",
    "expected_sources": ["partner_program", "refund_policy"],
    "reference_answer": "Deal registration is valid for 90 days. Separately, a customer cancelling for a pro-rata refund must give 90 days notice — the two 90-day periods are unrelated.",
    "answerable": true
  },
  {
    "id": "una-001",
    "query": "What is Harbourline's employee stock option policy?",
    "cluster": "unanswerable",
    "expected_sources": [],
    "reference_answer": "The corpus contains no information about stock options or equity. The correct response is to say so rather than speculate.",
    "answerable": false
  },
  {
    "id": "una-002",
    "query": "How much is the annual professional development budget per employee?",
    "cluster": "unanswerable",
    "expected_sources": [],
    "reference_answer": "The corpus contains no professional development or training budget. The correct response is to say so rather than speculate.",
    "answerable": false
  },
  {
    "id": "una-003",
    "query": "How many days of bereavement leave are employees entitled to?",
    "cluster": "unanswerable",
    "expected_sources": [],
    "reference_answer": "The leave policy covers annual and personal/carer's leave only; bereavement leave is not addressed anywhere in the corpus.",
    "answerable": false
  },
  {
    "id": "una-004",
    "query": "Which airline is Harbourline's preferred carrier for domestic flights?",
    "cluster": "unanswerable",
    "expected_sources": [],
    "reference_answer": "The travel policy sets allowances and cabin-class rules but names no preferred airline or booking vendor.",
    "answerable": false
  }
]
```

- [ ] **Step 2: Verify every expected source exists**

```bash
PYTHONPATH=. uv run python -c "
import json, pathlib
stems = {p.stem for p in pathlib.Path('docs/corpus').glob('*.pdf')}
items = json.load(open('eval/golden_set.json'))
missing = {s for i in items for s in i['expected_sources']} - stems
print('MISSING:', missing or 'none')
print('items:', len(items), 'unanswerable:', sum(1 for i in items if not i['answerable']))
"
```
Expected: `MISSING: none`, `items: 25 unanswerable: 4`

- [ ] **Step 3: Commit**

```bash
git add eval/golden_set.json
git commit -m "feat: rewrite the golden set against the Harbourline corpus"
```

---

### Task 6: LLM judges

**Files:**
- Create: `eval/judges.py`
- Test: `tests/test_eval_metrics.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `average_precision(relevance: list[bool]) -> float`, `async score_faithfulness(answer: str, contexts: list[str]) -> float | None`, `async score_answer_relevancy(query: str, answer: str) -> float`, `async score_context_precision(query: str, reference: str, contexts: list[str]) -> float`, `async score_abstention(answer: str) -> float`. All consumed by `eval/run.py` in Task 7.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_metrics.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from eval.judges import average_precision, score_context_precision, score_faithfulness


def test_average_precision_is_one_when_all_contexts_relevant():
    assert average_precision([True, True, True]) == 1.0


def test_average_precision_is_zero_when_none_relevant():
    assert average_precision([False, False]) == 0.0


def test_average_precision_rewards_relevant_contexts_ranked_higher():
    early = average_precision([True, False, False, False])
    late = average_precision([False, False, False, True])

    assert early > late


def test_average_precision_matches_hand_computed_value():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert average_precision([True, False, True]) == (1.0 + 2 / 3) / 2


def test_average_precision_handles_empty_input():
    assert average_precision([]) == 0.0


def test_faithfulness_is_fraction_of_supported_claims():
    verdict = MagicMock(
        claims=[
            MagicMock(supported=True),
            MagicMock(supported=True),
            MagicMock(supported=False),
            MagicMock(supported=False),
        ]
    )
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=verdict)

    with patch("eval.judges._faithfulness_llm", fake_llm):
        score = asyncio.run(score_faithfulness("answer", ["context"]))

    assert score == 0.5


def test_faithfulness_returns_none_when_no_claims_extracted():
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(claims=[]))

    with patch("eval.judges._faithfulness_llm", fake_llm):
        assert asyncio.run(score_faithfulness("answer", ["context"])) is None


def test_faithfulness_returns_none_without_contexts():
    assert asyncio.run(score_faithfulness("answer", [])) is None


def test_context_precision_uses_judged_relevance_in_rank_order():
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(
        return_value=MagicMock(relevance=[True, False, True])
    )

    with patch("eval.judges._relevance_llm", fake_llm):
        score = asyncio.run(
            score_context_precision("q", "ref", ["c1", "c2", "c3"])
        )

    assert score == (1.0 + 2 / 3) / 2


def test_context_precision_is_zero_without_contexts():
    assert asyncio.run(score_context_precision("q", "ref", [])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.judges'`

- [ ] **Step 3: Write `eval/judges.py`**

```python
"""LLM-judged answer quality metrics.

Hand-rolled rather than using RAGAS: ragas 0.4.3 imports
langchain_community.chat_models.vertexai, deleted in langchain-community 0.4.2,
so it cannot be imported against this project's LangChain v1 stack.

The judge model is deliberately not the agent's gpt-4o-mini — grading a model's
output with itself invites self-preference bias.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv(override=True)

JUDGE_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"
N_GENERATED_QUESTIONS = 3


class Claim(BaseModel):
    text: str = Field(description="A single atomic factual claim from the answer")
    supported: bool = Field(description="True if the retrieved context supports it")


class ClaimVerdict(BaseModel):
    claims: list[Claim] = Field(description="Every atomic claim found in the answer")


class RelevanceVerdict(BaseModel):
    relevance: list[bool] = Field(
        description="One boolean per context chunk, in the order given"
    )


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="Questions the answer would answer")


class AbstentionVerdict(BaseModel):
    declined: bool = Field(
        description="True if the answer declines to answer or says the "
        "information is unavailable"
    )


_faithfulness_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    ClaimVerdict
)
_relevance_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    RelevanceVerdict
)
_question_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    GeneratedQuestions
)
_abstention_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    AbstentionVerdict
)
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)


def average_precision(relevance: list[bool]) -> float:
    """Rank-aware precision: relevant items ranked higher score more.

    AP = sum(precision@i for relevant i) / total_relevant
    """
    total_relevant = sum(relevance)
    if not total_relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / i

    return precision_sum / total_relevant


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_faithfulness(answer: str, contexts: list[str]) -> float | None:
    """Fraction of the answer's atomic claims supported by the retrieved context.

    Returns None when there is nothing to score — no contexts, or no claims
    extracted — so the item is excluded from the aggregate rather than
    counted as a zero.
    """
    if not contexts or not answer.strip():
        return None

    joined = "\n\n---\n\n".join(contexts)
    verdict = await _faithfulness_llm.ainvoke(
        "Break the ANSWER into atomic factual claims. For each claim, decide "
        "whether the CONTEXT supports it. Judge only against the context — not "
        "your own knowledge. A claim contradicted by or absent from the context "
        "is unsupported.\n\n"
        f"CONTEXT:\n{joined}\n\nANSWER:\n{answer}"
    )

    if not verdict.claims:
        return None

    return sum(c.supported for c in verdict.claims) / len(verdict.claims)


async def score_answer_relevancy(query: str, answer: str) -> float:
    """Mean cosine similarity between the original query and questions generated
    from the answer alone.

    Harder to game than a direct rating prompt: an answer that addresses a
    different question generates different questions, which embed further away.
    """
    if not answer.strip():
        return 0.0

    generated = await _question_llm.ainvoke(
        f"Read the ANSWER and write exactly {N_GENERATED_QUESTIONS} questions "
        "that it answers. Infer the questions from the answer alone.\n\n"
        f"ANSWER:\n{answer}"
    )
    if not generated.questions:
        return 0.0

    vectors = await _embeddings.aembed_documents([query] + generated.questions)
    query_vec, question_vecs = vectors[0], vectors[1:]

    return sum(_cosine(query_vec, v) for v in question_vecs) / len(question_vecs)


async def score_context_precision(
    query: str, reference: str, contexts: list[str]
) -> float:
    """Average precision over per-chunk relevance judgements, in rank order."""
    if not contexts:
        return 0.0

    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    verdict = await _relevance_llm.ainvoke(
        "Decide, for each numbered CONTEXT chunk, whether it helps answer the "
        "QUESTION given the REFERENCE ANSWER. Return one boolean per chunk in "
        f"the same order. Return exactly {len(contexts)} booleans.\n\n"
        f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference}\n\n"
        f"CONTEXT:\n{numbered}"
    )

    relevance = list(verdict.relevance[: len(contexts)])
    relevance += [False] * (len(contexts) - len(relevance))

    return average_precision(relevance)


async def score_abstention(answer: str) -> float:
    """1.0 when the answer correctly declines, 0.0 when it asserts an answer.

    Only applied to golden items marked answerable: false. Scoring a correct
    "I don't know" on answer relevancy would penalise the right behaviour.
    """
    if not answer.strip():
        return 1.0

    verdict = await _abstention_llm.ainvoke(
        "Does the ANSWER decline to answer — saying the information is "
        "unavailable, not in the documents, or unknown? Answer true if it "
        "declines, false if it asserts a substantive answer.\n\n"
        f"ANSWER:\n{answer}"
    )
    return 1.0 if verdict.declined else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_eval_metrics.py -v`
Expected: 21 passed, no network calls made

- [ ] **Step 5: Remove ragas**

```bash
uv remove ragas
```

Confirm it is gone from `pyproject.toml` and that `uv run python -c "import ragas"` now fails with `ModuleNotFoundError`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add eval/judges.py tests/test_eval_metrics.py pyproject.toml uv.lock
git commit -m "feat: hand-roll the answer-quality judges and drop ragas"
```

---

### Task 7: Runner and report

**Files:**
- Create: `eval/run.py`
- Test: `tests/test_eval_metrics.py` (append)

**Interfaces:**
- Consumes: `eval.metrics.{hit_rate_at_k, recall_at_k, mrr}`, `eval.judges.{score_faithfulness, score_answer_relevancy, score_context_precision, score_abstention}`, `rag.retriever.retrieve`, `agent.graph.graph`.
- Produces: `aggregate(records: list[dict]) -> dict`, `async evaluate_item(item: dict, semaphore) -> dict`, `async main() -> int`.

- [ ] **Step 1: Write the failing aggregation tests**

Append to `tests/test_eval_metrics.py`:

```python
from eval.run import aggregate


def test_aggregate_averages_each_metric_over_scored_items():
    records = [
        {"cluster": "finance", "hit_rate": 1.0, "recall": 1.0, "mrr": 1.0,
         "faithfulness": 0.8, "error": None},
        {"cluster": "finance", "hit_rate": 0.0, "recall": 0.0, "mrr": 0.0,
         "faithfulness": 0.4, "error": None},
    ]

    result = aggregate(records)

    assert result["overall"]["hit_rate"] == 0.5
    assert round(result["overall"]["faithfulness"], 6) == 0.6


def test_aggregate_excludes_none_scores_rather_than_counting_them_as_zero():
    records = [
        {"cluster": "people", "faithfulness": 1.0, "error": None},
        {"cluster": "people", "faithfulness": None, "error": None},
    ]

    assert aggregate(records)["overall"]["faithfulness"] == 1.0


def test_aggregate_excludes_errored_items():
    records = [
        {"cluster": "people", "hit_rate": 1.0, "error": None},
        {"cluster": "people", "hit_rate": 0.0, "error": "boom"},
    ]

    result = aggregate(records)

    assert result["overall"]["hit_rate"] == 1.0
    assert result["failed"] == 1


def test_aggregate_breaks_metrics_down_by_cluster():
    records = [
        {"cluster": "finance", "hit_rate": 1.0, "error": None},
        {"cluster": "engineering", "hit_rate": 0.0, "error": None},
    ]

    result = aggregate(records)

    assert result["by_cluster"]["finance"]["hit_rate"] == 1.0
    assert result["by_cluster"]["engineering"]["hit_rate"] == 0.0


def test_aggregate_handles_no_scorable_records():
    assert aggregate([{"cluster": "x", "hit_rate": 1.0, "error": "boom"}])["overall"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.run'`

- [ ] **Step 3: Write `eval/run.py`**

```python
"""Standalone RAG evaluation CLI.

Runs two independent passes per golden item — retrieval-only, and full
end-to-end through the agent graph — then scores both. Needs live Pinecone and
live OpenAI, so it deliberately sits outside the mocked pytest suite.

Usage: PYTHONPATH=. uv run python -m eval.run
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.graph import graph
from eval.judges import (
    score_abstention,
    score_answer_relevancy,
    score_context_precision,
    score_faithfulness,
)
from eval.metrics import hit_rate_at_k, mrr, recall_at_k
from rag.retriever import retrieve

GOLDEN_SET = Path("eval/golden_set.json")
RESULTS_DIR = Path("eval/results")
TOP_K = 5
CONCURRENCY = 4

METRIC_KEYS = (
    "hit_rate",
    "recall",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "abstention",
)


def aggregate(records: list[dict]) -> dict:
    """Mean of each metric over scored items, overall and per cluster.

    None scores are excluded rather than counted as zero — a metric that could
    not be computed is absent data, not a failure.
    """
    scored = [r for r in records if not r.get("error")]

    def means(rows: list[dict]) -> dict:
        out = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                out[key] = sum(values) / len(values)
        return out

    clusters = sorted({r["cluster"] for r in scored})
    return {
        "overall": means(scored),
        "by_cluster": {
            c: means([r for r in scored if r["cluster"] == c]) for c in clusters
        },
        "scored": len(scored),
        "failed": len(records) - len(scored),
    }


async def evaluate_item(item: dict, semaphore: asyncio.Semaphore) -> dict:
    """Run both passes for one golden item. Never raises — errors are recorded."""
    record = {
        "id": item["id"],
        "cluster": item["cluster"],
        "query": item["query"],
        "error": None,
    }

    async with semaphore:
        try:
            hits = await asyncio.to_thread(retrieve, item["query"], TOP_K)
            sources = [h["source"] for h in hits]
            record["retrieved_sources"] = sources

            if item["answerable"]:
                expected = item["expected_sources"]
                record["hit_rate"] = hit_rate_at_k(sources, expected)
                record["recall"] = recall_at_k(sources, expected)
                record["mrr"] = mrr(sources, expected)

            state = await graph.ainvoke({"query": item["query"]})
            answer = state.get("report") or ""
            contexts = [d["text"] for d in (state.get("retrieved_docs") or [])]
            record["answer"] = answer

            if item["answerable"]:
                record["faithfulness"] = await score_faithfulness(answer, contexts)
                record["answer_relevancy"] = await score_answer_relevancy(
                    item["query"], answer
                )
                record["context_precision"] = await score_context_precision(
                    item["query"], item["reference_answer"], contexts
                )
            else:
                record["abstention"] = await score_abstention(answer)

        except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
            record["error"] = f"{type(e).__name__}: {e}"

    status = "!" if record["error"] else "."
    print(status, end="", flush=True)
    return record


def _print_report(summary: dict) -> None:
    def row(label: str, metrics: dict) -> str:
        cells = "  ".join(
            f"{k}={metrics[k]:.3f}" for k in METRIC_KEYS if k in metrics
        )
        return f"  {label:<14} {cells}"

    print("\n\n=== RAG eval ===")
    print(f"scored {summary['scored']}, failed {summary['failed']}\n")
    print(row("OVERALL", summary["overall"]))
    print()
    for cluster, metrics in summary["by_cluster"].items():
        print(row(cluster, metrics))


async def main() -> int:
    if not GOLDEN_SET.exists():
        raise SystemExit(f"Golden set not found at {GOLDEN_SET}")

    items = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    if not items:
        raise SystemExit("Golden set is empty — nothing to evaluate")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    print(f"Evaluating {len(items)} items", flush=True)
    records = await asyncio.gather(*(evaluate_item(i, semaphore) for i in items))

    summary = aggregate(list(records))
    _print_report(summary)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"eval_{stamp}.json"
    out.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport written to {out}")

    for record in records:
        if record["error"]:
            print(f"  ERROR {record['id']}: {record['error']}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/ -v`
Expected: all pass, 26 in `test_eval_metrics.py`

- [ ] **Step 5: Run the eval for real**

```bash
PYTHONPATH=. uv run python -m eval.run
echo "exit: $?"
```

Expected: a progress line of dots, an overall row, a per-cluster breakdown, a JSON file under `eval/results/`, and exit 0.

- [ ] **Step 6: Sanity-check the baseline**

Within-cluster retrieval should be visibly weaker than cross-cluster. If every cluster scores 1.0 across the board, the planted near-misses were too gentle — record that in the results commit message and treat corpus hardening as a follow-on rather than silently accepting a perfect score.

Confirm the four `unanswerable` items score well on abstention. A low abstention score is a real finding: it means the pipeline confabulates when the corpus has no answer.

- [ ] **Step 7: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add eval/run.py tests/test_eval_metrics.py
git commit -m "feat: add the eval runner with per-cluster reporting"
```

---

### Task 8: Documentation

The `CLAUDE.md` ingest command is now actively wrong — `rglob` over `docs/` would sweep design specs into the index.

**Files:**
- Modify: `.claude/CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces nothing consumed by other tasks.

- [ ] **Step 1: Fix the ingest command in `.claude/CLAUDE.md`**

Replace `uv run python -m rag.ingest docs/` with:

```bash
uv run python -m rag.ingest docs/corpus/ --reset
```

Add to the commands block:

```bash
PYTHONPATH=. uv run python -m eval.generate_corpus
PYTHONPATH=. uv run python -m eval.run
```

- [ ] **Step 2: Add an eval section to `.claude/CLAUDE.md` architecture notes**

Keep it to a few lines — this file is deliberately short:

```markdown
Eval lives in `eval/`: `metrics.py` (pure retrieval metrics), `judges.py`
(LLM-judged answer quality, `gpt-4o` to avoid self-preference bias),
`run.py` (CLI, needs live Pinecone + OpenAI, outside the mocked pytest suite).
Corpus content is authored in `eval/corpus_src/*.md` and rendered to
`docs/corpus/*.pdf`; every figure traces to `docs/corpus_facts.md`.
```

- [ ] **Step 3: Update `README.md`**

Update the ingest command the same way, and replace the roadmap note about hybrid search with the current state — the eval suite now exists, and hybrid search plus score-sorted merging are the measured follow-ons.

- [ ] **Step 4: Commit**

```bash
git add .claude/CLAUDE.md README.md
git commit -m "docs: point ingest at docs/corpus and document the eval suite"
```

---

## Follow-ons (explicitly out of scope)

Recorded so they are not lost. Each now has a baseline to measure against.

1. **Global ranking bug** — `agent/nodes.py:41-50` concatenates per-sub-query results in sub-query order rather than sorting by score, so a 0.31 chunk can outrank a 0.88 chunk in the writer's prompt. Highest-value fix.
2. **Score threshold** — no minimum similarity, so junk always reaches the writer. The prompt at `agent/nodes.py:90-93` is a band-aid over this.
3. **Chunking** — `chunk_size=512` on PDFs with headings and tables. Section-boundary chunking, measured before and after.
4. **Reranking / hybrid search** — the README roadmap item.
5. **Content-hash ingestion** — `--reset` is a blunt instrument; per-document change detection would be correct.
