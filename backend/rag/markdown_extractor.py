"""
Markdown extraction pipeline for structured documents (PDF → Markdown → JSON).

Prototype of the extraction approach for invoices / structured PDFs, designed to
generalize to any structured document:

    PDF
     └─(pdfplumber: deterministic)→ Markdown   (header text + tables as pipe tables)
         └─(local LLM: exhaustive, schema-less)→ JSON record
             └─(coverage check vs source text)→ omission report

Design principles (aligned with goal.md):
  • Local-only: pdfplumber (no cloud) for structure; the LLM step uses the
    project's local llm_client (Ollama). Nothing leaves the device.
  • Schema-less: the LLM is told to extract EVERY field/party/address/table it
    finds using the document's own labels — no predefined field list to maintain
    per document type. New formats need no code/schema change.
  • Completeness: the markdown preserves 2-D table structure so the model sees
    clean rows (not a jumbled token stream), and a deterministic coverage check
    flags anything in the source that didn't make it into the JSON.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── PDF → Markdown ────────────────────────────────────────────────────────────

def _clean_cell(v) -> str:
    return "" if v is None else str(v).replace("\n", " ").strip()


def _table_to_markdown(tbl: list[list]) -> str:
    rows = [[_clean_cell(c) for c in r] for r in tbl if any(_clean_cell(c) for c in r)]
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [(r + [""] * ncols)[:ncols] for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join("---" for _ in range(ncols)) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _header_markdown(page, header_bottom: float) -> str:
    """
    Render the header region (above the first table). If it's a multi-column
    layout (e.g. seller left / client right), split it into separate blocks so
    the LLM never merges the two parties. Falls back to a single block.
    """
    if header_bottom <= 1:
        return ""
    mid = page.width / 2.0
    left = (page.crop((0, 0, mid, header_bottom)).extract_text() or "").strip()
    right = (page.crop((mid, 0, page.width, header_bottom)).extract_text() or "").strip()

    # Genuine two-column header: both sides have real content.
    if left and right and len(right) >= 15:
        return f"## Header — left block\n{left}\n\n## Header — right block\n{right}"

    full = (page.crop((0, 0, page.width, header_bottom)).extract_text() or "").strip()
    return f"## Header\n{full}" if full else ""


def pdf_to_markdown(file_path: str) -> tuple[str, str]:
    """
    Convert a PDF to a clean markdown representation + return the raw text.

    Returns (markdown, raw_text):
      markdown  — a column-split header block (so multi-party headers like
                  seller/client are kept separate) followed by each table as a
                  clean markdown pipe table. The raw jumbled page text is NOT
                  included (it duplicated tables and merged columns), so the LLM
                  sees only clean structure.
      raw_text  — full plain text, used only for the coverage check.
    """
    import pdfplumber

    raw_parts: list[str] = []
    md_sections: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for pi, page in enumerate(pdf.pages):
            raw_parts.append(page.extract_text() or "")
            try:
                tables = page.find_tables()
            except Exception:
                tables = []
            header_bottom = min((t.bbox[1] for t in tables), default=page.height)

            hdr = _header_markdown(page, header_bottom)
            if hdr:
                md_sections.append(hdr)

            for ti, t in enumerate(tables):
                try:
                    md = _table_to_markdown(t.extract())
                except Exception:
                    md = ""
                if md:
                    md_sections.append(f"### Table (page {pi + 1}, #{ti + 1})\n{md}")

    raw_text = "\n".join(raw_parts).strip()
    return "\n\n".join(md_sections), raw_text


# ── Markdown → JSON (exhaustive, schema-less) ─────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You are an exhaustive document-extraction engine. You convert a document "
    "into a complete structured JSON object. You never summarise and never omit "
    "information. Output JSON only."
)

_EXTRACTION_PROMPT = """\
Extract this document into a single JSON object.

Rules:
- Capture EVERY field, party, address, identifier, number, date, and table you
  find. Do not omit or summarise anything.
- Use the document's OWN labels as JSON keys (lowercased, spaces -> underscores).
- Each header block is a SEPARATE party. A "left block" and a "right block" are
  DIFFERENT entities (e.g. left = seller, right = client/buyer) — keep them as
  separate top-level objects. Never merge two parties' names, addresses, or tax
  ids together. Name each party object by its ROLE label found inside the block
  (e.g. "seller", "client"/"buyer") — do NOT use "left_block"/"right_block" as keys.
- Group each party's fields (name, full address, city, state, postal code, tax
  ids, gstin) into its own nested object.
- Put line-item / tabular rows into a JSON array of objects, one per row, using
  the table's column headers as keys. Do NOT duplicate a table.
- Preserve values verbatim (keep numbers, codes and dates exactly as written).
- If a value is not present, omit the key (do not invent).

Return ONLY the JSON object.

DOCUMENT:
{markdown}
"""


def extract_record(file_path: str) -> dict:
    """
    Full pipeline: PDF → markdown → local LLM → JSON record.
    Requires the local LLM (Ollama) to be available via the project llm_client.
    """
    from rag.llm_client import call_llm_json

    markdown, _ = pdf_to_markdown(file_path)
    record = call_llm_json(
        _EXTRACTION_PROMPT.format(markdown=markdown),
        system=_EXTRACTION_SYSTEM,
        default={},
    )
    if not isinstance(record, dict):
        return {}
    return _lift_blocks(record)


def _lift_blocks(record: dict) -> dict:
    """
    Flatten positional wrapper keys the model sometimes emits
    ('left_block'/'right_block'/'header_*') by lifting their contents to the top
    level — so parties like seller/client sit at the top regardless of phrasing.
    """
    wrappers = ("left_block", "right_block", "header", "header_left",
                "header_right", "block_left", "block_right")
    out: dict = {}
    for k, v in record.items():
        kl = str(k).lower()
        if isinstance(v, dict) and (kl in wrappers or kl.endswith("_block")):
            for kk, vv in v.items():
                out.setdefault(kk, vv)
        else:
            out.setdefault(k, v)
    return out


# ── Coverage check (deterministic omission detector) ──────────────────────────

_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "date", "no", "id",
    "tax", "of", "to", "in", "on", "inr", "usd", "pcs", "vat",
}


def _significant_tokens(text: str) -> set[str]:
    """Source content tokens worth verifying: numbers/codes and Capitalized words."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-./,]{1,}", text)
    out = set()
    for t in toks:
        tl = t.strip(".,").lower()
        if len(tl) < 2 or tl in _STOP:
            continue
        # keep tokens that carry a digit (amounts/codes/zips) or are Capitalized
        if any(ch.isdigit() for ch in t) or t[0].isupper():
            out.add(tl)
    return out


def _all_tokens(text: str) -> set[str]:
    """Every content token (case-insensitive) — used for what the record captured,
    so a source label that became a lowercase JSON key still counts as covered."""
    out = set()
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-./,]{1,}", text):
        tl = t.strip(".,").lower()
        if len(tl) >= 2 and tl not in _STOP:
            out.add(tl)
    return out


def _flatten(record) -> str:
    """Flatten keys AND values — a source label that became a JSON key counts as
    captured, so only genuinely dropped DATA is reported missing."""
    if isinstance(record, dict):
        parts = []
        for k, v in record.items():
            parts.append(str(k).replace("_", " "))
            parts.append(_flatten(v))
        return " ".join(parts)
    if isinstance(record, list):
        return " ".join(_flatten(v) for v in record)
    return str(record)


def coverage_check(record: dict, raw_text: str) -> dict:
    """
    Deterministically check the JSON accounts for the source. Returns coverage %
    and the significant source tokens that did NOT make it into the record — a
    safety net that a fixed schema cannot provide (it flags real omissions).
    """
    src = _significant_tokens(raw_text)
    got = _all_tokens(_flatten(record))
    if not src:
        return {"coverage_pct": 100.0, "missing": []}
    missing = sorted(src - got)
    covered = len(src) - len(missing)
    return {
        "coverage_pct": round(100.0 * covered / len(src), 1),
        "source_token_count": len(src),
        "missing": missing,
    }
