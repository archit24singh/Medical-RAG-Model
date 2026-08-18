"""
Folder-per-format extraction — scales structured-PDF extraction to millions.

Idea (user-declared templates): all documents of the same layout live in one
folder (e.g. bucket/invoice1/). Per folder:

  • The FIRST file is extracted with the LLM (markdown → JSON) as the reference.
  • Every other file is extracted DETERMINISTICALLY (pdfplumber tables +
    column-split, label-based header parsing) — no LLM.
  • Each deterministic result is scored by the coverage check; if it falls below
    the threshold (an odd-format file slipped in), it falls back to a single LLM
    call for that file only.

Result: N folders → ~N LLM calls + rare fallbacks; 1M same-format invoices in
one folder → 1 LLM call + 1M millisecond passes. Fully local, stays on 8B.

The deterministic parser is generic (label-based, not format-hardcoded): it
reads the document's own "Label: value" markers and party blocks, and uses
common heuristics (e.g. "City, State - PostalCode") — so a new folder/format
needs no code change, only its own reference LLM call.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional

from rag.markdown_extractor import (
    pdf_to_markdown, extract_record, coverage_check, _table_to_markdown,
)

logger = logging.getLogger(__name__)

_PDF_EXTS = (".pdf",)
_SECTION_NOISE = {"items", "summary", "item", "particulars", "details"}
_PARTY_FIELD_LABELS = {"tax_id", "gstin", "vat_id", "tax_number", "pan", "cin", "gst"}


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


# ── Deterministic header parsing (generic, label-based) ───────────────────────

def _parse_column(text: str) -> tuple[dict, dict]:
    """Parse one header column into (top_level_kv, parties)."""
    top: dict = {}
    parties: dict = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # bare all-caps single word (ITEMS, SUMMARY) → section boundary
        if line.isupper() and len(line.split()) <= 1 and line.lower() in _SECTION_NOISE:
            current = None
            continue

        m = re.match(r"^([A-Za-z][A-Za-z /.]{1,30}):\s*(.*)$", line)
        if m:
            key = _norm_key(m.group(1))
            val = m.group(2).strip()
            if val == "":
                # label-only line → start a party block (Seller:, Client:, Buyer:)
                current = key
                parties[current] = {"_lines": []}
            elif current and key in _PARTY_FIELD_LABELS:
                parties[current][key] = val
            else:
                top[key] = val
        else:
            if current is not None:
                parties[current]["_lines"].append(line)
            # else: stray non-labeled line outside any block → ignore

    # Post-process each party's free lines → name / address / city / state / postal
    for _, d in parties.items():
        lines = d.pop("_lines", [])
        if not lines:
            continue
        d["name"] = lines[0]
        addr_parts = []
        for l in lines[1:]:
            cm = re.match(r"^(.*?),\s*(.*?)\s*[-–]\s*([A-Za-z0-9][A-Za-z0-9 ]*)$", l)
            if cm:
                d["city"] = cm.group(1).strip()
                d["state"] = cm.group(2).strip()
                d["postal_code"] = cm.group(3).strip()
            else:
                addr_parts.append(l)
        if addr_parts:
            d["address"] = ", ".join(addr_parts)
    return top, parties


# ── Deterministic table parsing ───────────────────────────────────────────────

_ITEM_HINTS = ("description", "item", "product", "particular", "qty", "quantity",
               "service", "hsn", "sku")


def _table_rows(tbl_data: list[list]) -> list[dict]:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r]
            for r in tbl_data if any((c or "").strip() for c in r)]
    if len(rows) < 2:
        return []
    header = [(_norm_key(h) or f"col_{i}") for i, h in enumerate(rows[0])]
    out = []
    for r in rows[1:]:
        r = (r + [""] * len(header))[:len(header)]
        out.append({header[i]: r[i] for i in range(len(header))})
    return out


def _looks_like_items(header: list) -> bool:
    h = " ".join(str(x).lower() for x in header)
    return any(k in h for k in _ITEM_HINTS)


def deterministic_extract(file_path: str) -> dict:
    """Extract a record WITHOUT the LLM: column-split labeled header + tables."""
    import pdfplumber

    record: dict = {}
    line_items: list[dict] = []
    other_tables: list[list[dict]] = []

    with pdfplumber.open(file_path) as pdf:
        for pi, page in enumerate(pdf.pages):
            try:
                tables = page.find_tables()
            except Exception:
                tables = []
            if pi == 0:
                header_bottom = min((t.bbox[1] for t in tables), default=page.height)
                mid = page.width / 2.0
                left = page.crop((0, 0, mid, header_bottom)).extract_text() or ""
                right = page.crop((mid, 0, page.width, header_bottom)).extract_text() or ""
                for col in (left, right):
                    if not col.strip():
                        continue
                    top, parties = _parse_column(col)
                    for k, v in top.items():
                        record.setdefault(k, v)
                    for pk, pv in parties.items():
                        record.setdefault(pk, pv)
            for t in tables:
                try:
                    data = t.extract()
                except Exception:
                    continue
                rows = _table_rows(data)
                if not rows:
                    continue
                if _looks_like_items([*rows[0].keys()]):
                    line_items.extend(rows)
                else:
                    other_tables.append(rows)

    if line_items:
        record["line_items"] = line_items
    if other_tables:
        record["summary"] = other_tables[0] if len(other_tables[0]) == 1 else other_tables
    return record


# ── Folder orchestration ──────────────────────────────────────────────────────

def extract_folder(folder_path: str, coverage_threshold: float = 90.0) -> dict:
    """
    Extract every PDF in a folder. First file → LLM reference; rest deterministic
    with an LLM fallback when coverage is low. Returns records + call stats.
    """
    files = sorted(
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if f.lower().endswith(_PDF_EXTS)
    )
    records = []
    stats = {"folder": folder_path, "files": len(files),
             "llm_calls": 0, "deterministic": 0, "fallbacks": 0}

    for i, f in enumerate(files):
        _, raw = pdf_to_markdown(f)
        if i == 0:
            rec = extract_record(f)                 # LLM reference (learn the format)
            stats["llm_calls"] += 1
            method = "llm_reference"
        else:
            rec = deterministic_extract(f)
            cov = coverage_check(rec, raw).get("coverage_pct", 0.0)
            if cov < coverage_threshold:
                rec = extract_record(f)             # LLM fallback for the odd one out
                stats["llm_calls"] += 1
                stats["fallbacks"] += 1
                method = f"llm_fallback(cov={cov})"
            else:
                stats["deterministic"] += 1
                method = f"deterministic(cov={cov})"
        rec["_source_file"] = os.path.basename(f)
        rec["_method"] = method
        records.append(rec)

    logger.info("extract_folder %s: %s", folder_path, stats)
    return {"records": records, "stats": stats}
