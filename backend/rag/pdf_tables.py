"""
PDF structured/table extraction (Phase 1b).

The legacy PDF path treated PDFs as unstructured text only — table pages were
kept atomic but never parsed into fields, and nothing from a PDF reached the
structured store. This module recovers real tabular structure using PyMuPDF's
``page.find_tables()`` and turns each table row into:

  1. canonical Events (via the same data/concept_map.yaml adapter), so PDF
     tables become first-class events for ETHER/AIR/DER and SQL-queryable
     alongside CSV/Excel data; and
  2. a list of row dicts (returned to the caller for optional raw storage).

If a page has no text layer (scanned PDF), find_tables() yields nothing — the
OCR path in ocr.py handles those pages instead (Phase 1c).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_tables_from_pdf(file_path: str) -> list[dict]:
    """
    Extract every table from a PDF as a list of row dicts.

    Returns a flat list of dicts, one per table row, with keys taken from the
    table header row. A ``_page`` key records the source page (1-based).
    Returns [] if PyMuPDF is unavailable or no tables are found.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover
        logger.warning("PyMuPDF unavailable (%s) — PDF table extraction skipped", exc)
        return []

    rows: list[dict] = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.warning("Could not open PDF %s (%s)", file_path, exc)
        return []

    for page_index, page in enumerate(doc):
        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.debug("find_tables failed on page %d (%s)", page_index + 1, exc)
            continue

        for table in getattr(finder, "tables", []) or []:
            try:
                data = table.extract()
            except Exception:
                continue
            if not data or len(data) < 2:
                continue

            header = [_clean_header(c, i) for i, c in enumerate(data[0])]
            for raw_row in data[1:]:
                if not any(_nonempty(c) for c in raw_row):
                    continue
                row = {header[i]: raw_row[i] for i in range(min(len(header), len(raw_row)))}
                row["_page"] = page_index + 1
                rows.append(row)

    logger.info("extract_tables_from_pdf: %d row(s) from %s", len(rows), file_path)
    return rows


def _clean_header(cell, idx: int) -> str:
    txt = (str(cell).strip() if cell is not None else "").replace("\n", " ")
    return txt if txt else f"col_{idx}"


def _clean_cell(v) -> str:
    return "" if v is None else str(v).replace("\n", " ").strip()


def _extract_doc_metadata(text: str) -> dict:
    """Pull document-level key/value fields (invoice no, date) from the raw text."""
    import re
    meta = {}
    m = re.search(r"invoice\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Za-z0-9\-]{3,})", text, re.I)
    if m:
        meta["invoice_no"] = m.group(1)
    m = re.search(r"(?:date of issue|invoice date|date)\s*[:]?\s*"
                  r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2})", text, re.I)
    if m:
        meta["invoice_date"] = m.group(1)
    return meta


def extract_and_stage_pdf_tables(file_path: str) -> dict:
    """
    Extract the primary line-item table from a PDF with pdfplumber and load it
    into the SAME staging pipeline as CSV/Excel — giving structured PDFs full
    parity (profile / field lookup / aggregation / document lookup).

    Multi-page item lists (same header across pages) are concatenated. Document
    metadata (invoice no, date) is attached to every row so items can be scoped
    to one document. Returns a summary dict (or {"staged": 0} if no table found).
    """
    try:
        import pdfplumber
        import pandas as pd
    except Exception as exc:
        logger.warning("pdfplumber/pandas unavailable (%s) — PDF staging skipped", exc)
        return {"staged": 0}

    src = file_path.rsplit("/", 1)[-1]
    groups: dict[tuple, list[list]] = {}
    full_text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    full_text += (page.extract_text() or "") + "\n"
                except Exception:
                    pass
                for tbl in page.extract_tables() or []:
                    if not tbl or len(tbl) < 2:
                        continue
                    header = tuple(_clean_header(c, i) for i, c in enumerate(tbl[0]))
                    data_rows = [
                        [_clean_cell(c) for c in r]
                        for r in tbl[1:]
                        if any(_clean_cell(c) for c in r)
                    ]
                    if data_rows:
                        groups.setdefault(header, []).extend(data_rows)
    except Exception as exc:
        logger.warning("pdfplumber failed on %s (%s)", src, exc)
        return {"staged": 0}

    if not groups:
        return {"staged": 0}

    # Pick the LINE-ITEM table, not the summary/totals table. Prefer a header
    # that looks like line items (has a description/item/qty-style column), then
    # more columns, then more rows — so a 2-item invoice's summary table (few
    # columns, VAT/Net Worth only) never wins over its items table.
    _ITEM_HINTS = ("description", "item", "product", "particular", "qty",
                   "quantity", "service", "hsn", "sku")

    def _score(header_rows):
        header, rows = header_rows
        hnorm = " ".join(str(h).lower() for h in header)
        looks_like_items = 1 if any(h in hnorm for h in _ITEM_HINTS) else 0
        return (looks_like_items, len(header), len(rows))

    header, rows = max(groups.items(), key=_score)
    ncols = len(header)
    norm_rows = [(r + [""] * ncols)[:ncols] for r in rows]

    df = pd.DataFrame(norm_rows, columns=list(header))

    # Attach document-level metadata columns
    meta = _extract_doc_metadata(full_text)
    for k, v in meta.items():
        df[k] = v
    df["source_pdf"] = src

    # Route through the existing tabular/staging ingest via a temp CSV
    import tempfile, os
    from rag.ingestion import _ingest_tabular

    stem = src.rsplit(".", 1)[0]
    tmp_dir = tempfile.mkdtemp()
    tmp_csv = os.path.join(tmp_dir, f"{stem}.csv")
    try:
        df.to_csv(tmp_csv, index=False)
        result = _ingest_tabular(tmp_csv, ".csv")
    finally:
        try:
            os.remove(tmp_csv)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    logger.info("extract_and_stage_pdf_tables: staged %d row(s) from %s (%s)",
                len(df), src, result.get("status"))
    return {"staged": len(df), "columns": list(df.columns), "ingest": result}


def _nonempty(cell) -> bool:
    return cell is not None and str(cell).strip() != ""


def extract_and_persist_pdf_events(file_path: str, source_file: Optional[str] = None) -> int:
    """
    Extract tables from a PDF, map each row to canonical Events, and persist them.

    Returns the number of events inserted. Safe no-op (returns 0) if the PDF has
    no parseable tables (e.g. pure-prose or scanned PDFs).
    """
    from rag.events import extract_events_from_row, persist_events

    rows = extract_tables_from_pdf(file_path)
    if not rows:
        return 0

    src = source_file or file_path.rsplit("/", 1)[-1]
    events = []
    for row in rows:
        clean = {k: v for k, v in row.items() if not str(k).startswith("_")}
        events.extend(extract_events_from_row(clean, source_file=src))

    inserted = persist_events(events) if events else 0
    logger.info("extract_and_persist_pdf_events: %d event(s) from %s", inserted, src)
    return inserted
