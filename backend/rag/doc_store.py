"""
Storage wiring for folder-per-format extraction.

For a folder of same-format documents:
  1. extract_folder() → records (1 LLM reference + deterministic rest).
  2. Full record → `documents` JSONB table (schema-agnostic, nothing dropped).
  3. Flattened header rows + line-item rows → the EXISTING staging pipeline
     (schema-on-read): same layout → one signature → one stg_ table with real
     columns → auto-exposed as a v_* view → reliable SQL + the deterministic
     query paths (document/field/aggregate) all work over it.

Nothing is hardcoded: header fields become columns from the record's OWN keys,
so a new format lands in its own staging table automatically.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

_SKIP_HEADER_KEYS = {"line_items", "summary", "tables", "tables_2", "_method", "text"}


def _flatten_header(record: dict) -> dict:
    """Flatten a record's header into a single flat row (nested objects → prefixed keys)."""
    out: dict = {}
    for k, v in record.items():
        if k in _SKIP_HEADER_KEYS:
            continue
        if k == "_source_file":
            out["source_file"] = v
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (str, int, float)):
                    out[f"{k}_{kk}"] = vv
        elif isinstance(v, (str, int, float)):
            out[k] = v
    return out


def _flatten_items(record: dict) -> list[dict]:
    """One flat row per line item, keyed back to the document."""
    items = record.get("line_items") or []
    dockey = record.get("invoice_no") or record.get("_source_file")
    rows = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        row = {k: v for k, v in it.items() if isinstance(v, (str, int, float))}
        row["invoice_no"] = dockey
        row["source_file"] = record.get("_source_file")
        row["item_index"] = idx
        rows.append(row)
    return rows


def _store_documents_jsonb(records: list[dict], folder: str) -> int:
    from psycopg2.extras import Json, execute_values
    from db.schema import get_db

    rows = [
        (r.get("_source_file"), folder, r.get("doc_type") or "document", Json(r))
        for r in records
    ]
    with get_db() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (source_file, doc_folder, doc_type, data) VALUES %s "
                "ON CONFLICT (source_file) DO UPDATE "
                "SET data = EXCLUDED.data, doc_folder = EXCLUDED.doc_folder, "
                "doc_type = EXCLUDED.doc_type",
                rows,
            )
    return len(rows)


def _stage_rows(rows: list[dict], label: str, folder: str) -> Optional[str]:
    """Write flat rows to the staging pipeline via a single temp CSV (one call)."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    import pandas as pd
    from rag.ingestion import _ingest_tabular

    df = pd.DataFrame(rows)
    tmp_dir = tempfile.mkdtemp()
    tmp_csv = os.path.join(tmp_dir, f"{folder}_{label}.csv")
    try:
        df.to_csv(tmp_csv, index=False)
        res = _ingest_tabular(tmp_csv, ".csv")
        return res.get("status")
    finally:
        try:
            os.remove(tmp_csv)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def ingest_folder(folder_path: str, coverage_threshold: float = 90.0) -> dict:
    """
    Full folder-per-format ingest: extract → JSONB store → flatten to staging →
    auto-expose views. Returns extraction stats + storage summary.
    """
    from rag.folder_extractor import extract_folder

    result = extract_folder(folder_path, coverage_threshold=coverage_threshold)
    records = result["records"]
    folder = os.path.basename(folder_path.rstrip("/")) or "documents"
    if not records:
        return {"records": 0, "stats": result["stats"]}

    stored = _store_documents_jsonb(records, folder)

    headers = [_flatten_header(r) for r in records]
    items: list[dict] = []
    for r in records:
        items.extend(_flatten_items(r))

    staged = {
        "headers": _stage_rows(headers, "headers", folder),
        "items": _stage_rows(items, "items", folder),
    }

    # Expose the new staging streams as v_* views + index them for querying.
    try:
        from db.schema import auto_expose_streams
        auto_expose_streams()
    except Exception as exc:
        logger.warning("auto_expose after folder ingest failed: %s", exc)
    try:
        from rag.profiles import ensure_entity_indexes
        ensure_entity_indexes()
    except Exception as exc:
        logger.warning("entity index after folder ingest failed: %s", exc)

    summary = {
        "records": len(records),
        "documents_stored": stored,
        "staged": staged,
        "stats": result["stats"],
    }
    logger.info("ingest_folder %s: %s", folder_path, summary)
    return summary
