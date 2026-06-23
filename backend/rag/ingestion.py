"""
Document ingestion pipeline — supports PDF, CSV, Excel, TXT, MD, JSON.

TEXT DOCUMENT pipeline (PDF / DOCX / TXT / MD):
  1. Extract text   — PyMuPDF (fitz) for PDFs; plain-text reader for others
  2. Chunk          — sliding-window splitter (1000 chars / 200 overlap)
  3. Header         — prepend "<Title> [chunk N/M]:" before each chunk
  4. Store          — ChromaDB (one entry per chunk, nomic-embed-text embeddings)

  Zero LLM calls.  Replaces the old Docling → LLM-chunker → enricher →
  structured_extractor chain that made 200+ Ollama calls per PDF.

TABULAR pipeline (CSV / Excel):
  PostgreSQL — direct columnar mapping via _write_rows_to_sql (no LLM)
  ChromaDB   — NOT USED for tabular data.  Structured rows go to PostgreSQL
               only; this keeps ChromaDB clean for unstructured knowledge PDFs
               and eliminates confusion when billing rows appear in RAG results.

JSON pipeline:
  Direct field-map extraction, stored as a single ChromaDB doc.

Storage note:
  Currently reads from the local bucket/ folder.
  To upgrade to S3: replace Path(...).rglob() calls with boto3 S3 listing,
  and replace open() file reads with s3.get_object() calls.
"""
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import yaml

from config import settings
from rag.vectorstore import add_document, add_documents_batch

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Tabular
    ".csv", ".xlsx", ".xls",
    # Structured
    ".json",
    # Text documents — PyMuPDF / plain-text reader
    ".pdf", ".txt", ".md",
    ".html", ".htm", ".xml",
    ".docx", ".doc",
}

# Cell values treated as missing even if non-null
_NULL_VALUES = {"", "nan", "none", "n/a", "na", "-", "--", "null", "undefined"}


# ── Field map (kept for JSON / text pipeline metadata extraction) ─────────────
# Not used by the tabular path — tabular uses structure-defined streams.

_FIELD_MAP = {
    "patient_name":  ["patient_name", "patient", "name", "full_name", "member_name"],
    "patient_id":    ["patient_id", "id", "mrn", "patient_number", "member_id",
                      "patient acct no", "subject_id"],
    "date":          ["date", "bill_date", "service_date", "date_of_service", "visit_date",
                      "service date", "claim date", "start date of service"],
    "doc_type":      ["doc_type", "document_type", "type", "record_type"],
    "provider_name": ["provider_name", "provider", "physician", "doctor", "physician_name",
                      "rendering provider", "appointment / servicing provider"],
    "provider_npi":  ["provider_npi", "npi", "npi_number"],
    "provider_dob":  ["provider_dob", "dob", "date_of_birth"],
    "total_amount":  ["total_amount", "total", "amount", "bill_amount", "billed charge",
                      "total payment", "balance", "amount_due"],
}


# ── PHI denylist — columns that must NEVER be stored (global defaults) ────────
# Per-stream extra denylists are configured in sources.yaml → phi_denylist.
# NOTE: The denylist controls what is STORED, not stream identity.
# Stream identity (column_signature) is computed BEFORE the denylist is applied.

_GLOBAL_PHI_DENYLIST: frozenset[str] = frozenset({
    "ssn",
    "social_security",
    "social_sec",
    "password",
    "passwd",
})


# ── Data cleaning utilities ───────────────────────────────────────────────────

def _clean_str(val) -> Optional[str]:
    """Normalize a cell value to a clean string or None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in _NULL_VALUES:
        return None
    return s


def _normalize_date(val: str) -> Optional[str]:
    """Parse a date string into YYYY-MM-DD; return original if parsing fails."""
    if not val:
        return None
    try:
        import pandas as pd
        dt = pd.to_datetime(val, infer_datetime_format=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return val


def _safe_cell(val) -> Optional[str]:
    """Extract a scalar string from a cell, handling NaN and Series edge cases."""
    import pandas as pd
    if isinstance(val, pd.Series):
        val = val.iloc[0] if not val.empty else None
    return _clean_str(val)


# ── Safe SQL identifier ───────────────────────────────────────────────────────

def _safe_identifier(raw: str) -> str:
    """
    Convert a raw column header to a safe lowercase PostgreSQL identifier.
    Raises ValueError if the result is empty (empty or all-punctuation headers).
    Leading digits are prefixed with 'c' (not '_' — that prefix is reserved
    for system columns like _source_file, _row_hash, _deleted_at).
    """
    s = str(raw).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)  # non-alphanum → underscore
    s = s.strip("_")                    # strip leading/trailing underscores
    if not s:
        raise ValueError(f"Header {raw!r} produced an empty identifier")
    if s[0].isdigit():
        s = "c" + s
    return s


# ── Structure-defined stream infrastructure ───────────────────────────────────

_DRIFT_THRESHOLD = 0.70    # Jaccard similarity above which we flag potential drift

# Module-level YAML config cache; invalidated by catalog_admin and force_reload
_sources_yaml_cache: Optional[dict] = None


def _load_sources_yaml(force_reload: bool = False) -> dict:
    """
    Load (or return cached) data/sources.yaml.
    Returns {} if the file does not exist.
    """
    global _sources_yaml_cache
    if _sources_yaml_cache is not None and not force_reload:
        return _sources_yaml_cache

    yaml_path = settings.SOURCES_YAML_FILE
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _sources_yaml_cache = data
        logger.info(
            "Loaded %d stream config(s) from %s",
            len(data.get("streams", [])), yaml_path,
        )
    except FileNotFoundError:
        logger.info(
            "sources.yaml not found at %s — no pre-declared stream configs",
            yaml_path,
        )
        _sources_yaml_cache = {}
    except Exception as exc:
        logger.warning("Failed to load sources.yaml (%s): %s", yaml_path, exc)
        _sources_yaml_cache = {}
    return _sources_yaml_cache


def _get_yaml_config_for_signature(signature: str) -> dict:
    """Return the sources.yaml entry for this full signature, or {}."""
    data = _load_sources_yaml()
    for stream in data.get("streams", []):
        # Fix 4: full 32-char signature match, not 8-char prefix
        if stream.get("column_signature") == signature:
            return stream
    return {}


def _apply_yaml_config_to_catalog(conn, signature: str, config: Optional[dict] = None) -> None:
    """
    Apply sources.yaml config overlay to source_catalog for the given signature.
    If config is None, fetches it from the YAML file.
    Only updates fields explicitly present in config (never NULLs an existing value).
    """
    from db.schema import fetchone_dict
    if config is None:
        config = _get_yaml_config_for_signature(signature)
    if not config:
        return

    set_clauses: list[str] = []
    values: list = []

    def _add(col: str, val, as_jsonb: bool = False) -> None:
        if val is None:
            return
        if as_jsonb:
            set_clauses.append(f"{col} = %s::jsonb")
            values.append(json.dumps(val))
        else:
            set_clauses.append(f"{col} = %s")
            values.append(val)

    if "human_label"    in config: _add("human_label",    config["human_label"])
    if "load_mode"      in config: _add("load_mode",      config["load_mode"])
    if "natural_key"    in config: _add("natural_key",    config["natural_key"],    as_jsonb=True)
    if "query_exposed"  in config: _add("query_exposed",  config["query_exposed"])
    if "column_mapping" in config: _add("column_mapping", config["column_mapping"], as_jsonb=True)

    if not set_clauses:
        return

    values.append(signature)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE source_catalog SET {', '.join(set_clauses)} "
            f"WHERE column_signature = %s",
            values,
        )
    conn.commit()


def _compute_signature(raw_columns: list[str]) -> tuple[str, list[str]]:
    """
    Pass 1 — denylist-free.

    Sanitise ALL headers → safe identifiers → sort + dedupe → md5.
    This is the stream identity.  Denylist changes never change this value.

    Returns (signature_hex, sorted_safe_columns_pre_denylist).
    Raises ValueError if zero valid identifiers result.
    """
    safe: list[str] = []
    for col in raw_columns:
        try:
            safe.append(_safe_identifier(col))
        except ValueError:
            logger.warning(
                "Empty identifier from header %r — excluded from signature", col
            )
    if not safe:
        raise ValueError(
            "All headers produced empty identifiers — rejecting file"
        )
    unique_sorted = sorted(set(safe))
    sig = hashlib.md5("|".join(unique_sorted).encode()).hexdigest()
    return sig, unique_sorted


def _apply_denylist(
    all_safe:        list[str],
    extra_denylist:  frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """
    Pass 2 — controls what lands in PostgreSQL.  Never touches the signature.

    Returns (stored_columns, dropped_columns).
    Raises ValueError if zero stored columns remain (file is 100 % denylist).
    """
    denylist = _GLOBAL_PHI_DENYLIST | extra_denylist
    stored: list[str] = []
    dropped: list[str] = []
    for col in all_safe:
        if any(token in col for token in denylist):
            logger.warning("PHI denylist: column %r will not be stored", col)
            dropped.append(col)
        else:
            stored.append(col)
    if not stored:
        raise ValueError(
            f"All columns dropped by PHI denylist — rejecting file. "
            f"Dropped: {dropped}"
        )
    return stored, dropped


def _detect_drift(
    new_sig:   str,
    new_safe:  set[str],
    conn,
) -> Optional[str]:
    """
    Return the column_signature of an existing stream whose columns overlap
    ≥ DRIFT_THRESHOLD (Jaccard) with new_safe, or None.
    Does NOT auto-merge — records the candidate for admin review only.
    """
    from db.schema import fetchall_dicts
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_signature, safe_columns "
            "FROM source_catalog "
            "WHERE aliased_to IS NULL AND staging_table IS NOT NULL"
        )
        rows = cur.fetchall()

    for existing_sig, safe_cols_json in rows:
        if existing_sig == new_sig:
            continue
        try:
            existing_safe = set(
                json.loads(safe_cols_json)
                if isinstance(safe_cols_json, str)
                else (safe_cols_json or [])
            )
        except Exception:
            continue
        if not existing_safe:
            continue
        union = new_safe | existing_safe
        jaccard = len(new_safe & existing_safe) / len(union) if union else 0.0
        if jaccard >= _DRIFT_THRESHOLD:
            logger.warning(
                "Drift detected: new sig %s overlaps %.0f%% with existing "
                "stream %s — run `catalog_admin alias` if these are the same "
                "stream, or ignore if they are intentionally different",
                new_sig[:8], jaccard * 100, existing_sig[:8],
            )
            return existing_sig
    return None


def _resolve_stream(
    signature:    str,
    all_safe:     list[str],
    stored_cols:  list[str],
    raw_headers:  list[str],
    conn,
) -> dict:
    """
    Look up signature in source_catalog.

    HIT + aliased_to  → follow alias to target; sync extra cols; return target.
    HIT no alias      → apply YAML config overlay (Fix 4); return updated row.
    MISS              → detect drift; register new stream; create table; return.

    Never auto-merges — alias must be set explicitly by catalog_admin.
    """
    from db.schema import (
        fetchone_dict, _create_staging_table, _sync_staging_columns,
    )

    def _parse_json_field(val):
        if val is None:
            return []
        if isinstance(val, (list, dict)):
            return val
        try:
            return json.loads(val)
        except Exception:
            return []

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM source_catalog WHERE column_signature = %s",
            [signature],
        )
        row = fetchone_dict(cur)

    if row:
        # ── Alias ────────────────────────────────────────────────────────────
        if row.get("aliased_to"):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM source_catalog WHERE column_signature = %s",
                    [row["aliased_to"]],
                )
                target = fetchone_dict(cur)
            if not target or target.get("aliased_to"):
                raise RuntimeError(
                    f"Alias target {(row.get('aliased_to') or '')[:8]!r} "
                    "not found or is itself an alias — chained aliases are not allowed"
                )
            # Ensure any columns carried by this sig exist in the target table
            my_stored     = set(_parse_json_field(row.get("stored_columns")))
            target_stored = set(_parse_json_field(target.get("stored_columns")))
            extra         = [c for c in all_safe if c in my_stored and c not in target_stored]
            if extra:
                _sync_staging_columns(conn, target["staging_table"], extra)
                new_stored = sorted(target_stored | set(extra))
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE source_catalog SET stored_columns = %s::jsonb "
                        "WHERE column_signature = %s",
                        [json.dumps(new_stored), target["column_signature"]],
                    )
                conn.commit()
            return dict(target)

        # ── Known stream — apply YAML overlay at lookup time (Fix 4) ─────────
        _apply_yaml_config_to_catalog(conn, signature)

        # Ensure stored_cols (post-denylist from THIS file) are all present
        if row.get("staging_table"):
            _sync_staging_columns(conn, row["staging_table"], stored_cols)

        # Re-fetch to pick up YAML-applied changes
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM source_catalog WHERE column_signature = %s",
                [signature],
            )
            row = fetchone_dict(cur)
        return dict(row)

    # ── New signature ─────────────────────────────────────────────────────────
    staging_table = "stg_" + signature[:8]

    # Load any pre-declared YAML config for this signature (config may pre-date first file)
    yaml_config = _get_yaml_config_for_signature(signature)

    # Drift detection — logs a warning, never auto-merges
    candidate_drift = _detect_drift(signature, set(all_safe), conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_catalog
                (column_signature, staging_table, representative_headers,
                 safe_columns, stored_columns, load_mode, query_exposed,
                 candidate_drift_of, first_ingested_at)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s, NOW())
            ON CONFLICT (column_signature) DO NOTHING
            """,
            [
                signature,
                staging_table,
                json.dumps(raw_headers),
                json.dumps(all_safe),
                json.dumps(stored_cols),
                yaml_config.get("load_mode", "append"),
                yaml_config.get("query_exposed", False),
                candidate_drift,
            ],
        )
    conn.commit()

    # Apply any remaining YAML fields (human_label, column_mapping, natural_key …)
    _apply_yaml_config_to_catalog(conn, signature, yaml_config)

    _create_staging_table(conn, staging_table, stored_cols)

    return {
        "column_signature": signature,
        "staging_table":    staging_table,
        "aliased_to":       None,
        "safe_columns":     json.dumps(all_safe),
        "stored_columns":   json.dumps(stored_cols),
        "load_mode":        yaml_config.get("load_mode", "append"),
        "natural_key":      yaml_config.get("natural_key"),
        "query_exposed":    yaml_config.get("query_exposed", False),
        "column_mapping":   yaml_config.get("column_mapping"),
        "view_name":        None,
    }


def _write_rows_to_staging(
    conn,
    staging_table:  str,
    stored_cols:    list[str],
    safe_rows:      list[dict],
    row_hashes:     list[str],
    filename:       str,
    load_mode:      str,
) -> int:
    """
    Insert rows into the staging table, skipping exact duplicates.

    append mode   : skip rows whose (hash, file) combo already exists.
    snapshot mode : skip rows whose hash already exists for this file with
                    _deleted_at IS NULL.  Reconciliation (soft-delete of
                    absent rows) is handled separately by
                    _reconcile_snapshot_deletions in schema.py.

    Returns number of rows inserted.
    """
    from psycopg2.extras import execute_values
    from db.schema import _quote_identifier

    qtable = _quote_identifier(staging_table)

    # Fetch existing hashes to determine what to skip
    with conn.cursor() as cur:
        if load_mode == "snapshot":
            cur.execute(
                f"SELECT _row_hash FROM {qtable} "
                f"WHERE _source_file = %s AND _deleted_at IS NULL",
                [filename],
            )
        else:
            # append: never re-insert the exact same row from the same file
            cur.execute(
                f"SELECT _row_hash FROM {qtable} WHERE _source_file = %s",
                [filename],
            )
        existing = {row[0] for row in cur.fetchall()}

    # Build quoted column list
    col_quoted = (
        [_quote_identifier(c) for c in stored_cols]
        + ['"_source_file"', '"_row_hash"']
    )
    insert_sql = (
        f"INSERT INTO {qtable} ({', '.join(col_quoted)}) VALUES %s"
    )

    batch = []
    seen_this_run: set[str] = set()
    for safe_row, rh in zip(safe_rows, row_hashes):
        if rh in existing or rh in seen_this_run:
            continue
        batch.append([safe_row.get(c) for c in stored_cols] + [filename, rh])
        seen_this_run.add(rh)

    if not batch:
        return 0

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, batch)
        count = cur.rowcount

    conn.commit()
    return count


# ── Metadata helpers ──────────────────────────────────────────────────────────

_CATEGORY_DOC_TYPE_MAP = {
    "discharge summary": "record",
    "nursing":           "record",
    "nursing/other":     "record",
    "physician":         "record",
    "physician ":        "record",
    "general":           "record",
    "case management":   "record",
    "consult":           "record",
    "radiology":         "lab_result",
    "ecg":               "lab_result",
    "echo":              "lab_result",
    "rehab services":    "record",
    "social work":       "record",
    "pharmacy":          "prescription",
}


def _doc_type_from_filename(filename: str) -> str:
    fn = filename.lower()
    if "bill" in fn or "invoice" in fn:
        return "bill"
    if "provider" in fn or "npi" in fn or "doctor" in fn:
        return "provider_info"
    if "prescription" in fn or "rx" in fn:
        return "prescription"
    if "lab" in fn:
        return "lab_result"
    if "record" in fn:
        return "record"
    return "other"


def _doc_type_from_category(category: str | None) -> str | None:
    if not category:
        return None
    return _CATEGORY_DOC_TYPE_MAP.get(category.strip().lower())


def _meta_from_filename(filename: str) -> dict:
    meta = {k: None for k in list(_FIELD_MAP.keys()) + ["doc_type", "summary"]}
    meta["doc_type"] = _doc_type_from_filename(filename)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if date_match:
        meta["date"] = date_match.group()
    meta["summary"] = "Document: " + filename
    return meta


def _extract_meta_from_row(row: dict, filename: str, row_idx: int, total_rows: int) -> dict:
    """Extract and normalize metadata from a single data row."""
    lower_row = {k.lower(): v for k, v in row.items() if k}
    meta = {k: None for k in list(_FIELD_MAP.keys()) + ["doc_type", "summary"]}

    for meta_key, candidates in _FIELD_MAP.items():
        for candidate in candidates:
            if candidate in lower_row:
                val = _clean_str(lower_row[candidate])
                if val:
                    meta[meta_key] = val
                    break

    if meta.get("date"):
        meta["date"] = _normalize_date(meta["date"]) or meta["date"]
    if meta.get("patient_name"):
        meta["patient_name"] = meta["patient_name"].title()
    if meta.get("provider_name"):
        meta["provider_name"] = meta["provider_name"].title()

    if not meta["doc_type"]:
        meta["doc_type"] = (
            _doc_type_from_category(lower_row.get("category"))
            or _doc_type_from_filename(filename)
        )

    entity = meta.get("patient_name") or meta.get("provider_name") or "unknown"
    meta["summary"] = (
        f"{meta.get('doc_type', 'Record')} for {entity}"
        f" (row {row_idx + 1}/{total_rows} of {filename})"
    )
    meta["row_index"] = str(row_idx)
    return meta


def _extract_meta_structured(data: object, filename: str) -> dict:
    """Map a JSON object's keys directly to metadata fields."""
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return _meta_from_filename(filename)

    lower_data = {k.lower(): v for k, v in data.items()}
    meta = {k: None for k in _FIELD_MAP}
    meta["doc_type"] = None
    meta["summary"] = None

    for meta_key, candidates in _FIELD_MAP.items():
        for candidate in candidates:
            if candidate in lower_data and lower_data[candidate] is not None:
                meta[meta_key] = str(lower_data[candidate])
                break

    if not meta["doc_type"]:
        meta["doc_type"] = _doc_type_from_filename(filename)

    entity = meta.get("patient_name") or meta.get("provider_name") or "unknown"
    meta["summary"] = (
        str(meta.get("doc_type", "Document"))
        + " for " + entity
        + " dated " + str(meta.get("date", "unknown date"))
    )
    return meta


# ── Document ID helpers ───────────────────────────────────────────────────────

def _row_doc_id(file_path: str, row_idx: int) -> str:
    """Stable ID for a single row — updates in place on re-ingest."""
    key = f"{Path(file_path).resolve()}::row::{row_idx}"
    return hashlib.md5(key.encode()).hexdigest()


def _doc_id(file_path: str) -> str:
    return hashlib.md5(str(Path(file_path).resolve()).encode()).hexdigest()




# ── Text-document helpers ─────────────────────────────────────────────────────

def _split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Sliding-window character splitter — zero dependencies, zero LLM calls.

    Mirrors LangChain's RecursiveCharacterTextSplitter(chunk_size, chunk_overlap,
    length_function=len).  Tries to break at the cleanest boundary available
    (paragraph → line → space → character).

    Bug fix: only search for separators in the SECOND HALF of the window
    (start + chunk_size//2 … end).  The original code searched the full window,
    so rfind could return a separator very close to `start`, making
    `cut - chunk_overlap` fall below `start` and clamping the advance to just
    1 character per iteration — turning a 315K-char PDF into 23K micro-chunks
    instead of ~393 proper ones.  By restricting the search to the second half
    we guarantee at least chunk_size//2 chars of forward progress per chunk.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    min_advance = chunk_size // 2   # never step forward by less than this

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end == len(text):
            # Last segment — take everything remaining
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Only look for separators in the second half of the window so the
        # chosen cut point is always well past `start + min_advance`.
        cut = end
        search_from = start + min_advance
        for sep in ("\n\n", "\n", " "):
            pos = text.rfind(sep, search_from, end)
            if pos != -1:
                cut = pos + len(sep)
                break

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # Step forward; overlap pulls start back but never below last cut
        next_start = cut - chunk_overlap
        start = next_start if next_start > start else cut

    return chunks


# ── Table-aware PDF chunker ───────────────────────────────────────────────────

def _page_has_table(page) -> bool:
    """
    Heuristic: a PDF page looks like a data table if it contains many short
    text blocks arranged in a grid (as opposed to long flowing prose paragraphs).

    This catches ICD/CPT code tables, billing grids, HCPCS fee schedules, and
    other structured data layouts that should NOT be split across chunk boundaries.

    Thresholds (empirically tuned on medical billing PDFs):
      • ≥ 8 text blocks on the page
      • average block length ≤ 200 characters
      • ≥ 70 % of blocks are "short" (≤ 120 chars, i.e. a code + description)
    """
    try:
        # fitz block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text, 1 = image
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    except Exception:
        return False

    if len(text_blocks) < 8:
        return False

    total_len = sum(len(b[4]) for b in text_blocks)
    avg_len   = total_len / len(text_blocks)
    short_frac = sum(1 for b in text_blocks if len(b[4]) <= 120) / len(text_blocks)

    return avg_len <= 200 and short_frac >= 0.70


def _split_pdf_table_aware(
    file_path:    str,
    chunk_size:   int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Table-aware PDF chunker — keeps table pages atomic, splits prose normally.

    Algorithm
    ---------
    1. Open with PyMuPDF and process the PDF page by page.
    2. For each page, classify it as "table" or "prose" using _page_has_table().
    3. Accumulate consecutive prose pages; when a table page is reached (or at
       EOF), flush the prose buffer through the standard sliding-window splitter,
       then add the table page as one atomic chunk.
    4. Repeat until all pages are processed.

    ICD/CPT code preservation
    --------------------------
    Because table pages are kept whole, lines like:
        "410.0   Acute myocardial infarction of anterolateral wall"
    always stay in the same chunk as their code — the code and description can
    never be separated by a chunk boundary.

    Falls back to plain text extraction + sliding-window splitting if PyMuPDF
    is unavailable or the file cannot be opened as a structured PDF.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.warning(
            "Table-aware chunker: PyMuPDF could not open %s (%s) — "
            "falling back to plain text extraction",
            file_path, exc,
        )
        text = _extract_text_from_file(file_path, ".pdf")
        return _split_text_into_chunks(text, chunk_size, chunk_overlap)

    # Classify each page as "table" or "prose"
    segments: list[tuple[str, str]] = []  # (type, page_text)
    try:
        for page in doc:
            page_text = page.get_text().strip()
            if not page_text:
                continue
            seg_type = "table" if _page_has_table(page) else "prose"
            segments.append((seg_type, page_text))
    finally:
        doc.close()

    if not segments:
        return []

    chunks: list[str] = []
    prose_buffer: list[str] = []

    for seg_type, seg_text in segments:
        if seg_type == "prose":
            prose_buffer.append(seg_text)
        else:
            # Flush accumulated prose before the table
            if prose_buffer:
                prose_full = "\n\n".join(prose_buffer)
                chunks.extend(
                    _split_text_into_chunks(prose_full, chunk_size, chunk_overlap)
                )
                prose_buffer = []

            # Keep the table page as one atomic chunk (never split mid-table).
            # If the page is larger than chunk_size, still keep it whole —
            # splitting a table risks separating a code from its description.
            if seg_text.strip():
                chunks.append(seg_text.strip())

    # Flush any remaining prose after the last table (or the entire document
    # if it contained no table pages at all).
    if prose_buffer:
        prose_full = "\n\n".join(prose_buffer)
        chunks.extend(
            _split_text_into_chunks(prose_full, chunk_size, chunk_overlap)
        )

    return chunks


def _extract_text_from_file(file_path: str, ext: str) -> str:
    """
    Extract raw text from a document using the appropriate reader.

    PDF      → PyMuPDF (fitz) with pypdf fallback
    DOCX/DOC → PyMuPDF (can open DOCX via internal format support)
    Others   → plain UTF-8 read (tabs stripped, HTML tags removed for HTML files)
    """
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except Exception as exc_fitz:
            logger.warning("PyMuPDF failed for %s (%s) — falling back to pypdf", file_path, exc_fitz)
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc_pypdf:
                raise RuntimeError(
                    f"Both PyMuPDF and pypdf failed for {file_path}: {exc_pypdf}"
                ) from exc_pypdf

    if ext in (".docx", ".doc"):
        try:
            import fitz
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as exc:
            # Last-resort: read as text (works for some .doc files)
            logger.warning("fitz failed for %s (%s) — reading as plain text", file_path, exc)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    if ext in (".html", ".htm"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        # Strip tags — good enough for text retrieval
        return re.sub(r"<[^>]+>", " ", raw)

    # .txt, .md, .xml, and anything else — plain read
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().replace("\t", " ")


def _row_to_chroma_text(row: dict, filename: str) -> str:
    """
    Convert one spreadsheet row to a ChromaDB document string.
    Format: "[Source: <file>] Col1: Val1 | Col2: Val2 | …"
    """
    parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
    body = " | ".join(parts)
    return f"[Source: {filename}] {body}" if body else ""


# ── Tabular ingestion (CSV / Excel) — structure-defined streams ───────────────

def _ingest_tabular(file_path: str, ext: str) -> dict:
    """
    Ingest a CSV or Excel file into a per-stream staging table in PostgreSQL.

    Stream identity is derived from the column signature (md5 of pre-denylist
    safe identifiers) so every unique schema gets its own stg_* table.
    ChromaDB is NOT written — tabular rows are structured data only.

    Pipeline
    --------
    1. Load file → pandas DataFrame
    2. Compute column_signature (pre-denylist — stream identity never changes)
    3. Apply PHI denylist → stored_cols (subset kept in Postgres)
    4. Resolve stream via source_catalog (create stg_* on first sight)
    5. Build safe-identifier row dicts + row hashes
    6. INSERT new rows (skip exact duplicates)
    7. Reconcile soft-deletions for snapshot streams
    8. Update catalog watermarks
    """
    import pandas as pd
    from db.schema import (
        get_db, _reconcile_snapshot_deletions,
    )

    path     = Path(file_path)
    filename = path.name

    # 1. Load
    try:
        df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to load {filename}: {exc}"}

    if df.empty:
        return {"status": "error", "message": f"No data found in {filename}"}

    df.columns  = [str(c).strip() for c in df.columns]
    df          = df.dropna(how="all").reset_index(drop=True)
    total_rows  = len(df)
    if total_rows == 0:
        return {"status": "error", "message": f"No data rows in {filename}"}

    raw_headers = list(df.columns)
    logger.info("Loaded %d rows from %s", total_rows, filename)

    # 2. Compute signature (pre-denylist — Fix 3)
    try:
        signature, all_safe = _compute_signature(raw_headers)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    # 3. Apply PHI denylist (post-signature — Fix 3)
    try:
        stored_cols, dropped_cols = _apply_denylist(all_safe)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if dropped_cols:
        logger.info(
            "PHI denylist dropped %d column(s) from %s: %s",
            len(dropped_cols), filename, dropped_cols,
        )

    # 4. Resolve stream
    try:
        with get_db() as conn:
            stream = _resolve_stream(signature, all_safe, stored_cols, raw_headers, conn)
    except Exception as exc:
        return {
            "status":  "error",
            "message": f"Stream resolution failed for {filename}: {exc}",
        }

    staging_table = stream["staging_table"]
    load_mode     = stream.get("load_mode") or "append"

    # Build raw-header → safe-identifier mapping
    raw_to_safe: dict[str, str] = {}
    for raw_h in raw_headers:
        try:
            safe = _safe_identifier(raw_h)
            raw_to_safe[raw_h] = safe
        except ValueError:
            pass

    # 5. Build safe-identifier row dicts + row hashes
    safe_rows:  list[dict] = []
    row_hashes: list[str]  = []

    for idx in range(total_rows):
        safe_row: dict = {}
        for raw_h in raw_headers:
            safe_h = raw_to_safe.get(raw_h)
            if safe_h and safe_h in stored_cols:
                try:
                    safe_row[safe_h] = _safe_cell(df.at[idx, raw_h])
                except Exception:
                    safe_row[safe_h] = None

        hash_content = "|".join(
            f"{c}={safe_row.get(c) or ''}" for c in sorted(stored_cols)
        )
        safe_rows.append(safe_row)
        row_hashes.append(hashlib.md5(hash_content.encode()).hexdigest())

    # 6 & 7. Insert + reconcile
    with get_db() as conn:
        inserted = _write_rows_to_staging(
            conn, staging_table, stored_cols, safe_rows, row_hashes, filename, load_mode,
        )
        logger.info(
            "Staging insert: %d/%d row(s) → %s (%s)",
            inserted, total_rows, staging_table, load_mode,
        )

        deleted = -1
        if load_mode == "snapshot":
            deleted = _reconcile_snapshot_deletions(
                conn, staging_table, filename, row_hashes, total_rows,
            )

        # 8. Update catalog watermarks
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE source_catalog
                SET rows_total         = rows_total + %s,
                    rows_last_inserted = %s,
                    last_ingested_at   = NOW()
                WHERE column_signature = %s
                """,
                [inserted, inserted, signature],
            )
        conn.commit()

    result: dict = {
        "status":            "success",
        "doc_id":            _doc_id(file_path),
        "file_name":         filename,
        "rows_ingested":     inserted,
        "rows_processed":    total_rows,
        "staging_table":     staging_table,
        "column_signature":  signature[:8],
        "load_mode":         load_mode,
        "metadata": {
            "doc_type":  "tabular",
            "file_name": filename,
            "load_mode": load_mode,
        },
    }
    if deleted >= 0:
        result["rows_reconciled"] = deleted
    return result


# ── Text-document ingestion (PDF / DOCX / TXT / MD) ──────────────────────────

def _ingest_text_document(file_path: str, ext: str) -> dict:
    """
    Text document ingestion pipeline — zero LLM calls.

    1. Extract text  — PyMuPDF for PDF, plain reader for others
    2. Chunk         — sliding-window splitter (CHUNK_SIZE / CHUNK_OVERLAP)
    3. Header        — prepend "<Title> [chunk N/M]:" for context
    4. Store         — ChromaDB, one entry per chunk

    The contextual header (step 3) ensures that even when a chunk is retrieved
    in isolation the embedding captures which document it came from — inspired
    by the contextual_chunk_headers.ipynb reference notebook.
    """
    path     = Path(file_path)
    filename = path.name
    doc_title = path.stem.replace("_", " ").replace("-", " ").title()

    logger.info("Text ingest: %s", filename)

    # 1. Extract
    try:
        text = _extract_text_from_file(file_path, ext)
    except Exception as exc:
        return {"status": "error", "message": f"Text extraction failed for {filename}: {exc}"}

    text = text.strip()
    if not text:
        return {"status": "error", "message": f"No text extracted from {filename}"}

    # 2. Chunk
    # PDFs use the table-aware chunker which keeps table pages atomic so that
    # ICD/CPT codes are never split from their descriptions.  All other document
    # types use the standard sliding-window splitter.
    if ext == ".pdf":
        chunks = _split_pdf_table_aware(file_path, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    else:
        chunks = _split_text_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        return {"status": "error", "message": f"Chunking produced no output for {filename}"}

    logger.info("Chunked %s into %d chunk(s)", filename, len(chunks))

    # 3 & 4. Header + batch-store
    base_meta = _meta_from_filename(filename)
    base_meta.update({
        "file_name":    filename,
        "file_path":    str(path.resolve()),
        "file_type":    ext.lstrip("."),
        "total_chunks": str(len(chunks)),
        "source":       "pymupdf",
    })

    # Clear stale SQLite records for this file
    try:
        from db.operations import delete_records_by_source_file
        deleted = delete_records_by_source_file(filename)
        if deleted:
            logger.info("Removed %d existing record(s) for %s", deleted, filename)
    except Exception as exc:
        logger.warning("Failed to clear old records for %s: %s", filename, exc)

    ids, texts, metas = [], [], []
    for chunk_idx, chunk_text in enumerate(chunks):
        # Contextual header gives the embedding model document-level context
        contextual_text = f"{doc_title} [chunk {chunk_idx + 1}/{len(chunks)}]:\n{chunk_text}"

        chunk_id = hashlib.md5(
            f"{path.resolve()}::chunk::{chunk_idx}".encode()
        ).hexdigest()

        ids.append(chunk_id)
        texts.append(contextual_text)
        metas.append({
            **base_meta,
            "chunk_number":     str(chunk_idx),
            "chunk_word_count": str(len(chunk_text.split())),
        })

    ingested = add_documents_batch(ids, texts, metas)

    logger.info(
        "Text ingest complete: %s — %d/%d chunk(s) stored",
        filename, ingested, len(chunks),
    )
    return {
        "status":          "success",
        "doc_id":          _doc_id(file_path),
        "file_name":       filename,
        "chunks_ingested": ingested,
        "metadata":        base_meta,
    }


# ── Main entry points ─────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> dict:
    """
    Load and store a single file in the vector store (and SQLite where relevant).

    Routing:
      .csv / .xlsx / .xls  → tabular pipeline (row-per-doc ChromaDB + SQL)
      .pdf / .txt / .md /
      .docx / .doc /
      .html / .htm / .xml  → text-document pipeline (chunked ChromaDB)
      .json                → direct field-map extraction

    Returns dict with keys: status, and optionally doc_id, file_name, metadata.
    """
    path = Path(file_path)

    if not path.exists():
        return {"status": "error", "message": "File not found: " + file_path}

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"status": "skipped", "message": "Unsupported extension: " + ext}

    filename = path.name

    # Tabular
    if ext in (".csv", ".xlsx", ".xls"):
        return _ingest_tabular(file_path, ext)

    # Text documents
    if ext in (".pdf", ".txt", ".md", ".html", ".htm", ".xml", ".docx", ".doc"):
        return _ingest_text_document(file_path, ext)

    # Structured JSON
    if ext == ".json":
        doc_id = _doc_id(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            text = json.dumps(raw, indent=2)
            meta = _extract_meta_structured(raw, filename)
        except Exception as exc:
            return {"status": "error", "message": f"Failed to load {filename}: {exc}"}

        if not text:
            return {"status": "error", "message": f"No text extracted from {filename}"}

        meta["file_path"] = str(path.resolve())
        meta["file_name"] = filename
        meta["file_type"] = "json"

        success = add_document(doc_id, text, meta)
        if success:
            logger.info("Ingested JSON: %s (id=%s)", filename, doc_id)
            return {"status": "success", "doc_id": doc_id, "file_name": filename, "metadata": meta}
        return {"status": "error", "message": f"Vector store rejected {filename}"}

    return {"status": "skipped", "message": f"No handler for extension: {ext}"}


def ingest_directory(bucket_dir: str = None) -> dict:
    """
    Walk bucket_dir recursively and ingest every supported file.
    Returns a summary dict with keys: success, skipped, errors.
    """
    data_dir = bucket_dir or settings.BUCKET_DIR
    results  = {"success": [], "skipped": [], "errors": []}

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Bucket directory not found: %s", data_dir)
        return results

    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(data_path.rglob("*" + ext))

    # Skip files inside any directory whose name starts with '_'.
    # Convention: rename a folder to _skip_<name> or _ignore_<name> to exclude it
    # without moving it out of the bucket (e.g. large files not ready for ingest).
    def _is_excluded(fp: Path) -> bool:
        relative_parts = fp.relative_to(data_path).parts[:-1]  # all parts except filename
        return any(part.startswith("_") for part in relative_parts)

    excluded = [fp for fp in files if _is_excluded(fp)]
    files    = [fp for fp in files if not _is_excluded(fp)]

    if excluded:
        logger.info(
            "Skipping %d file(s) in underscore-prefixed directories: %s",
            len(excluded),
            sorted({fp.parent.name for fp in excluded}),
        )

    if not files:
        logger.info("No supported files found in %s", data_dir)
        return results

    logger.info("Found %d file(s) to ingest from %s", len(files), data_dir)

    for fp in files:
        result = ingest_file(str(fp))
        status = result["status"]
        if status == "success":
            results["success"].append(result)
        elif status == "skipped":
            results["skipped"].append({"file": str(fp), "reason": result["message"]})
        else:
            results["errors"].append({"file": str(fp), "error": result["message"]})

    logger.info(
        "Ingestion complete — %d ok, %d skipped, %d errors",
        len(results["success"]), len(results["skipped"]), len(results["errors"]),
    )
    return results
