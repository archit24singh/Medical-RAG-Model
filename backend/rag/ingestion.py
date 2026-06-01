"""
Document ingestion pipeline - supports PDF, JSON, CSV, Excel, and plain text.

For each file:
  1. Load and extract/clean content
  2. Extract metadata (patient name, date, doc type, provider NPI, etc.)
     - CSV / Excel: each row becomes its own document with cleaned, normalized metadata
     - JSON: read fields directly from the data (or fall back to LLM)
     - PDF / TXT: ask the LLM to extract fields
  3. Store text + metadata in ChromaDB

CSV/Excel cleaning pipeline (applied before ingestion):
  - Detect the real header row (skips merged-cell titles and blank rows)
  - Drop entirely empty rows and columns
  - Strip leading/trailing whitespace from every cell
  - Forward-fill merged cells so every row has a value
  - Normalize dates to YYYY-MM-DD
  - Title-case patient and provider names
  - Convert blank/"nan"/"N/A" cells to null

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

from config import settings
from rag.vectorstore import add_document, add_documents_batch
from rag.llm_client import call_llm

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    # Tabular (grouped by patient)
    ".csv", ".xlsx", ".xls",
    # Structured
    ".json",
    # Unstructured — routed through Docling → chunker → enricher
    ".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".xml",
    ".txt", ".md",
    # Images
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}

# Extensions handled by the full unstructured parsing pipeline
_UNSTRUCTURED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".xml",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}

# Cell values treated as missing even if non-null
_NULL_VALUES = {"", "nan", "none", "n/a", "na", "-", "--", "null", "undefined"}

_META_PROMPT = """You are analyzing a medical document to extract key metadata.

Filename: {filename}

Document content (first 2000 characters):
{content}

Extract the following. Return ONLY valid JSON - no explanation, no markdown fences:
{{
  "doc_type":      "bill" | "record" | "prescription" | "lab_result" | "provider_info" | "other",
  "patient_name":  "Full Name" or null,
  "patient_id":    "ID string" or null,
  "date":          "YYYY-MM-DD" or null,
  "provider_name": "Full Name" or null,
  "provider_npi":  "10-digit number as string" or null,
  "provider_dob":  "YYYY-MM-DD" or null,
  "total_amount":  "numeric string e.g. 1250.00" or null,
  "summary":       "One sentence describing this document"
}}"""


# ── File loaders ──────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _load_json(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2), data


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ── Data cleaning utilities ───────────────────────────────────────────────────

def _clean_str(val) -> str | None:
    """
    Normalize a cell value to a clean string or None.
    Handles NaN floats, strips whitespace, rejects known null-like strings.
    """
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in _NULL_VALUES:
        return None
    return s


def _normalize_date(val: str) -> str | None:
    """
    Parse a date string into YYYY-MM-DD format.
    Accepts most common formats (MM/DD/YYYY, DD-MM-YYYY, etc.).
    Returns the original value unchanged if parsing fails.
    """
    if not val:
        return None
    try:
        import pandas as pd
        dt = pd.to_datetime(val, infer_datetime_format=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return val


def _safe_cell(val) -> str | None:
    """
    Extract a scalar string from a cell value.
    Handles NaN, None, and the edge case where pandas returns a Series
    instead of a scalar (e.g. duplicate column names).
    """
    import pandas as pd
    if isinstance(val, pd.Series):
        val = val.iloc[0] if not val.empty else None
    return _clean_str(val)


def _find_patient_column(columns: list) -> str | None:
    """
    Return the column name that represents the patient (name or ID).
    Checks against _FIELD_MAP candidates, case-insensitively.
    Prefers patient_name; falls back to patient_id.
    """
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_MAP["patient_name"]:
        if candidate in lower_map:
            return lower_map[candidate]
    for candidate in _FIELD_MAP["patient_id"]:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _find_column(columns: list, field_key: str) -> str | None:
    """Return the first column matching any candidate for a given _FIELD_MAP key."""
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_MAP[field_key]:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _build_patient_text(patient_name: str, rows: list[dict], filename: str,
                        max_chars: int = 4000) -> str:
    """
    Build a readable text block for a patient's grouped rows.
    Starts with a header summary, then appends row data until max_chars is reached.
    """
    lines = [
        f"Patient: {patient_name}",
        f"Source:  {filename}",
        f"Records: {len(rows)} rows",
        "",
    ]
    header = "\n".join(lines)
    body_lines = []

    for i, row in enumerate(rows):
        parts = [f"{k}: {v}" for k, v in row.items() if v]
        body_lines.append(f"Row {i + 1}: " + " | ".join(parts))

    body = "\n".join(body_lines)
    full = header + body

    if len(full) > max_chars:
        # Always keep the header; truncate body with a note
        truncated = body[:max_chars - len(header) - 60]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        full = header + truncated + f"\n... ({len(rows)} rows total, truncated for embedding)"

    return full


def _extract_patient_metadata(patient_name: str, rows: list[dict],
                               filename: str, ext: str,
                               file_path: str, date_col: str | None,
                               amount_col: str | None) -> dict:
    """Build metadata for a patient-grouped document."""
    meta = {k: None for k in list(_FIELD_MAP.keys()) + ["doc_type", "summary"]}

    meta["patient_name"] = patient_name
    meta["doc_type"]     = _doc_type_from_filename(filename)
    meta["file_name"]    = filename
    meta["file_path"]    = str(Path(file_path).resolve())
    meta["file_type"]    = ext.lstrip(".")
    meta["row_count"]    = str(len(rows))

    # Date range: find min and max dates across all rows
    if date_col:
        raw_dates = [_normalize_date(_safe_cell(r.get(date_col))) for r in rows]
        valid_dates = sorted(d for d in raw_dates if d and re.match(r"\d{4}-\d{2}-\d{2}", d))
        if valid_dates:
            meta["date"]     = valid_dates[0]   # earliest (used for filtering)
            meta["date_end"] = valid_dates[-1]  # latest

    # Total amount: sum all numeric values in the amount column
    if amount_col:
        total = 0.0
        for r in rows:
            try:
                val = _safe_cell(r.get(amount_col))
                if val:
                    total += float(re.sub(r"[^\d.]", "", val))
            except Exception:
                pass
        if total > 0:
            meta["total_amount"] = f"{total:.2f}"

    # Patient ID — grab from first row that has it
    id_col = _find_column(list(rows[0].keys()) if rows else [], "patient_id")
    if id_col:
        meta["patient_id"] = _safe_cell(rows[0].get(id_col))

    date_range = (
        f"{meta['date']} to {meta.get('date_end', meta['date'])}"
        if meta.get("date") else "unknown date range"
    )
    meta["summary"] = (
        f"{len(rows)} record(s) for {patient_name} from {filename} ({date_range})"
    )
    return meta


# ── Metadata extraction ───────────────────────────────────────────────────────

_FIELD_MAP = {
    "patient_name":  ["patient_name", "patient", "name", "full_name", "member_name"],
    "patient_id":    ["patient_id", "id", "mrn", "patient_number", "member_id"],
    "date":          ["date", "bill_date", "service_date", "date_of_service", "visit_date"],
    "doc_type":      ["doc_type", "document_type", "type", "record_type"],
    "provider_name": ["provider_name", "provider", "physician", "doctor", "physician_name"],
    "provider_npi":  ["provider_npi", "npi", "npi_number"],
    "provider_dob":  ["provider_dob", "dob", "date_of_birth"],
    "total_amount":  ["total_amount", "total", "amount", "bill_amount", "balance", "amount_due"],
}


def _extract_meta_from_row(row: dict, filename: str, row_idx: int, total_rows: int) -> dict:
    """
    Extract and normalize metadata from a single data row.
    Fields present in _FIELD_MAP are mapped; everything else is stored as-is.
    Rows with missing fields are kept (stored with nulls).
    """
    lower_row = {k.lower(): v for k, v in row.items() if k}
    meta = {k: None for k in list(_FIELD_MAP.keys()) + ["doc_type", "summary"]}

    for meta_key, candidates in _FIELD_MAP.items():
        for candidate in candidates:
            if candidate in lower_row:
                val = _clean_str(lower_row[candidate])
                if val:
                    meta[meta_key] = val
                    break

    # Normalize date
    if meta.get("date"):
        meta["date"] = _normalize_date(meta["date"]) or meta["date"]

    # Normalize names to Title Case
    if meta.get("patient_name"):
        meta["patient_name"] = meta["patient_name"].title()
    if meta.get("provider_name"):
        meta["provider_name"] = meta["provider_name"].title()

    if not meta["doc_type"]:
        meta["doc_type"] = _doc_type_from_filename(filename)

    entity = meta.get("patient_name") or meta.get("provider_name") or "unknown"
    meta["summary"] = (
        f"{meta.get('doc_type', 'Record')} for {entity}"
        f" (row {row_idx + 1}/{total_rows} of {filename})"
    )
    meta["row_index"] = str(row_idx)

    return meta


def _extract_meta_llm(content: str, filename: str) -> dict:
    """Ask the LLM to extract metadata from unstructured document content."""
    try:
        raw = call_llm(_META_PROMPT.format(filename=filename, content=content[:2000]))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning("LLM metadata extraction failed for %s: %s", filename, e)
    return _meta_from_filename(filename)


def _extract_meta_structured(data: object, filename: str) -> dict:
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
        str(meta.get("doc_type", "Document")) + " for " + entity +
        " dated " + str(meta.get("date", "unknown date"))
    )
    return meta


def _meta_from_filename(filename: str) -> dict:
    meta = {k: None for k in list(_FIELD_MAP.keys()) + ["doc_type", "summary"]}
    meta["doc_type"] = _doc_type_from_filename(filename)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if date_match:
        meta["date"] = date_match.group()
    meta["summary"] = "Document: " + filename
    return meta


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


def _row_doc_id(file_path: str, row_idx: int) -> str:
    """Stable ID for a single row — updates in place on re-ingest."""
    key = f"{Path(file_path).resolve()}::row::{row_idx}"
    return hashlib.md5(key.encode()).hexdigest()


def _doc_id(file_path: str) -> str:
    return hashlib.md5(str(Path(file_path).resolve()).encode()).hexdigest()


# ── Tabular ingestion (CSV / Excel) ──────────────────────────────────────────


def _ingest_tabular(file_path: str, ext: str) -> dict:
    """
    Ingest a CSV or Excel file grouped by patient.

    One document per unique patient per file is created, containing all of
    that patient's rows as text. This scales to 20 000+ row files without
    blowing up ChromaDB or embedding time.

    If no patient column is detected, the whole file becomes one document.
    """
    import pandas as pd

    path = Path(file_path)
    filename = path.name

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load {filename}: {e}"}

    if df.empty:
        return {"status": "error", "message": f"No data found in {filename}"}

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    total_rows = len(df)
    if total_rows == 0:
        return {"status": "error", "message": f"No data rows in {filename}"}

    logger.info("Loaded %d rows from %s — grouping by patient", total_rows, filename)

    patient_col = _find_patient_column(list(df.columns))
    date_col    = _find_column(list(df.columns), "date")
    amount_col  = _find_column(list(df.columns), "total_amount")

    # Build clean row dicts (safe scalar extraction)
    clean_rows: list[dict] = []
    for idx in range(total_rows):
        row_dict: dict = {}
        for col in df.columns:
            try:
                row_dict[col] = _safe_cell(df.at[idx, col])
            except Exception:
                row_dict[col] = None
        clean_rows.append(row_dict)

    # Group by patient
    if patient_col:
        groups: dict[str, list] = {}
        for row in clean_rows:
            key = (row.get(patient_col) or "Unknown").strip().title()
            groups.setdefault(key, []).append(row)
        logger.info("Found %d unique patients in %s", len(groups), filename)
    else:
        logger.warning("No patient column in %s — storing as single document", filename)
        groups = {"Unknown": clean_rows}

    # Build and batch-upsert one document per patient
    all_ids:   list = []
    all_texts: list = []
    all_metas: list = []

    for patient_name, rows in groups.items():
        text = _build_patient_text(patient_name, rows, filename)
        if not text.strip():
            continue

        doc_id = hashlib.md5(
            f"{path.resolve()}::patient::{patient_name}".encode()
        ).hexdigest()

        meta = _extract_patient_metadata(
            patient_name, rows, filename, ext, file_path, date_col, amount_col
        )
        all_ids.append(doc_id)
        all_texts.append(text)
        all_metas.append(meta)

    if not all_ids:
        return {"status": "error", "message": f"No valid patient groups in {filename}"}

    ingested = add_documents_batch(all_ids, all_texts, all_metas)

    # Also write to SQLite so tabular data is reachable via exact SQL lookup
    try:
        from rag.structured_extractor import extract_and_store_batch
        tabular_chunks = [
            {"text": t, "original_text": t, "page_number": 1,
             "chunk_index": i, "word_count": len(t.split())}
            for i, t in enumerate(all_texts)
        ]
        sql_count = extract_and_store_batch(tabular_chunks, filename)
        logger.info("SQLite extraction (tabular): %d record(s) stored from %s",
                    sql_count, filename)
    except Exception as exc:
        logger.warning("SQLite extraction failed for tabular %s: %s", filename, exc)

    logger.info(
        "Tabular ingest complete: %s — %d patient docs from %d rows",
        filename, ingested, total_rows,
    )

    return {
        "status":            "success",
        "doc_id":            _doc_id(file_path),
        "file_name":         filename,
        "patients_ingested": ingested,
        "rows_processed":    total_rows,
        "metadata":          {"doc_type": _doc_type_from_filename(filename), "file_name": filename},
    }


# ── Unstructured ingestion (PDF / DOCX / XML / images) ───────────────────────

def _ingest_unstructured(file_path: str, ext: str) -> dict:
    """
    Full processing pipeline for unstructured documents:

      Parse (Docling + RapidOCR)
        → Chunk (header split + LLM semantic split)
          → Enrich (contextual sentences via Mistral)
            → Store (ChromaDB — one entry per chunk)

    Each chunk gets rich metadata including page_number and chunk_number so
    queries can surface exactly the right part of a large document.
    """
    from rag.document_parser import parse_document
    from rag.chunker import chunk_pages
    from rag.enricher import enrich_chunks

    path     = Path(file_path)
    filename = path.name

    logger.info("Unstructured ingest: %s", filename)

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    try:
        describe_images = settings.ENABLE_IMAGE_DESCRIPTION
        pages = parse_document(file_path, describe_images=describe_images)
    except Exception as exc:
        return {"status": "error", "message": f"Parsing failed for {filename}: {exc}"}

    if not pages or not any(p.get("text", "").strip() for p in pages):
        return {"status": "error", "message": f"No text could be extracted from {filename}"}

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_pages(pages, filename)

    if not chunks:
        return {"status": "error", "message": f"Chunking produced no output for {filename}"}

    # ── 3. Contextual enrichment (optional, controlled by config) ─────────────
    if settings.ENABLE_CONTEXTUAL_ENRICHMENT:
        try:
            chunks = enrich_chunks(chunks, pages, filename)
            logger.info("Enrichment complete: %d chunk(s) enriched for %s", len(chunks), filename)
        except Exception as exc:
            logger.warning(
                "Enrichment failed for %s: %s — storing plain chunks", filename, exc
            )

    # ── 3b. Structured extraction → SQLite (per-chunk, every patient) ─────────
    # This is the accuracy layer: every patient/provider/visit/bill mentioned
    # in every chunk is written to SQLite as structured, queryable rows.
    # This runs AFTER enrichment so original_text is always available.
    try:
        from rag.structured_extractor import extract_and_store_batch
        sql_count = extract_and_store_batch(chunks, filename)
        logger.info("SQLite extraction: %d record(s) stored from %s", sql_count, filename)
    except Exception as exc:
        logger.warning("Structured extraction failed for %s: %s — continuing", filename, exc)

    # ── 4. Extract base metadata from the first chunk via LLM ────────────────
    first_text = "\n".join(c["text"] for c in chunks[:2])[:2000]
    base_meta  = _extract_meta_llm(first_text, filename)

    # Normalise names to Title Case (consistent with tabular pipeline)
    if base_meta.get("patient_name"):
        base_meta["patient_name"] = base_meta["patient_name"].strip().title()
    if base_meta.get("provider_name"):
        base_meta["provider_name"] = base_meta["provider_name"].strip().title()

    total_pages  = max(c["page_number"] for c in chunks)
    total_chunks = len(chunks)

    base_meta.update({
        "file_name":    filename,
        "file_path":    str(path.resolve()),
        "file_type":    ext.lstrip("."),
        "total_pages":  str(total_pages),
        "total_chunks": str(total_chunks),
        "source":       "docling",
    })

    # ── 5. Batch store — one ChromaDB entry per chunk ─────────────────────────
    ids, texts, metas = [], [], []
    for chunk in chunks:
        chunk_id = hashlib.md5(
            f"{path.resolve()}::chunk::{chunk['chunk_index']}".encode()
        ).hexdigest()

        chunk_meta = {
            **base_meta,
            "page_number":      str(chunk["page_number"]),
            "chunk_number":     str(chunk["chunk_index"]),
            "chunk_word_count": str(chunk["word_count"]),
            "token_estimate":   str(chunk["token_estimate"]),
            "has_context":      str(chunk.get("has_context", False)),
        }

        ids.append(chunk_id)
        texts.append(chunk["text"])
        metas.append(chunk_meta)

    ingested = add_documents_batch(ids, texts, metas)

    logger.info(
        "Unstructured ingest complete: %s — %d/%d chunk(s) stored from %d page(s)",
        filename, ingested, total_chunks, len(pages),
    )

    return {
        "status":           "success",
        "doc_id":           _doc_id(file_path),
        "file_name":        filename,
        "chunks_ingested":  ingested,
        "pages_processed":  len(pages),
        "metadata":         base_meta,
    }


# ── Main entry points ─────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> dict:
    """
    Load, clean, and store a single file in the vector store.

    CSV / Excel files are ingested row by row (one document per row).
    PDF / TXT / MD files are ingested as a single document via LLM metadata extraction.
    JSON files use direct field mapping with LLM fallback.

    Returns dict with keys: status, and optionally doc_id, file_name, metadata.
    """
    path = Path(file_path)

    if not path.exists():
        return {"status": "error", "message": "File not found: " + file_path}

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"status": "skipped", "message": "Unsupported extension: " + ext}

    filename = path.name

    # ── Tabular: CSV / Excel — grouped by patient ─────────────────────────────
    if ext in (".csv", ".xlsx", ".xls"):
        return _ingest_tabular(file_path, ext)

    # ── Unstructured: PDF / DOCX / XML / images / TXT / MD ───────────────────
    # All these go through the full Docling → chunk → enrich pipeline.
    if ext in _UNSTRUCTURED_EXTENSIONS or ext in (".txt", ".md"):
        return _ingest_unstructured(file_path, ext)

    # ── Structured JSON — direct field mapping ────────────────────────────────
    if ext == ".json":
        doc_id = _doc_id(file_path)
        try:
            text, raw = _load_json(file_path)
            meta = _extract_meta_structured(raw, filename)
            if not meta.get("patient_name") and not meta.get("provider_name"):
                meta = _extract_meta_llm(text, filename)
        except Exception as exc:
            return {"status": "error", "message": f"Failed to load {filename}: {exc}"}

        if not text:
            return {"status": "error", "message": f"No text extracted from {filename}"}

        meta["file_path"] = str(path.resolve())
        meta["file_name"] = filename
        meta["file_type"] = ext.lstrip(".")

        success = add_document(doc_id, text, meta)
        if success:
            logger.info("Ingested JSON: %s (id=%s)", filename, doc_id)
            return {"status": "success", "doc_id": doc_id, "file_name": filename, "metadata": meta}
        return {"status": "error", "message": f"Vector store rejected {filename}"}

    return {"status": "skipped", "message": f"No handler for extension: {ext}"}


def ingest_directory(bucket_dir: str = None) -> dict:
    """
    Walk bucket_dir recursively and ingest every supported file.

    Currently reads from the local bucket/ folder.
    To upgrade to S3: replace Path(bucket_dir).rglob(...) with an S3 listing call.

    Returns a summary dict with keys: success, skipped, errors.
    """
    data_dir = bucket_dir or settings.BUCKET_DIR
    results = {"success": [], "skipped": [], "errors": []}

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Bucket directory not found: %s", data_dir)
        return results

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(data_path.rglob("*" + ext))

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
        "Ingestion complete - %d ok, %d skipped, %d errors",
        len(results["success"]), len(results["skipped"]), len(results["errors"])
    )
    return results
