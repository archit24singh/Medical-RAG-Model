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
  ChromaDB  — one doc per row, text = "Col1: Val1 | Col2: Val2 | …"
  SQLite    — direct columnar mapping via _write_rows_to_sql (no LLM)

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


# ── Field maps ─────────────────────────────────────────────────────────────────

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

# Additional column candidates used for per-row SQLite ingestion
_ROW_FIELD_MAP = {
    "patient_dob":    ["patient dob", "date_of_birth", "dob"],
    "patient_gender": ["patient gender", "gender", "sex"],
    "claim_number":   ["claim no", "claim number", "claim_no", "claim id"],
    "address_line1":  ["patient address line 1", "address line 1", "address", "patient address"],
    "address_line2":  ["patient address line 2", "address line 2"],
    "city":           ["patient city", "city"],
    "state":          ["patient state", "state"],
    "zip":            ["patient zip code", "patient zip", "zip", "zip code", "postal code"],
    "phone":          ["patient cell phone", "patient home phone", "patient phone",
                       "phone", "telephone"],
    "insurance_id":   ["primary payer subscriber no", "insurance id", "policy number"],
    "diagnosis_code": ["icd1 code", "icd code", "diagnosis code"],
    "diagnosis_name": ["icd1 name", "diagnosis name", "diagnosis"],
    "cpt_code":       ["cpt code", "procedure code"],
    "cpt_name":       ["cpt description", "procedure description"],
    # MIMIC-III admission / diagnosis fields
    "hadm_id":        ["hadm_id"],
    "icd9_code":      ["icd9_code"],
    "seq_num":        ["seq_num"],
    "admittime":      ["admittime"],
    "dischtime":      ["dischtime"],
    "deathtime":      ["deathtime"],
    "admission_type": ["admission_type"],
}


# ── Data cleaning utilities ───────────────────────────────────────────────────

def _clean_str(val) -> str | None:
    """Normalize a cell value to a clean string or None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in _NULL_VALUES:
        return None
    return s


def _normalize_date(val: str) -> str | None:
    """Parse a date string into YYYY-MM-DD; return original if parsing fails."""
    if not val:
        return None
    try:
        import pandas as pd
        dt = pd.to_datetime(val, infer_datetime_format=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return val


def _safe_cell(val) -> str | None:
    """Extract a scalar string from a cell, handling NaN and Series edge cases."""
    import pandas as pd
    if isinstance(val, pd.Series):
        val = val.iloc[0] if not val.empty else None
    return _clean_str(val)


# ── Fuzzy column-header matching ──────────────────────────────────────────────

_FUZZY_HEADER_THRESHOLD = 0.5


def _header_tokens(header: str) -> set[str]:
    """Tokenize a column header into normalized lowercase tokens."""
    if not header:
        return set()
    s = str(header)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    s = re.sub(r"[_\-/.,#:()]+", " ", s)
    return {t for t in s.lower().split() if t}


def _header_match_score(header_tokens: set[str], candidate: str) -> float:
    """Jaccard similarity between a header's tokens and a candidate's tokens."""
    cand = _header_tokens(candidate)
    if not header_tokens or not cand:
        return 0.0
    union = header_tokens | cand
    return len(header_tokens & cand) / len(union) if union else 0.0


def _find_best_fuzzy_column(
    columns: list, candidates: list[str],
    threshold: float = _FUZZY_HEADER_THRESHOLD,
) -> str | None:
    """Return the column whose tokens best overlap with any candidate, above threshold."""
    best_col, best_score = None, 0.0
    for col in columns:
        tokens = _header_tokens(col)
        if not tokens:
            continue
        for candidate in candidates:
            score = _header_match_score(tokens, candidate)
            if score > best_score:
                best_score, best_col = score, col
    return best_col if best_score >= threshold else None


def _find_patient_column(columns: list) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_MAP["patient_name"]:
        if candidate in lower_map:
            return lower_map[candidate]
    for candidate in _FIELD_MAP["patient_id"]:
        if candidate in lower_map:
            return lower_map[candidate]
    fuzzy = _find_best_fuzzy_column(columns, _FIELD_MAP["patient_name"])
    if fuzzy:
        return fuzzy
    return _find_best_fuzzy_column(columns, _FIELD_MAP["patient_id"])


def _find_column(columns: list, field_key: str) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_MAP[field_key]:
        if candidate in lower_map:
            return lower_map[candidate]
    return _find_best_fuzzy_column(columns, _FIELD_MAP[field_key])


def _find_row_column(columns: list, field_key: str) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in _ROW_FIELD_MAP[field_key]:
        if candidate in lower_map:
            return lower_map[candidate]
    return _find_best_fuzzy_column(columns, _ROW_FIELD_MAP[field_key])


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


# ── Address helper ────────────────────────────────────────────────────────────

def _compose_address(row: dict, cols: dict) -> str | None:
    """Combine address-related columns into a single human-readable string."""
    line1 = _safe_cell(row.get(cols.get("address_line1"))) if cols.get("address_line1") else None
    line2 = _safe_cell(row.get(cols.get("address_line2"))) if cols.get("address_line2") else None
    city  = _safe_cell(row.get(cols.get("city")))  if cols.get("city")  else None
    state = _safe_cell(row.get(cols.get("state"))) if cols.get("state") else None
    zipc  = _safe_cell(row.get(cols.get("zip")))   if cols.get("zip")   else None

    parts = [p for p in (line1, line2) if p]
    city_state = ", ".join(p for p in (city, state) if p)
    if zipc:
        city_state = f"{city_state} {zipc}".strip()
    if city_state:
        parts.append(city_state)
    return ", ".join(parts) if parts else None


def _build_patient_text(
    patient_name: str, rows: list[dict], filename: str, max_chars: int = 4000,
) -> str:
    """Build a readable text blob for a patient's rows (used by _write_rows_to_sql)."""
    header = (
        f"Patient: {patient_name}\n"
        f"Source:  {filename}\n"
        f"Records: {len(rows)} rows\n\n"
    )
    body_lines = [
        f"Row {i + 1}: " + " | ".join(f"{k}: {v}" for k, v in row.items() if v)
        for i, row in enumerate(rows)
    ]
    body = "\n".join(body_lines)
    full = header + body
    if len(full) > max_chars:
        truncated = body[:max_chars - len(header) - 60]
        last_nl = truncated.rfind("\n")
        truncated = truncated[:last_nl] if last_nl > 0 else truncated
        full = header + truncated + f"\n... ({len(rows)} rows total, truncated)"
    return full


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


# ── Direct per-row SQLite ingestion ──────────────────────────────────────────

def _write_rows_to_sql(
    clean_rows:   list[dict],
    patient_col:  str | None,
    date_col:     str | None,
    amount_col:   str | None,
    columns:      list,
    filename:     str,
) -> int:
    """
    Write each spreadsheet row directly to SQLite as a claim/visit record.
    Deterministic, no LLM, scales to tens of thousands of rows.
    Returns the number of records inserted.
    """
    from db.operations import (
        find_or_create_patient, find_or_create_provider, find_or_create_admission,
        insert_record,
    )
    from rag.icd9_lookup import lookup_icd9

    if not patient_col:
        return 0

    row_cols = {key: _find_row_column(columns, key) for key in _ROW_FIELD_MAP}
    id_col            = _find_column(columns, "patient_id")
    provider_name_col = _find_column(columns, "provider_name")
    provider_npi_col  = _find_column(columns, "provider_npi")

    _lower_col_map = {c.lower().strip(): c for c in columns if c}
    subject_id_col = _lower_col_map.get("subject_id")

    _consumed_cols = {
        patient_col, id_col, date_col, amount_col,
        provider_name_col, provider_npi_col, subject_id_col,
    }
    _consumed_cols.update(c for c in row_cols.values() if c)
    extra_cols = [c for c in columns if c and c not in _consumed_cols]

    patient_pk_cache:  dict[str, int] = {}
    provider_pk_cache: dict[str, int] = {}
    count = 0

    for idx, row in enumerate(clean_rows):
        patient_name = (row.get(patient_col) or "").strip().title()
        if not patient_name:
            continue

        subject_id_val = _safe_cell(row.get(subject_id_col)) if subject_id_col else None

        if patient_name not in patient_pk_cache:
            try:
                patient_pk_cache[patient_name] = find_or_create_patient(
                    name=patient_name,
                    patient_id=_safe_cell(row.get(id_col)) if id_col else None,
                    dob=_normalize_date(_safe_cell(row.get(row_cols["patient_dob"])))
                        if row_cols["patient_dob"] else None,
                    gender=_safe_cell(row.get(row_cols["patient_gender"]))
                        if row_cols["patient_gender"] else None,
                    phone=_safe_cell(row.get(row_cols["phone"]))
                        if row_cols["phone"] else None,
                    address=_compose_address(row, row_cols),
                    insurance_id=_safe_cell(row.get(row_cols["insurance_id"]))
                        if row_cols["insurance_id"] else None,
                    subject_id=subject_id_val,
                    source_file=filename,
                )
            except Exception as exc:
                logger.warning("find_or_create_patient failed for %s: %s", patient_name, exc)
                continue
        patient_pk = patient_pk_cache[patient_name]

        provider_pk = None
        provider_name = _safe_cell(row.get(provider_name_col)) if provider_name_col else None
        if provider_name:
            provider_key = provider_name.title()
            if provider_key not in provider_pk_cache:
                try:
                    provider_pk_cache[provider_key] = find_or_create_provider(
                        name=provider_key,
                        npi=_safe_cell(row.get(provider_npi_col)) if provider_npi_col else None,
                        source_file=filename,
                    )
                except Exception as exc:
                    logger.warning("find_or_create_provider failed for %s: %s", provider_key, exc)
                    provider_pk_cache[provider_key] = None
            provider_pk = provider_pk_cache[provider_key]

        hadm_id_val       = _safe_cell(row.get(row_cols["hadm_id"]))  if row_cols["hadm_id"]  else None
        icd9_code_val     = _safe_cell(row.get(row_cols["icd9_code"])) if row_cols["icd9_code"] else None
        seq_num_val       = _safe_cell(row.get(row_cols["seq_num"]))  if row_cols["seq_num"]  else None
        icd9_description  = lookup_icd9(icd9_code_val) if icd9_code_val else None

        admittime_val      = _safe_cell(row.get(row_cols["admittime"]))      if row_cols["admittime"]      else None
        dischtime_val      = _safe_cell(row.get(row_cols["dischtime"]))      if row_cols["dischtime"]      else None
        deathtime_val      = _safe_cell(row.get(row_cols["deathtime"]))      if row_cols["deathtime"]      else None
        admission_type_val = _safe_cell(row.get(row_cols["admission_type"])) if row_cols["admission_type"] else None

        diag_code = _safe_cell(row.get(row_cols["diagnosis_code"])) if row_cols["diagnosis_code"] else None
        diag_name = _safe_cell(row.get(row_cols["diagnosis_name"])) if row_cols["diagnosis_name"] else None
        diagnosis = " - ".join(
            p for p in (diag_code or icd9_code_val, diag_name or icd9_description) if p
        ) or None

        cpt_code  = _safe_cell(row.get(row_cols["cpt_code"])) if row_cols["cpt_code"] else None
        cpt_name  = _safe_cell(row.get(row_cols["cpt_name"])) if row_cols["cpt_name"] else None
        treatment = " - ".join(p for p in (cpt_code, cpt_name) if p) or None

        claim_number = _safe_cell(row.get(row_cols["claim_number"])) if row_cols["claim_number"] else None
        total_amount = _safe_cell(row.get(amount_col)) if amount_col else None
        record_date  = _normalize_date(_safe_cell(row.get(date_col))) if date_col else None
        record_type  = "bill" if amount_col else "visit"

        raw_text = _build_patient_text(patient_name, [row], filename, max_chars=4000)

        details: dict = {}
        if row_cols["address_line2"] and row.get(row_cols["address_line2"]):
            details["address_line_2"] = _safe_cell(row.get(row_cols["address_line2"]))
        for col in extra_cols:
            val = _safe_cell(row.get(col))
            if val:
                details[col] = val

        try:
            insert_record(
                patient_id=patient_pk,
                record_type=record_type,
                raw_text=raw_text,
                source_file=filename,
                provider_id=provider_pk,
                record_date=record_date,
                total_amount=total_amount,
                claim_number=claim_number,
                diagnosis=diagnosis,
                treatment=treatment,
                details=details or None,
                page_number=None,
                chunk_index=str(idx),
                hadm_id=hadm_id_val,
                icd9_code=icd9_code_val,
                icd9_description=icd9_description,
                seq_num=seq_num_val,
            )
            count += 1
        except Exception as exc:
            logger.warning("insert_record failed for row %d (%s): %s", idx, patient_name, exc)

        if hadm_id_val:
            try:
                find_or_create_admission(
                    patient_pk=patient_pk,
                    subject_id=subject_id_val,
                    hadm_id=hadm_id_val,
                    admittime=admittime_val,
                    dischtime=dischtime_val,
                    deathtime=deathtime_val,
                    admission_type=admission_type_val,
                    diagnosis=diag_name or icd9_description,
                    source_file=filename,
                )
            except Exception as exc:
                logger.warning(
                    "find_or_create_admission failed for hadm_id %s: %s", hadm_id_val, exc
                )

    return count


# ── Tabular ingestion (CSV / Excel) ──────────────────────────────────────────

def _ingest_tabular(file_path: str, ext: str) -> dict:
    """
    Ingest a CSV or Excel file.

    ChromaDB: one document per row — text = "Col1: Val1 | Col2: Val2 | …"
              This matches the reference-repo CSVLoader pattern and makes every
              individual row retrievable via semantic search.

    SQLite:   direct columnar mapping via _write_rows_to_sql — no LLM,
              captures every field for exact ID-based queries.
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

    logger.info("Loaded %d rows from %s", total_rows, filename)

    patient_col = _find_patient_column(list(df.columns))
    date_col    = _find_column(list(df.columns), "date")
    amount_col  = _find_column(list(df.columns), "total_amount")

    # Build clean row dicts
    clean_rows: list[dict] = []
    for idx in range(total_rows):
        row_dict: dict = {}
        for col in df.columns:
            try:
                row_dict[col] = _safe_cell(df.at[idx, col])
            except Exception:
                row_dict[col] = None
        clean_rows.append(row_dict)

    # ── ChromaDB: one document per row ───────────────────────────────────────
    all_ids:   list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []

    for idx, row in enumerate(clean_rows):
        text = _row_to_chroma_text(row, filename)
        if not text.strip():
            continue

        meta = _extract_meta_from_row(row, filename, idx, total_rows)
        meta["file_name"] = filename
        meta["file_type"] = ext.lstrip(".")

        all_ids.append(_row_doc_id(file_path, idx))
        all_texts.append(text)
        all_metas.append(meta)

    if not all_ids:
        return {"status": "error", "message": f"No non-empty rows in {filename}"}

    ingested = add_documents_batch(all_ids, all_texts, all_metas)
    logger.info("ChromaDB: %d/%d row doc(s) stored from %s", ingested, total_rows, filename)

    # ── SQLite: clear old records, then write all rows ────────────────────────
    try:
        from db.operations import delete_records_by_source_file
        deleted = delete_records_by_source_file(filename)
        if deleted:
            logger.info("Removed %d existing record(s) for %s before re-ingest", deleted, filename)
    except Exception as exc:
        logger.warning("Failed to clear old records for %s: %s", filename, exc)

    try:
        sql_count = _write_rows_to_sql(
            clean_rows, patient_col, date_col, amount_col, list(df.columns), filename
        )
        logger.info("SQLite: %d record(s) stored from %s", sql_count, filename)
    except Exception as exc:
        logger.warning("SQLite ingestion failed for %s: %s", filename, exc)

    return {
        "status":         "success",
        "doc_id":         _doc_id(file_path),
        "file_name":      filename,
        "rows_ingested":  ingested,
        "rows_processed": total_rows,
        "metadata":       {"doc_type": _doc_type_from_filename(filename), "file_name": filename},
    }


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
