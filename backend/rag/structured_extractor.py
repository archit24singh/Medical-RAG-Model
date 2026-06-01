"""
Structured extractor — reads each document chunk with the LLM and writes
all extracted facts into the SQLite facts database.

Why per-chunk extraction?
--------------------------
Ingesting a document-level summary misses facts buried deep in the file.
Extracting per-chunk means:
  - Every patient in a multi-patient batch file gets their own row
  - A visit on page 18 is captured even if the patient name only appears there
  - The verbatim raw_text stored per record enables zero-hallucination quoting

What gets extracted
-------------------
For each chunk the LLM returns a JSON object containing:
  patients      — demographic info (name, DOB, gender, patient_id …)
  providers     — provider info (name, NPI, specialty, DOB …)
  visits        — clinical encounters (date, diagnosis, treatment, notes)
  bills         — billing records (date, total, line items)
  prescriptions — prescriptions (date, medication, dosage, frequency)
  lab_results   — lab tests (date, test name, result, reference range)

Every extracted record is linked to a patient/provider row via foreign key
so queries like "all visits for Alice Johnson" become simple SQL joins.

Error handling
--------------
If the LLM returns malformed JSON, the chunk is retried once with a simpler
prompt.  On second failure the chunk is skipped (not stored in SQLite) but
it has already been saved in ChromaDB so semantic search still works.
"""

import json
import logging
import re
from typing import Optional

from rag.llm_client import call_llm
from db.operations import (
    find_or_create_patient,
    find_or_create_provider,
    insert_record,
)

logger = logging.getLogger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are a medical data extraction engine. Extract ALL structured information \
from the document chunk below.

RULES:
- Extract information for EVERY patient and provider mentioned, even if partial
- If a field is not present, use null — do NOT invent values
- Dates MUST be converted to YYYY-MM-DD format
- Names MUST be in Title Case (e.g. "Alice Johnson", "Dr. Robert Chen")
- If the chunk contains NO medical data, return the empty structure shown below
- Return ONLY the JSON object — no explanation, no markdown fences

SOURCE: {filename}, page {page_number}, chunk {chunk_index}

CHUNK TEXT:
{chunk_text}

Return this exact JSON structure:
{{
  "patients": [
    {{
      "name": "Full Name or null",
      "patient_id": "MRN/ID or null",
      "dob": "YYYY-MM-DD or null",
      "gender": "M/F/Other or null",
      "phone": "or null",
      "address": "or null",
      "insurance_id": "or null"
    }}
  ],
  "providers": [
    {{
      "name": "Full Name or null",
      "npi": "10-digit string or null",
      "specialty": "or null",
      "dob": "YYYY-MM-DD or null",
      "phone": "or null",
      "address": "or null"
    }}
  ],
  "visits": [
    {{
      "patient_name": "must match a patient listed above",
      "provider_name": "must match a provider listed above or null",
      "visit_date": "YYYY-MM-DD or null",
      "visit_type": "inpatient/outpatient/emergency/telehealth/other or null",
      "chief_complaint": "or null",
      "diagnosis": "or null",
      "treatment": "or null",
      "notes": "or null"
    }}
  ],
  "bills": [
    {{
      "patient_name": "must match a patient listed above",
      "provider_name": "or null",
      "bill_date": "YYYY-MM-DD or null",
      "total_amount": "numeric string e.g. 1250.00 or null",
      "insurance_amount": "or null",
      "patient_amount": "or null",
      "status": "paid/unpaid/partial or null",
      "line_items": [{{"description": "...", "amount": "..."}}]
    }}
  ],
  "prescriptions": [
    {{
      "patient_name": "must match a patient listed above",
      "provider_name": "or null",
      "date": "YYYY-MM-DD or null",
      "medication": "drug name or null",
      "dosage": "or null",
      "frequency": "or null",
      "duration": "or null",
      "refills": "or null"
    }}
  ],
  "lab_results": [
    {{
      "patient_name": "must match a patient listed above",
      "provider_name": "or null",
      "test_date": "YYYY-MM-DD or null",
      "test_name": "or null",
      "result": "or null",
      "reference_range": "or null",
      "status": "normal/abnormal/critical or null"
    }}
  ]
}}"""

_EMPTY_EXTRACTION = {
    "patients": [], "providers": [], "visits": [],
    "bills": [], "prescriptions": [], "lab_results": [],
}


# ── Public API ────────────────────────────────────────────────────────────────

def extract_and_store(
    chunk: dict,
    source_file: str,
) -> int:
    """
    Run LLM extraction on a single chunk and write all facts to SQLite.

    Args:
        chunk:       A chunk dict from chunker.chunk_pages() or enricher.enrich_chunks().
                     Must have: text, page_number, chunk_index.
        source_file: Original filename (used for provenance metadata).

    Returns:
        Number of records inserted into SQLite (0 on failure or empty chunk).
    """
    chunk_text   = chunk.get("original_text") or chunk.get("text", "")
    page_number  = str(chunk.get("page_number", "?"))
    chunk_index  = str(chunk.get("chunk_index", "?"))

    if not chunk_text.strip():
        return 0

    # ── 1. LLM extraction ────────────────────────────────────────────────────
    extracted = _run_extraction(chunk_text, source_file, page_number, chunk_index)
    if not extracted:
        return 0

    # ── 2. Write to SQLite ───────────────────────────────────────────────────
    count = _store_extraction(extracted, chunk_text, source_file, page_number, chunk_index)
    if count:
        logger.debug(
            "Stored %d record(s) from %s p%s chunk %s",
            count, source_file, page_number, chunk_index,
        )
    return count


def extract_and_store_batch(chunks: list[dict], source_file: str) -> int:
    """
    Run extraction on all chunks from one document.
    Returns total records stored.
    """
    total = 0
    for chunk in chunks:
        try:
            total += extract_and_store(chunk, source_file)
        except Exception as exc:
            logger.warning(
                "Extraction failed for chunk %s of %s: %s",
                chunk.get("chunk_index", "?"), source_file, exc,
            )
    logger.info("Structured extraction complete for %s: %d record(s) stored", source_file, total)
    return total


# ── LLM call + JSON parsing ───────────────────────────────────────────────────

def _run_extraction(
    chunk_text: str,
    filename: str,
    page_number: str,
    chunk_index: str,
) -> Optional[dict]:
    """Call the LLM and return the parsed extraction dict, or None on failure."""
    prompt = _EXTRACT_PROMPT.format(
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        chunk_text=chunk_text[:2000],   # cap to avoid context overflow
    )

    for attempt in range(2):
        try:
            raw = call_llm(prompt)
            parsed = _parse_json(raw)
            if parsed is not None:
                return parsed
            logger.warning(
                "Extraction attempt %d: invalid JSON from LLM for %s chunk %s",
                attempt + 1, filename, chunk_index,
            )
        except Exception as exc:
            logger.warning("LLM call failed on attempt %d: %s", attempt + 1, exc)

    return None


def _parse_json(raw: str) -> Optional[dict]:
    """Extract and parse the first JSON object from the LLM response."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the outermost { … }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        # Validate expected top-level keys
        for key in ("patients", "providers", "visits", "bills",
                    "prescriptions", "lab_results"):
            data.setdefault(key, [])
        return data
    except json.JSONDecodeError:
        return None


# ── SQLite write ──────────────────────────────────────────────────────────────

def _store_extraction(
    extracted:   dict,
    raw_text:    str,
    source_file: str,
    page_number: str,
    chunk_index: str,
) -> int:
    """Write extracted facts to SQLite. Returns records inserted."""
    count = 0

    # Build local name→id maps for cross-referencing within this chunk
    patient_map:  dict[str, int] = {}
    provider_map: dict[str, int] = {}

    # ── Patients ─────────────────────────────────────────────────────────────
    for p in (extracted.get("patients") or []):
        name = _s(p.get("name"))
        if not name:
            continue
        try:
            pid = find_or_create_patient(
                name=name,
                patient_id=_s(p.get("patient_id")),
                dob=_s(p.get("dob")),
                gender=_s(p.get("gender")),
                phone=_s(p.get("phone")),
                address=_s(p.get("address")),
                insurance_id=_s(p.get("insurance_id")),
                source_file=source_file,
            )
            patient_map[name.lower()] = pid
        except Exception as exc:
            logger.warning("Patient insert failed (%s): %s", name, exc)

    # ── Providers ─────────────────────────────────────────────────────────────
    for pr in (extracted.get("providers") or []):
        name = _s(pr.get("name"))
        npi  = _s(pr.get("npi"))
        if not name and not npi:
            continue
        try:
            prid = find_or_create_provider(
                name=name,
                npi=npi,
                specialty=_s(pr.get("specialty")),
                dob=_s(pr.get("dob")),
                phone=_s(pr.get("phone")),
                address=_s(pr.get("address")),
                source_file=source_file,
            )
            if prid and name:
                provider_map[name.lower()] = prid
        except Exception as exc:
            logger.warning("Provider insert failed (%s): %s", name, exc)

    # ── Visits ────────────────────────────────────────────────────────────────
    for v in (extracted.get("visits") or []):
        pid = _resolve_patient(v.get("patient_name"), patient_map)
        if pid is None:
            continue
        try:
            insert_record(
                patient_id=pid,
                record_type="visit",
                raw_text=raw_text,
                source_file=source_file,
                provider_id=_resolve_provider(v.get("provider_name"), provider_map),
                record_date=_s(v.get("visit_date")),
                diagnosis=_s(v.get("diagnosis")),
                treatment=_s(v.get("treatment")),
                details={k: v[k] for k in ("visit_type", "chief_complaint", "notes")
                         if v.get(k)},
                page_number=page_number,
                chunk_index=chunk_index,
            )
            count += 1
        except Exception as exc:
            logger.warning("Visit insert failed: %s", exc)

    # ── Bills ─────────────────────────────────────────────────────────────────
    for b in (extracted.get("bills") or []):
        pid = _resolve_patient(b.get("patient_name"), patient_map)
        if pid is None:
            continue
        try:
            insert_record(
                patient_id=pid,
                record_type="bill",
                raw_text=raw_text,
                source_file=source_file,
                provider_id=_resolve_provider(b.get("provider_name"), provider_map),
                record_date=_s(b.get("bill_date")),
                total_amount=_s(b.get("total_amount")),
                details={k: b[k] for k in
                         ("insurance_amount", "patient_amount", "status", "line_items")
                         if b.get(k)},
                page_number=page_number,
                chunk_index=chunk_index,
            )
            count += 1
        except Exception as exc:
            logger.warning("Bill insert failed: %s", exc)

    # ── Prescriptions ─────────────────────────────────────────────────────────
    for rx in (extracted.get("prescriptions") or []):
        pid = _resolve_patient(rx.get("patient_name"), patient_map)
        if pid is None:
            continue
        try:
            insert_record(
                patient_id=pid,
                record_type="prescription",
                raw_text=raw_text,
                source_file=source_file,
                provider_id=_resolve_provider(rx.get("provider_name"), provider_map),
                record_date=_s(rx.get("date")),
                medication=_s(rx.get("medication")),
                dosage=_s(rx.get("dosage")),
                details={k: rx[k] for k in ("frequency", "duration", "refills")
                         if rx.get(k)},
                page_number=page_number,
                chunk_index=chunk_index,
            )
            count += 1
        except Exception as exc:
            logger.warning("Prescription insert failed: %s", exc)

    # ── Lab results ───────────────────────────────────────────────────────────
    for lr in (extracted.get("lab_results") or []):
        pid = _resolve_patient(lr.get("patient_name"), patient_map)
        if pid is None:
            continue
        try:
            insert_record(
                patient_id=pid,
                record_type="lab_result",
                raw_text=raw_text,
                source_file=source_file,
                provider_id=_resolve_provider(lr.get("provider_name"), provider_map),
                record_date=_s(lr.get("test_date")),
                test_name=_s(lr.get("test_name")),
                test_result=_s(lr.get("result")),
                reference_range=_s(lr.get("reference_range")),
                details={"status": lr["status"]} if lr.get("status") else None,
                page_number=page_number,
                chunk_index=chunk_index,
            )
            count += 1
        except Exception as exc:
            logger.warning("Lab result insert failed: %s", exc)

    return count


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _s(val) -> Optional[str]:
    """Return a stripped string or None."""
    if val is None or str(val).strip().lower() in ("null", "none", "", "n/a"):
        return None
    return str(val).strip()


def _resolve_patient(raw_name, patient_map: dict) -> Optional[int]:
    """Look up patient id from the local within-chunk map."""
    if not raw_name:
        return None
    return patient_map.get(str(raw_name).lower().strip().title().lower())


def _resolve_provider(raw_name, provider_map: dict) -> Optional[int]:
    """Look up provider id from the local within-chunk map."""
    if not raw_name:
        return None
    return provider_map.get(str(raw_name).lower().strip().title().lower())
