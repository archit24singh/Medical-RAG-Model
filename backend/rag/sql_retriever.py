"""
SQL retriever — converts a parsed intent into SQLite queries and returns
verbatim results with zero LLM involvement in the answer.

Anti-hallucination guarantee
-----------------------------
Every result row includes `raw_text` — the verbatim chunk text from the
source document.  The answer formatter quotes this directly.  The LLM is
used ONLY to format the presentation, not to generate any facts.

Result format
-------------
Returns a list of result dicts compatible with the ChromaDB result format
so the rest of the pipeline (frontend, answer generator) needs no changes:

  {
    "id":              str,   — "sql:<record_id>"
    "content":         str,   — formatted fact string (human-readable)
    "metadata":        dict,  — patient/provider/record metadata
    "relevance_score": float, — always 1.0 for exact SQL matches
    "source":          "sql", — tag so answer generator knows to quote verbatim
    "raw_text":        str,   — verbatim source text for quoting
  }
"""

import json
import logging
from typing import Optional

from db.operations import (
    query_patient_records,
    query_provider,
    query_all_patient_records,
    get_patient_info,
    query_records_by_hadm_id,
    get_admission_info,
)

logger = logging.getLogger(__name__)

# Map intent doc_type → SQLite record_type
_DOC_TYPE_MAP = {
    "bill":          "bill",
    "record":        "visit",
    "prescription":  "prescription",
    "lab_result":    "lab_result",
    "provider_info": None,   # provider lookup, not a record type
    "other":         None,
}

# Fields that live on the `patients` table (demographics), not on individual
# clinical/billing records. If one of these is asked for, we need to consult
# get_patient_info() even when the patient also has clinical records — those
# records don't carry the patient's address/phone/etc.
_DEMOGRAPHIC_FIELD_KEYWORDS = (
    "address", "phone", "telephone", "insurance", "gender",
    "location", "mrn", "patient id", "account number", "acct",
    "chart number", "member id",
)


def _is_demographic_field(specific_field: Optional[str]) -> bool:
    if not specific_field:
        return False
    field_lower = specific_field.lower()
    return any(kw in field_lower for kw in _DEMOGRAPHIC_FIELD_KEYWORDS)


def lookup(intent: dict) -> list[dict]:
    """
    Execute the appropriate SQL query for the given intent.

    Returns a list of result dicts (empty list = nothing found in SQLite).
    """
    patient_name  = intent.get("patient_name")
    patient_id    = intent.get("patient_id")
    provider_name = intent.get("provider_name")
    provider_npi  = intent.get("provider_npi")
    subject_id    = intent.get("subject_id")
    hadm_id       = intent.get("hadm_id")
    date          = intent.get("date")
    doc_type      = intent.get("doc_type")
    specific_field = intent.get("specific_field")

    record_type = _DOC_TYPE_MAP.get(doc_type)   # may be None

    # ── Provider query ────────────────────────────────────────────────────────
    if provider_name or provider_npi:
        rows = query_provider(
            provider_name=provider_name,
            provider_npi=provider_npi,
        )
        return [_provider_to_result(r, specific_field) for r in rows]

    # ── Admission (HADM_ID) query ────────────────────────────────────────────
    # MIMIC-III: a single admission (HADM_ID) groups together an admissions
    # row (admittime/dischtime/admission_type) and zero or more
    # diagnosis/billing records (DIAGNOSES_ICD.csv rows) that share the same
    # HADM_ID. Surface both, admission info first.
    if hadm_id and not (patient_name or patient_id or subject_id):
        results = []
        admission = get_admission_info(hadm_id)
        if admission:
            results.append(_admission_to_result(admission, specific_field))

        rows = query_records_by_hadm_id(hadm_id, limit=30)
        results.extend(_record_to_result(r, specific_field) for r in rows)
        return results

    # ── Patient query (name, patient_id, and/or subject_id) ──────────────────
    if patient_name or patient_id or subject_id:
        # If a specific date is given, filter records by that date
        rows = query_patient_records(
            patient_name=patient_name,
            patient_id_str=patient_id,
            record_type=record_type,
            record_date=date,
            limit=30,
            subject_id=subject_id,
        )

        # If an HADM_ID was also given, narrow results to that admission only.
        if hadm_id:
            rows = [r for r in rows if r.get("hadm_id") == hadm_id]

        results = [_record_to_result(r, specific_field) for r in rows] if rows else []

        # If the user asked for a demographic field (address, phone, insurance,
        # gender, MRN, ...), clinical/billing records won't have it — those
        # columns only exist on the `patients` table. Fetch patient
        # demographics regardless of whether clinical records were found, and
        # surface that field. If no records exist at all, fall back to the
        # demographics-only result.
        if not results or _is_demographic_field(specific_field):
            patient = get_patient_info(patient_name, patient_id, subject_id=subject_id)
            if patient:
                demo_result = _patient_demographics_result(patient, specific_field)
                if _is_demographic_field(specific_field) and results:
                    # Put the demographic answer first so it's clearly surfaced,
                    # while keeping the clinical/billing records for context.
                    results = [demo_result] + results
                elif not results:
                    results = [demo_result]

        return results

    return []


# ── Result formatters ─────────────────────────────────────────────────────────

def _record_to_result(row: dict, specific_field: Optional[str]) -> dict:
    """Convert a SQLite records row to a standard result dict."""
    record_type = row.get("record_type", "record")
    patient_name = row.get("patient_name", "Unknown")
    record_date  = row.get("record_date", "unknown date")

    # Build a human-readable content string highlighting the requested field
    content_lines = [
        f"Patient: {patient_name}",
        f"Record type: {record_type}",
        f"Date: {record_date}",
    ]

    if row.get("provider_name"):
        content_lines.append(f"Provider: {row['provider_name']}")
    if row.get("provider_npi"):
        content_lines.append(f"NPI: {row['provider_npi']}")

    # Type-specific fields
    if record_type == "visit":
        if row.get("diagnosis"):
            content_lines.append(f"Diagnosis: {row['diagnosis']}")
        if row.get("treatment"):
            content_lines.append(f"Treatment: {row['treatment']}")

    elif record_type == "bill":
        if row.get("total_amount"):
            content_lines.append(f"Total amount: ${row['total_amount']}")
        if row.get("claim_number"):
            content_lines.append(f"Claim No: {row['claim_number']}")

    elif record_type == "prescription":
        if row.get("medication"):
            content_lines.append(f"Medication: {row['medication']}")
        if row.get("dosage"):
            content_lines.append(f"Dosage: {row['dosage']}")

    elif record_type == "lab_result":
        if row.get("test_name"):
            content_lines.append(f"Test: {row['test_name']}")
        if row.get("test_result"):
            content_lines.append(f"Result: {row['test_result']}")
        if row.get("reference_range"):
            content_lines.append(f"Reference range: {row['reference_range']}")

    # MIMIC-III admission / diagnosis fields (DIAGNOSES_ICD.csv rows)
    if row.get("hadm_id"):
        content_lines.append(f"Admission (HADM_ID): {row['hadm_id']}")
    if row.get("icd9_code"):
        icd9_line = f"ICD9 code: {row['icd9_code']}"
        if row.get("icd9_description"):
            icd9_line += f" ({row['icd9_description']})"
        content_lines.append(icd9_line)
    if row.get("seq_num"):
        content_lines.append(f"Diagnosis sequence: {row['seq_num']}")

    # If a specific field was asked for, highlight it prominently. Merge in
    # details_json first — that's where columns without a dedicated SQL field
    # (e.g. "Balance", "Total Payment", "ICD2 Code", "Modifier 1") are stored,
    # keyed by their original spreadsheet header.
    if specific_field:
        value = _extract_specific_field(_merge_details(row), specific_field)
        if value:
            content_lines.insert(0, f"[{specific_field.upper()}]: {value}")

    content_lines.append(f"\nSource: {row.get('source_file', 'unknown')}"
                         f" (page {row.get('page_number', '?')})")

    return {
        "id":              f"sql:{row['id']}",
        "content":         "\n".join(content_lines),
        "raw_text":        row.get("raw_text", ""),
        "relevance_score": 1.0,
        "source":          "sql",
        "metadata": {
            "patient_name":  patient_name,
            "patient_id":    row.get("patient_mrn"),
            "record_type":   record_type,
            "date":          record_date,
            "provider_name": row.get("provider_name"),
            "provider_npi":  row.get("provider_npi"),
            "total_amount":  row.get("total_amount"),
            "claim_number":  row.get("claim_number"),
            "diagnosis":     row.get("diagnosis"),
            "medication":    row.get("medication"),
            "test_name":     row.get("test_name"),
            "test_result":   row.get("test_result"),
            "file_name":     row.get("source_file"),
            "page_number":   row.get("page_number"),
            "doc_type":      record_type,
            "hadm_id":          row.get("hadm_id"),
            "icd9_code":        row.get("icd9_code"),
            "icd9_description": row.get("icd9_description"),
            "seq_num":          row.get("seq_num"),
        },
    }


def _admission_to_result(row: dict, specific_field: Optional[str]) -> dict:
    """Convert an `admissions` table row to a standard result dict."""
    content_lines = [
        f"Admission (HADM_ID): {row.get('hadm_id', 'Unknown')}",
    ]
    if row.get("subject_id"):
        content_lines.append(f"Subject ID: {row['subject_id']}")
    if row.get("admittime"):
        content_lines.append(f"Admit time: {row['admittime']}")
    if row.get("dischtime"):
        content_lines.append(f"Discharge time: {row['dischtime']}")
    if row.get("deathtime"):
        content_lines.append(f"Death time: {row['deathtime']}")
    if row.get("admission_type"):
        content_lines.append(f"Admission type: {row['admission_type']}")
    if row.get("diagnosis"):
        content_lines.append(f"Diagnosis: {row['diagnosis']}")

    if specific_field:
        value = _extract_specific_field(row, specific_field)
        if value:
            content_lines.insert(0, f"[{specific_field.upper()}]: {value}")

    return {
        "id":              f"sql:admission:{row.get('id')}",
        "content":         "\n".join(content_lines),
        "raw_text":        "",
        "relevance_score": 1.0,
        "source":          "sql",
        "metadata": {
            "subject_id":     row.get("subject_id"),
            "hadm_id":        row.get("hadm_id"),
            "admittime":      row.get("admittime"),
            "dischtime":      row.get("dischtime"),
            "deathtime":      row.get("deathtime"),
            "admission_type": row.get("admission_type"),
            "diagnosis":      row.get("diagnosis"),
            "doc_type":       "admission",
            "file_name":      row.get("source_file"),
        },
    }


def _provider_to_result(row: dict, specific_field: Optional[str]) -> dict:
    """Convert a providers row to a standard result dict."""
    content_lines = [
        f"Provider: {row.get('name', 'Unknown')}",
    ]
    if row.get("npi"):
        content_lines.append(f"NPI: {row['npi']}")
    if row.get("specialty"):
        content_lines.append(f"Specialty: {row['specialty']}")
    if row.get("dob"):
        content_lines.append(f"Date of birth: {row['dob']}")
    if row.get("phone"):
        content_lines.append(f"Phone: {row['phone']}")
    if row.get("address"):
        content_lines.append(f"Address: {row['address']}")

    if specific_field:
        value = _extract_specific_field(row, specific_field)
        if value:
            content_lines.insert(0, f"[{specific_field.upper()}]: {value}")

    return {
        "id":              f"sql:provider:{row['id']}",
        "content":         "\n".join(content_lines),
        "raw_text":        "",   # provider rows don't have source text
        "relevance_score": 1.0,
        "source":          "sql",
        "metadata": {
            "provider_name": row.get("name"),
            "provider_npi":  row.get("npi"),
            "provider_dob":  row.get("dob"),
            "specialty":     row.get("specialty"),
            "doc_type":      "provider_info",
            "file_name":     row.get("source_files", "").split(",")[0],
        },
    }


def _patient_demographics_result(row: dict, specific_field: Optional[str]) -> dict:
    """Return a patient demographics result when no records are found."""
    content_lines = [f"Patient: {row.get('name', 'Unknown')}"]
    for field in ("patient_id", "dob", "gender", "phone", "address", "insurance_id"):
        if row.get(field):
            content_lines.append(f"{field.replace('_', ' ').title()}: {row[field]}")

    if specific_field:
        value = _extract_specific_field(row, specific_field)
        if value:
            content_lines.insert(0, f"[{specific_field.upper()}]: {value}")

    return {
        "id":              f"sql:patient:{row['id']}",
        "content":         "\n".join(content_lines),
        "raw_text":        "",
        "relevance_score": 1.0,
        "source":          "sql",
        "metadata": {
            "patient_name": row.get("name"),
            "patient_id":   row.get("patient_id"),
            "doc_type":     "demographics",
            "file_name":    row.get("source_files", "").split(",")[0],
        },
    }


def _merge_details(row: dict) -> dict:
    """
    Return a copy of `row` with details_json's key/value pairs merged in.

    Spreadsheet columns that don't have a dedicated SQL column (e.g.
    "Balance", "Total Payment", "ICD2 Code", "Modifier 1" — see the
    catch-all in rag.ingestion._write_rows_to_sql) are stored as a JSON
    blob in details_json, keyed by their original header text. Merging
    them in lets _extract_specific_field's generic substring fallback find
    these fields too, e.g. a query mentioning "balance" matches the
    "Balance" key from details_json.
    """
    details_raw = row.get("details_json")
    if not details_raw:
        return row

    try:
        details = json.loads(details_raw)
    except (TypeError, ValueError):
        return row

    merged = dict(row)
    for k, v in details.items():
        # Don't let a generic details_json key clobber a populated,
        # dedicated SQL column.
        if v and not merged.get(k):
            merged[k] = v
    return merged


def _extract_specific_field(row: dict, field: str) -> Optional[str]:
    """Try to find the requested specific field in a result row."""
    field_lower = field.lower()

    # Direct column aliases
    aliases = {
        "npi":           ["npi", "provider_npi"],
        "npi number":    ["npi", "provider_npi"],
        "date of birth": ["dob", "patient_dob", "provider_dob"],
        "dob":           ["dob", "patient_dob", "provider_dob"],
        "total amount":  ["total_amount"],
        "amount":        ["total_amount"],
        "claim no":      ["claim_number"],
        "claim number":  ["claim_number"],
        "claim id":      ["claim_number"],
        "claim":         ["claim_number"],
        "diagnosis":     ["diagnosis"],
        "medication":    ["medication"],
        "dosage":        ["dosage"],
        "test result":   ["test_result"],
        "result":        ["test_result"],
        "visit date":    ["record_date"],
        "date":          ["record_date"],
        "address":       ["address"],
        "phone":         ["phone"],
        "telephone":     ["phone"],
        "insurance":     ["insurance_id"],
        "gender":        ["gender"],
        "patient id":    ["patient_id", "patient_mrn"],
        "mrn":           ["patient_id", "patient_mrn"],
        "subject id":    ["subject_id"],
        "subject_id":    ["subject_id"],
        "hadm id":       ["hadm_id"],
        "hadm_id":       ["hadm_id"],
        "admission id":  ["hadm_id"],
        "icd9 code":     ["icd9_code"],
        "icd9_code":     ["icd9_code"],
        "icd9 description": ["icd9_description"],
        "admission type": ["admission_type"],
        "admit time":    ["admittime"],
        "discharge time": ["dischtime"],
        "death time":    ["deathtime"],
        "deathtime":     ["deathtime"],
        "date of death": ["deathtime"],
        "time of death": ["deathtime"],
    }

    for alias_key, columns in aliases.items():
        if alias_key in field_lower:
            for col in columns:
                if row.get(col):
                    return str(row[col])

    # Generic fallback — search all string columns
    for col, val in row.items():
        if val and field_lower in col.lower():
            return str(val)

    return None
