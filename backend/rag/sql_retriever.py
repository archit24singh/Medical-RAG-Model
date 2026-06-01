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

import logging
from typing import Optional

from db.operations import (
    query_patient_records,
    query_provider,
    query_all_patient_records,
    get_patient_info,
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


def lookup(intent: dict) -> list[dict]:
    """
    Execute the appropriate SQL query for the given intent.

    Returns a list of result dicts (empty list = nothing found in SQLite).
    """
    patient_name  = intent.get("patient_name")
    patient_id    = intent.get("patient_id")
    provider_name = intent.get("provider_name")
    provider_npi  = intent.get("provider_npi")
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

    # ── Patient query ─────────────────────────────────────────────────────────
    if patient_name or patient_id:
        # If a specific date is given, filter records by that date
        rows = query_patient_records(
            patient_name=patient_name,
            patient_id_str=patient_id,
            record_type=record_type,
            record_date=date,
            limit=30,
        )

        if rows:
            results = [_record_to_result(r, specific_field) for r in rows]
            return results

        # Fallback: return patient demographics if no records found
        patient = get_patient_info(patient_name, patient_id)
        if patient:
            return [_patient_demographics_result(patient, specific_field)]

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

    # If a specific field was asked for, highlight it prominently
    if specific_field:
        value = _extract_specific_field(row, specific_field)
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
            "diagnosis":     row.get("diagnosis"),
            "medication":    row.get("medication"),
            "test_name":     row.get("test_name"),
            "test_result":   row.get("test_result"),
            "file_name":     row.get("source_file"),
            "page_number":   row.get("page_number"),
            "doc_type":      record_type,
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
        "diagnosis":     ["diagnosis"],
        "medication":    ["medication"],
        "dosage":        ["dosage"],
        "test result":   ["test_result"],
        "result":        ["test_result"],
        "visit date":    ["record_date"],
        "date":          ["record_date"],
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
