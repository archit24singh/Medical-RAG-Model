"""
Intent parser — converts a natural language query into structured search criteria.

Example:
  Input:  "Get patient Alice Johnson's bill for 27-10-2025"
  Output: {
              "query_type":    "patient",
              "patient_name":  "Alice Johnson",
              "date":          "2025-10-27",
              "doc_type":      "bill",
              ...
          }

The LLM is tried first (most accurate).
If it is unavailable a regex fallback handles common patterns.
"""
import json
import logging
import re
from datetime import datetime

from rag.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── LLM prompt ────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """You are a medical records search assistant. Extract search criteria from the user's query.

Return ONLY a valid JSON object with these exact fields (use null if a field is not mentioned):
{{
  "query_type":     "patient" or "provider",
  "patient_name":   full patient name as string or null,
  "patient_id":     patient ID or MRN or null,
  "date":           date in YYYY-MM-DD format or null,
  "doc_type":       "bill" | "record" | "prescription" | "lab_result" | "provider_info" or null,
  "provider_name":  full provider / doctor name or null,
  "provider_npi":   10-digit NPI number as string or null,
  "specific_field": the specific data point requested (e.g. "NPI number", "date of birth", "total amount") or null
}}

Date conversion rules:
  "27-10-2025"        → "2025-10-27"
  "October 27, 2025"  → "2025-10-27"
  "10/27/2025"        → "2025-10-27"

Examples:
  Query: "Get patient Alice Johnson's bill for 27-10-2025"
  JSON:  {{"query_type":"patient","patient_name":"Alice Johnson","patient_id":null,"date":"2025-10-27","doc_type":"bill","provider_name":null,"provider_npi":null,"specific_field":null}}

  Query: "What is the NPI number for Dr. Robert Chen?"
  JSON:  {{"query_type":"provider","patient_name":null,"patient_id":null,"date":null,"doc_type":"provider_info","provider_name":"Dr. Robert Chen","provider_npi":null,"specific_field":"NPI number"}}

  Query: "Show me records for patient P001"
  JSON:  {{"query_type":"patient","patient_name":null,"patient_id":"P001","date":null,"doc_type":"record","provider_name":null,"provider_npi":null,"specific_field":null}}

  Query: "What is the date of birth for provider NPI 9876543210?"
  JSON:  {{"query_type":"provider","patient_name":null,"patient_id":null,"date":null,"doc_type":"provider_info","provider_name":null,"provider_npi":"9876543210","specific_field":"date of birth"}}

User query: "{query}"
JSON:"""


# ── Public API ────────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    """
    Parse a user query and return a dict of extracted search criteria.
    Falls back to regex if the LLM is unavailable.
    """
    # 1. Try LLM — most accurate, handles any phrasing
    try:
        raw = call_llm(_INTENT_PROMPT.format(query=query))
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            parsed = _normalize_intent(parsed)
            logger.info(f"Intent (LLM): {parsed}")
            return parsed
    except Exception as e:
        logger.warning(f"LLM intent parsing failed: {e} — using regex fallback")

    # 2. Regex fallback
    parsed = _normalize_intent(_regex_parse(query))
    logger.info(f"Intent (regex): {parsed}")
    return parsed


def _normalize_intent(intent: dict) -> dict:
    """
    Normalize extracted names to Title Case so they match the stored metadata.
    Ingestion stores patient_name in title case; queries must match exactly.
    """
    for field in ("patient_name", "provider_name"):
        val = intent.get(field)
        if val and isinstance(val, str):
            intent[field] = val.strip().title()
    return intent


def build_where_filter(intent: dict) -> dict | None:
    """
    Convert parsed intent into a ChromaDB `where` filter dict.

    Rules:
    - Only include fields that were actually extracted (non-null, non-empty).
    - For person names, require at least first + last name before exact-matching
      (a single token like "Alice" is better handled by semantic search alone).
    - Returns None if no filter criteria are available.
    """
    conditions = []

    if intent.get("doc_type"):
        conditions.append({"doc_type": {"$eq": intent["doc_type"]}})

    if intent.get("patient_id"):
        conditions.append({"patient_id": {"$eq": intent["patient_id"]}})

    # Only exact-match a name if it looks like a full name (≥ 2 tokens)
    patient_name = intent.get("patient_name") or ""
    if len(patient_name.split()) >= 2:
        conditions.append({"patient_name": {"$eq": patient_name}})

    if intent.get("date"):
        conditions.append({"date": {"$eq": intent["date"]}})

    if intent.get("provider_npi"):
        conditions.append({"provider_npi": {"$eq": intent["provider_npi"]}})

    provider_name = intent.get("provider_name") or ""
    if len(provider_name.split()) >= 2:
        conditions.append({"provider_name": {"$eq": provider_name}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Regex fallback ────────────────────────────────────────────────────────────

def _regex_parse(query: str) -> dict:
    q = query.lower()
    result = {
        "query_type":    None,
        "patient_name":  None,
        "patient_id":    None,
        "date":          None,
        "doc_type":      None,
        "provider_name": None,
        "provider_npi":  None,
        "specific_field": None,
    }

    # --- query_type ---
    if any(w in q for w in ["provider", "doctor", "dr.", "physician", "npi", "specialist"]):
        result["query_type"] = "provider"
    else:
        result["query_type"] = "patient"

    # --- doc_type ---
    doc_map = {
        "bill":         ["bill", "invoice", "charge", "payment"],
        "record":       ["record", "medical record", "chart", "history"],
        "prescription": ["prescription", "rx", "medication"],
        "lab_result":   ["lab", "test result", "blood work"],
        "provider_info":["provider info", "npi", "doctor info"],
    }
    for dtype, kws in doc_map.items():
        if any(kw in q for kw in kws):
            result["doc_type"] = dtype
            break
    if not result["doc_type"] and result["query_type"] == "provider":
        result["doc_type"] = "provider_info"

    # --- date ---
    date_fmts = [
        (r"\b(\d{4})-(\d{2})-(\d{2})\b",  lambda m: m.group()),
        (r"\b(\d{2})-(\d{2})-(\d{4})\b",  lambda m: _reformat(m.group(), "%d-%m-%Y")),
        (r"\b(\d{2})/(\d{2})/(\d{4})\b",  lambda m: _reformat(m.group(), "%m/%d/%Y")),
        (r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
         r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b",
         lambda m: _reformat(m.group(), "%d %B %Y")),
    ]
    for pat, fmt_fn in date_fmts:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            try:
                result["date"] = fmt_fn(m)
                break
            except Exception:
                pass

    # --- NPI (10-digit number) ---
    npi_m = re.search(r"\b(\d{10})\b", query)
    if npi_m:
        result["provider_npi"] = npi_m.group()
        result["query_type"] = "provider"

    # --- Patient ID (e.g. P001) ---
    pid_m = re.search(r"\b[Pp](\d{3,6})\b", query)
    if pid_m:
        result["patient_id"] = f"P{pid_m.group(1)}"

    # --- Specific field ---
    field_map = {
        "NPI number":    ["npi number", "npi"],
        "date of birth": ["dob", "date of birth", "birth date", "birthday"],
        "total amount":  ["total amount", "total", "how much", "amount due", "balance"],
        "address":       ["address", "location"],
        "phone":         ["phone", "telephone", "contact number"],
    }
    for field, kws in field_map.items():
        if any(kw in q for kw in kws):
            result["specific_field"] = field
            break

    return result


def _reformat(date_str: str, input_fmt: str) -> str:
    """Convert a date string to YYYY-MM-DD."""
    return datetime.strptime(date_str.strip(), input_fmt).strftime("%Y-%m-%d")
