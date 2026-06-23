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

Regex-only — zero LLM calls, zero network latency.
The LLM-based path was removed because it added one Ollama round-trip to every
query and had a known failure mode where ICD codes (e.g. "Z15") were mistaken
for patient IDs, routing reference-document queries to the SQL path and
returning "No matching records found."
"""
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Analytical query detection ────────────────────────────────────────────────
# Patterns that indicate the user wants aggregate/trending data rather than a
# specific patient or provider record.  When any of these match, the intent
# parser sets is_analytical=True and the query is routed to the text-to-SQL
# path instead of the SQL exact-lookup or RAG paths.

_ANALYTICAL_PATTERNS = [
    # Aggregation / analytics action words
    r"\b(count|total|sum|average|avg|mean|rate|trend|trending|revenue|denial|denials"
    r"|ar\b|ytd|breakdown|distribution|percentage|percent|statistics|stat|metric|metrics"
    r"|how many|how much|top \d+|ranking|ranked)\b",
    # "by <dimension>" analytical patterns
    r"\bby\s+(provider|month|payer|year|quarter|diagnosis|icd|cpt|insurance|specialty"
    r"|date|week|day|type|category|service)\b",
    # Temporal / comparative analytics
    r"\b(month-over-month|year-over-year|ytd|quarter-to-date|qtd|last\s+\d+\s+(month|year|week)s?"
    r"|across all|overall|aggregate|among all|total billed|total paid|total charges"
    r"|gross revenue|net revenue|collection rate|denial rate|accounts receivable)\b",
]

_ANALYTICAL_RE = re.compile(
    "|".join(f"(?:{p})" for p in _ANALYTICAL_PATTERNS),
    re.IGNORECASE,
)


def _is_analytical_query(query: str) -> bool:
    """
    Returns True if the query is asking for aggregate / analytical data
    (counts, totals, trends, breakdowns by dimension) rather than a specific
    patient or provider record.

    Important: a query like "total amount for Alice Johnson" IS NOT analytical —
    it's a specific-patient lookup.  Analytical queries typically have no specific
    patient/provider name AND ask for aggregate metrics.
    """
    return bool(_ANALYTICAL_RE.search(query))


# ── Public API ────────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    """
    Parse a user query and return a dict of extracted search criteria.
    Uses regex-based extraction — fast, deterministic, no LLM calls.
    """
    parsed = _normalize_intent(_regex_parse(query))
    logger.info("Intent (regex): %s", parsed)
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

    # The LLM intent parser is told patient_id is "ID or MRN", which is broad
    # enough that a query like "Z15 code definition" or "what is the E11.9
    # diagnosis code" gets ICD-10-CM codes ("Z15", "E11.9") extracted into
    # patient_id/subject_id/hadm_id/provider_npi. That makes has_entity=True
    # for a query that names no patient at all, which routes to sql/hybrid,
    # builds a ChromaDB where-filter on a patient_id that matches nothing,
    # and — because that filter looks "entity-tied" — the no-results path
    # refuses to broaden the search, so the system reports "No matching
    # records found" even though the answer is sitting in an ingested
    # reference document (e.g. the ICD-10-CM guidelines PDF).
    #
    # Strip out any ID-like field that is itself an ICD-10-CM code so these
    # queries fall through to a plain, unfiltered RAG search instead.
    for field in ("patient_id", "subject_id", "hadm_id", "provider_npi"):
        val = intent.get(field)
        if val and isinstance(val, str) and _ICD10_CODE_RE.match(val.strip()):
            logger.info(
                "Dropping %s=%r from intent — looks like an ICD-10-CM code, "
                "not a patient/provider identifier",
                field, val,
            )
            intent[field] = None

    return intent


# ICD-10-CM diagnosis codes: a letter followed by 2 digits, optionally a
# decimal point and 1-4 more digits/letters (e.g. "Z15", "E11.9", "S72.001A").
# Real patient IDs in this system are either purely numeric (MIMIC-III
# SUBJECT_ID/HADM_ID), 10-digit NPIs, or "P###"-style MRNs with 3+ digits
# after the "P" — none of which match this pattern.
_ICD10_CODE_RE = re.compile(r"^[A-Za-z]\d{2}(\.[A-Za-z0-9]{1,4})?$")


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

    # MIMIC-III SUBJECT_ID is stored in the same ChromaDB "patient_id"
    # metadata field as other patient identifiers (tabular ingestion maps
    # SUBJECT_ID columns into patient_id), so it uses the same condition.
    # hadm_id has no ChromaDB metadata equivalent — it's resolved via the
    # SQL facts database only (see sql_retriever.lookup).
    if intent.get("subject_id"):
        conditions.append({"patient_id": {"$eq": intent["subject_id"]}})

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
        "subject_id":    None,
        "hadm_id":       None,
        "specific_field": None,
        # True when the query asks for aggregate / trending data rather than a
        # specific patient/provider record — routes to the text-to-SQL path.
        "is_analytical": _is_analytical_query(query),
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

    # --- MIMIC-III SUBJECT_ID / HADM_ID (explicit keyword phrasing) ---
    # e.g. "subject id 10006", "subject_id 10006", "hadm id 142345",
    # "hadm_id 142345", "admission id 142345", "admission 142345".
    # Checked before the bare-numeric fallback below so an explicit
    # "hadm id ..." phrase isn't mistaken for a generic patient ID.
    hadm_m = re.search(r"\bhadm[\s_]?id\D{0,5}(\d{1,8})\b", q)
    if not hadm_m:
        hadm_m = re.search(r"\badmission(?:\s+id)?\D{0,5}(\d{3,8})\b", q)
    if hadm_m:
        result["hadm_id"] = hadm_m.group(1)

    subj_m = re.search(r"\bsubject[\s_]?id\D{0,5}(\d{1,8})\b", q)
    if subj_m:
        result["subject_id"] = subj_m.group(1)

    # --- Bare numeric patient identifier (MIMIC-III SUBJECT_ID) ---
    # Relax the patient-ID match for IDs given as plain numbers (no "P"
    # prefix), as in "patient 10006" or "records for patient 10006".
    # Skipped if a P-prefixed ID, NPI, subject_id, or hadm_id was already
    # found above to avoid double-matching the same number.
    if (not result["patient_id"] and not result["provider_npi"]
            and not result["subject_id"] and not result["hadm_id"]
            and result["query_type"] == "patient"):
        bare_m = re.search(r"\bpatient\D{0,5}(\d{2,8})\b", q)
        if bare_m:
            result["subject_id"] = bare_m.group(1)

    # --- Specific field ---
    field_map = {
        "NPI number":    ["npi number", "npi"],
        "date of birth": ["dob", "date of birth", "birth date", "birthday"],
        "total amount":  ["total amount", "total", "how much", "amount due", "balance"],
        "claim number":  ["claim no", "claim number", "claim id", "claim"],
        "patient id":    ["patient account number", "patient acct no", "account number",
                          "acct no", "acct number", "patient id", "mrn", "chart number",
                          "member id"],
        "address":       ["address", "location"],
        "phone":         ["phone", "telephone", "contact number"],
        "insurance":     ["insurance id", "insurance number", "insurance"],
        "gender":        ["gender", "sex"],
        "death time":    ["death time", "deathtime", "date of death", "time of death",
                          "died", "expired", "expiry date", "passed away"],
        "admission type": ["admission type"],
        "discharge time": ["discharge time", "dischtime", "discharge date"],
        "admit time":    ["admit time", "admittime", "admission date", "admission time"],
    }
    for field, kws in field_map.items():
        if any(kw in q for kw in kws):
            result["specific_field"] = field
            break

    # --- Patient / provider name ---
    # The LLM path normally extracts this. Without this fallback, any query
    # where the LLM call fails (timeout, rate limit, bad JSON, etc.) loses
    # the named entity entirely: patient_name/provider_name stay None,
    # has_entity becomes False, and the query gets routed to an unfiltered
    # RAG search over the whole corpus — which often surfaces nothing for a
    # precise "<field> for <Name>" question even though the patient exists.
    #
    # Recover a name from common phrasings: "... for <Name>",
    # "patient/provider/doctor/dr <Name>". We scan word-by-word after the
    # keyword and stop at the first word that looks like part of the
    # question rather than the name (a stopword, or a token containing a
    # digit/punctuation that isn't part of a name).
    name = _extract_name_after(query, r"\bfor\b")
    if not name:
        name = _extract_name_after(query, r"\b(?:patient|provider|doctor|dr\.?)\b")
    if not name:
        # Handle "<Name>'s <field>" phrasing (e.g. "Susan L Hill's MRN?"),
        # which has no "for"/"patient"/"provider" keyword to anchor on.
        name = _extract_name_before_possessive(query)

    if name:
        if result["query_type"] == "provider":
            result["provider_name"] = name
        else:
            result["patient_name"] = name

    return result


# Words that indicate the text following a "for"/"patient"/"provider" keyword
# is part of the question, not a person's name — extraction stops here.
_NAME_STOPWORDS = {
    "bill", "bills", "record", "records", "prescription", "prescriptions",
    "invoice", "invoices", "lab", "labs", "result", "results", "report",
    "reports", "claim", "claims", "for", "on", "of", "the", "a", "an", "is",
    "was", "does", "do", "who", "what", "show", "get", "give", "tell", "find",
    "me", "their", "his", "her", "and", "info", "information", "details",
    "id", "number", "no", "npi", "dob", "patient", "patients", "provider",
    "providers", "doctor", "doctors",
}


def _extract_name_after(query: str, keyword_pattern: str) -> str | None:
    """
    Find `keyword_pattern` (e.g. "for") and return the run of name-like
    tokens that follows it (up to 4 tokens), stopping at punctuation, a
    digit-containing token, or a word in _NAME_STOPWORDS. Strips a trailing
    possessive ("Johnson's" -> "Johnson"). Returns None if nothing usable
    follows the keyword.
    """
    m = re.search(keyword_pattern + r"\s+(.+)", query, re.IGNORECASE)
    if not m:
        return None

    tokens = re.findall(r"[A-Za-z][A-Za-z'\-.]*", m.group(1))
    name_tokens = []
    for tok in tokens:
        clean = re.sub(r"['’]s$", "", tok)
        if clean.lower() in _NAME_STOPWORDS:
            break
        name_tokens.append(clean)
        if len(name_tokens) >= 4:
            break

    return " ".join(name_tokens) if name_tokens else None


def _extract_name_before_possessive(query: str) -> str | None:
    """
    Handle "<Name>'s <field>" phrasing (e.g. "Susan L Hill's MRN?"), which has
    no "for"/"patient"/"provider" keyword for _extract_name_after to anchor
    on. Finds the run of tokens immediately before the first "'s"/"'s", then
    drops any leading question-word tokens ("what is", "who is", ...) so only
    the name remains.
    """
    m = re.search(r"([A-Za-z][A-Za-z'\-.]*(?:\s+[A-Za-z][A-Za-z'\-.]*){0,4})['’]s\b", query)
    if not m:
        return None

    tokens = re.findall(r"[A-Za-z][A-Za-z'\-.]*", m.group(1))
    while tokens and tokens[0].lower() in _NAME_STOPWORDS:
        tokens.pop(0)

    tokens = tokens[-4:]
    return " ".join(tokens) if tokens else None


def _reformat(date_str: str, input_fmt: str) -> str:
    """Convert a date string to YYYY-MM-DD."""
    return datetime.strptime(date_str.strip(), input_fmt).strftime("%Y-%m-%d")
