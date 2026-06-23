"""
SQL retriever — converts a parsed intent into PostgreSQL queries and returns
verbatim results with zero LLM involvement in the answer.

Two distinct query paths
------------------------

lookup(intent)
  Exact / template-based lookup for patient/provider/admission queries.
  Anti-hallucination guarantee: every result row includes `raw_text` — the
  verbatim chunk text from the source document.  The LLM is used ONLY to
  format the presentation, not to generate any facts.

analytical_lookup(user_query)
  Text-to-SQL path for aggregate/analytical queries (counts, totals, trends,
  breakdowns).  The LLM generates a SQL SELECT from a YAML schema file that
  includes table descriptions and few-shot examples.  The generated SQL is:
    1. Parsed by sqlglot — must be a single SELECT, no DML/DDL.
    2. LIMIT-injected if missing.
    3. Executed under SET LOCAL statement_timeout to prevent runaway queries.
    4. Run on POSTGRES_READONLY_DSN (SELECT-only role when configured).

Result format
-------------
Returns a list of result dicts compatible with the ChromaDB result format
so the rest of the pipeline (frontend, answer generator) needs no changes:

  {
    "id":              str,   — "sql:<record_id>" or "analytical:<idx>"
    "content":         str,   — formatted fact string (human-readable)
    "metadata":        dict,  — row metadata
    "relevance_score": float, — 1.0 for all SQL / analytical results
    "source":          str,   — "sql" or "analytical"
    "raw_text":        str,   — verbatim source text (empty for analytical)
    "sql_generated":   str,   — SQL used (analytical path only, first result)
  }
"""

import json
import logging
import re
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


# ── Analytical (text-to-SQL) path ─────────────────────────────────────────────

_TEXT_TO_SQL_PROMPT = """\
You are a SQL expert generating PostgreSQL SELECT queries for a medical billing/RCM database.

DATABASE SCHEMA:
{schema_text}

EXAMPLE QUERIES:
{examples_text}

STRICT RULES — you MUST follow every rule or the query will be rejected:
1. Generate ONLY a single SELECT statement.
2. NEVER use INSERT, UPDATE, DELETE, CREATE, DROP, TRUNCATE, or any DML/DDL.
3. NEVER use CTEs (WITH clauses) that contain DML.
4. Cast text columns to NUMERIC for arithmetic: total_amount::NUMERIC
5. Guard text-to-date casts with a regex check: WHERE col ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
6. Use NULLIF to avoid division-by-zero: NULLIF(COUNT(*), 0)
7. Do NOT add a LIMIT clause — it will be injected automatically.
8. Output ONLY the raw SQL statement — no markdown, no explanation, no code fences.

USER QUESTION: {question}

SQL:"""


def _load_schema_yaml(path: str) -> dict:
    """Load and parse the schema metadata YAML file (few-shot examples only)."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Schema-linking from live catalog (replaces static _build_schema_text) ─────

# Canonical tables always shown to the LLM (clinical + billing entities)
_ALWAYS_INCLUDE: frozenset[str] = frozenset({
    "patients", "providers", "records", "admissions",
})

# Maximum number of query-relevant exposed views to include
_SCHEMA_LINK_TOP_N: int = 5


def _score_table(
    table_name:  str,
    col_names:   list[str],
    query_tokens: set[str],
) -> float:
    """
    Token-overlap relevance score for schema-linking.

    Scores on both the view-name tokens (v_<label> stem) and the canonical
    column-name tokens so the ranking matches user vocabulary.
    Returns 0.0 if the vocab set is empty.
    Swappable for embedding-based scoring without changing the caller.
    """
    # View-name stem: "v_campbell_billing" → {"campbell", "billing"}
    stem = set(re.sub(r"^v_", "", table_name).split("_"))
    # Column tokens: "service_date" → {"service", "date"}
    col_tokens = {
        tok
        for col in col_names
        for tok in col.split("_")
        if len(tok) > 2
    }
    vocab = stem | col_tokens
    if not vocab:
        return 0.0
    return len(query_tokens & vocab) / len(vocab)


def _build_schema_from_catalog(conn, user_query: str) -> str:
    """
    Build a schema text block for the LLM prompt by querying live PostgreSQL
    metadata rather than a static YAML file.

    Canonical tables (patients / providers / records / admissions) — always
    included.  Raw stg_* staging tables — NEVER included.

    Query-exposed streams appear as v_<label> views — ranked by token overlap
    with the user query, top-N included.  The LLM sees canonical column names
    (from the VIEW definition) and correct types (from safe_cast helpers baked
    into the VIEW) — no translation step needed.

    Metadata columns (_source_file, _row_hash, …) are never shown.
    """
    query_tokens = set(
        re.sub(r"[^a-z0-9]", " ", user_query.lower()).split()
    )

    # Canonical tables — read from information_schema
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM   information_schema.columns
                WHERE  table_schema = 'public'
                  AND  table_name   = ANY(%s)
                  AND  column_name NOT LIKE '\\_%%'
                ORDER BY table_name, ordinal_position
                """,
                [list(_ALWAYS_INCLUDE)],
            )
            canon_rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Schema-linking: canonical table query failed: %s", exc)
        canon_rows = []

    # Query-exposed views (v_*) — read from information_schema.views + columns
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.table_name, c.column_name
                FROM   information_schema.columns  c
                JOIN   information_schema.views    v
                    ON v.table_schema = c.table_schema
                   AND v.table_name   = c.table_name
                WHERE  c.table_schema = 'public'
                  AND  c.table_name   LIKE 'v\\_%%'
                ORDER BY c.table_name, c.ordinal_position
                """
            )
            view_rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Schema-linking: view column query failed: %s", exc)
        view_rows = []

    # Group by table/view name
    def _group(rows):
        out: dict[str, list[str]] = {}
        for tname, cname in rows:
            out.setdefault(tname, []).append(cname)
        return out

    canonical = _group(canon_rows)
    views     = _group(view_rows)

    # Rank views by token overlap; always include top-N even at zero overlap
    ranked_views = sorted(
        views.items(),
        key=lambda kv: -_score_table(kv[0], kv[1], query_tokens),
    )[:_SCHEMA_LINK_TOP_N]

    lines: list[str] = []
    for table_name, cols in list(canonical.items()) + ranked_views:
        lines.append(f"TABLE: {table_name}")
        for col in cols:
            lines.append(f"  {col}")
        lines.append("")
    return "\n".join(lines)


def _build_examples_text(schema_data: dict) -> str:
    """Convert few-shot examples from the YAML to a text block for the LLM prompt."""
    lines = []
    for i, ex in enumerate(schema_data.get("few_shot_examples", []), 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Q: {ex['question']}")
        lines.append(f"  SQL: {ex['sql'].strip()}")
        lines.append("")
    return "\n".join(lines)


def _extract_sql_from_response(text: str) -> str:
    """Strip markdown code fences and return the raw SQL."""
    text = text.strip()
    # Remove ```sql ... ``` or ``` ... ``` blocks
    m = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _validate_and_limit_sql(sql_text: str, max_rows: int) -> str:
    """
    Parse SQL with sqlglot, validate it's a single SELECT (no DML/DDL),
    and inject a LIMIT clause if missing.

    Raises ValueError if the SQL fails validation.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp
    except ImportError as exc:
        raise ImportError("sqlglot is required for analytical queries") from exc

    try:
        statements = sqlglot.parse(sql_text, dialect="postgres")
    except Exception as exc:
        raise ValueError(f"SQL parse error: {exc}") from exc

    if not statements:
        raise ValueError("Empty SQL — LLM returned no statement")

    if len(statements) != 1:
        raise ValueError(
            f"Expected exactly 1 SQL statement, got {len(statements)}. "
            "Multi-statement SQL is not allowed."
        )

    stmt = statements[0]

    # Must be a plain SELECT — reject everything else
    if not isinstance(stmt, exp.Select):
        raise ValueError(
            f"Only SELECT statements are allowed, got: {type(stmt).__name__}. "
            "INSERT/UPDATE/DELETE/CREATE/DROP are prohibited."
        )

    # Belt-and-suspenders: walk the AST for any DML/DDL node
    forbidden_types = (
        exp.Insert, exp.Update, exp.Delete,
        exp.Create, exp.Drop, exp.Command,
    )
    for node in stmt.walk():
        if isinstance(node, forbidden_types):
            raise ValueError(
                f"Prohibited SQL node found in statement: {type(node).__name__}"
            )

    # Inject LIMIT if the LLM omitted it (rule 7 above)
    if stmt.args.get("limit") is None:
        stmt = stmt.limit(max_rows)

    return stmt.sql(dialect="postgres")


def analytical_lookup(user_query: str) -> list[dict]:
    """
    Text-to-SQL path for analytical / aggregate queries.

    Pipeline
    --------
    1. Load schema + few-shot examples from TEXT_TO_SQL_SCHEMA_FILE (YAML).
    2. Build LLM prompt with schema context and examples.
    3. Call local LLM (Ollama) to generate SQL — no external API calls.
    4. sqlglot AST validation: must be single SELECT, no DML/DDL.
    5. LIMIT injection: add TEXT_TO_SQL_MAX_ROWS if missing.
    6. SET LOCAL statement_timeout for the connection.
    7. Execute on POSTGRES_READONLY_DSN (read-only role when configured).
    8. Return rows as standard result dicts.

    Returns an empty list (not an exception) on any error so the calling
    code can fall back gracefully.
    """
    from db.schema import get_readonly_db, fetchall_dicts
    from rag.llm_client import call_llm
    from config import settings

    # 1. Build schema text from live catalog (VIEWs + canonical tables)
    #    and load few-shot examples from the YAML file.
    try:
        with get_readonly_db() as conn:
            schema_text = _build_schema_from_catalog(conn, user_query)
    except Exception as exc:
        logger.warning("Schema-linking failed: %s — falling back to empty schema", exc)
        schema_text = ""

    try:
        schema_data   = _load_schema_yaml(settings.TEXT_TO_SQL_SCHEMA_FILE)
        examples_text = _build_examples_text(schema_data)
    except FileNotFoundError:
        logger.warning(
            "Schema file not found: %s — few-shot examples disabled. "
            "Create the file or update TEXT_TO_SQL_SCHEMA_FILE in .env",
            settings.TEXT_TO_SQL_SCHEMA_FILE,
        )
        examples_text = "(no examples)"
    except Exception as exc:
        logger.warning("Cannot load schema YAML (%s): %s", settings.TEXT_TO_SQL_SCHEMA_FILE, exc)
        examples_text = "(no examples)"

    # 2 & 3. Build prompt and call LLM
    prompt = _TEXT_TO_SQL_PROMPT.format(
        schema_text=schema_text,
        examples_text=examples_text,
        question=user_query,
    )

    try:
        raw_sql_response = call_llm(prompt)
    except Exception as exc:
        logger.warning("LLM SQL generation failed: %s", exc)
        return []

    raw_sql = _extract_sql_from_response(raw_sql_response)
    logger.info("LLM generated SQL (raw): %s", raw_sql[:300])

    # 4 & 5. Validate + inject LIMIT
    try:
        validated_sql = _validate_and_limit_sql(raw_sql, settings.TEXT_TO_SQL_MAX_ROWS)
    except (ValueError, ImportError) as exc:
        logger.warning("SQL validation failed: %s | raw SQL: %s", exc, raw_sql[:200])
        return []

    logger.info("Validated SQL: %s", validated_sql[:300])

    # 6 & 7. Execute under statement_timeout on read-only connection
    try:
        with get_readonly_db() as conn:
            with conn.cursor() as cur:
                # SET LOCAL applies only within this transaction
                cur.execute(
                    f"SET LOCAL statement_timeout = '{settings.TEXT_TO_SQL_TIMEOUT_MS}ms'"
                )
                cur.execute(validated_sql)
                rows = fetchall_dicts(cur)
    except Exception as exc:
        logger.warning("Analytical SQL execution failed: %s | SQL: %s", exc, validated_sql[:200])
        return []

    logger.info("Analytical query returned %d row(s)", len(rows))

    if not rows:
        return []

    # 8. Format results
    results = []
    for idx, row in enumerate(rows):
        content_parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
        content = " | ".join(content_parts)

        result: dict = {
            "id":              f"analytical:{idx}",
            "content":         content,
            "raw_text":        "",
            "relevance_score": 1.0,
            "source":          "analytical",
            "metadata":        {k: str(v) if v is not None else None for k, v in row.items()},
        }
        # Attach the generated SQL to the first result so the orchestrator can
        # log it in the audit record.
        if idx == 0:
            result["sql_generated"] = validated_sql

        results.append(result)

    return results


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
