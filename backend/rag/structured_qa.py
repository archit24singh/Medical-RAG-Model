"""
Deterministic structured QA — field lookups and numeric aggregation that NEVER
let the LLM extract a value or do arithmetic.

Two paths:
  • field_lookup  — "date of birth / phone / claim status for <patient>" → read
                    the mapped column directly from the entity's rows, verbatim.
  • aggregate     — "total amount paid by the payer [by <dimension>]" → compute
                    SUM/AVG/COUNT in SQL (safe_numeric), optionally GROUP BY.

Both are config-driven via data/concept_map.yaml (field_aliases / measures /
dimensions). A new dataset with different column names needs only synonyms there
— never code. The LLM is not involved, so it cannot answer a different question
than asked (e.g. returning a payment sum when asked for a date of birth).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from db.schema import get_readonly_db, fetchall_dicts, _quote_identifier
from rag.events import load_concept_map, _normalize_col
from rag.profiles import _staging_tables, _table_columns, _match_synonym, entity_profile

logger = logging.getLogger(__name__)


def _phrase_present(ql: str, phrase: str) -> bool:
    """Word-boundary match so 'age' doesn't match 'average' and 'pay' not 'payer'."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", ql) is not None


def _longest_phrase_match(ql: str, alias_block: dict) -> tuple[Optional[str], list[str]]:
    """Return (canonical, columns) for the alias whose phrase best matches the query."""
    best_canon, best_cols, best_len = None, [], 0
    for canon, spec in (alias_block or {}).items():
        for ph in spec.get("phrases", []):
            if _phrase_present(ql, ph) and len(ph) > best_len:
                best_canon, best_cols, best_len = canon, spec.get("columns", []), len(ph)
    return best_canon, best_cols


# ── Field lookup ──────────────────────────────────────────────────────────────

def try_field_lookup(user_query: str, intent: dict) -> Optional[dict]:
    cmap = load_concept_map()
    ql = user_query.lower()

    field, cols = _longest_phrase_match(ql, cmap.get("field_aliases", {}))
    if not field:
        return None

    role = "provider" if intent.get("query_type") == "provider" else "patient"
    if role == "provider":
        name = intent.get("provider_name") or ""
        key = intent.get("provider_npi")
    else:
        name = intent.get("patient_name") or ""
        key = intent.get("patient_id") or intent.get("subject_id")

    toks = name.split()
    if not (len(toks) >= 2 or (len(toks) == 1 and len(toks[0]) >= 3) or key):
        return None

    prof = entity_profile(name=name or None, key=key, role=role)
    if not prof:
        other = "patient" if role == "provider" else "provider"
        prof = entity_profile(name=name or None, key=key, role=other)
    if not prof:
        return None

    # Resolve the actual column present for this entity
    all_cols = set(prof["identity"].keys())
    for li in prof["line_items"]:
        all_cols.update(li.keys())
    norm_to_actual = {_normalize_col(c): c for c in all_cols}
    actual = None
    for c in cols:
        cn = _normalize_col(c)
        if cn in norm_to_actual:
            actual = norm_to_actual[cn]
            break
    if not actual:
        return None  # this dataset has no such column → let other paths try

    label = field.replace("_", " ").title()
    who = prof["entity"]

    if actual in prof["identity"]:
        answer = f"{label} for {who}: {prof['identity'][actual]}"
    else:
        vals = sorted({str(li[actual]).strip() for li in prof["line_items"]
                       if li.get(actual) not in (None, "")})
        if not vals:
            answer = f"{label} for {who}: not recorded."
        elif len(vals) == 1:
            answer = f"{label} for {who}: {vals[0]}"
        else:
            answer = f"{label} for {who}:\n" + "\n".join(f"• {v}" for v in vals)

    return {
        "answer": answer,
        "documents": [],
        "intent": intent,
        "filter_used": {"field": field, "column": actual, "entity": who},
        "query_path": "field_lookup",
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

_AGG_CUES = ("total", "sum", "how much", "average", "avg", "count", "how many", "number of")


def try_aggregate(user_query: str, intent: dict) -> Optional[dict]:
    cmap = load_concept_map()
    ql = user_query.lower()
    if not any(c in ql for c in _AGG_CUES):
        return None

    if "average" in ql or "avg" in ql:
        func = "AVG"
    elif "how many" in ql or "count" in ql or "number of" in ql:
        func = "COUNT"
    else:
        func = "SUM"

    measure, mcols = _longest_phrase_match(ql, cmap.get("measures", {}))
    if func in ("SUM", "AVG") and not measure:
        return None  # need a measure column to sum/average

    # Optional GROUP BY dimension
    dim_canon, dim_syns = None, []
    for canon, syns in (cmap.get("dimensions", {}) or {}).items():
        if f"by {canon}" in ql or f"per {canon}" in ql or f"by each {canon}" in ql:
            dim_canon, dim_syns = canon, syns
            break

    # Optional entity (patient) filter
    pname = intent.get("patient_name") or ""

    results, used_table, measure_col_used, dim_col_used = [], None, None, None
    scalar_total = 0.0
    scalar_count = 0

    with get_readonly_db() as conn:
        for tbl in _staging_tables(conn):
            cols = _table_columns(conn, tbl)
            mcol = _match_synonym(cols, mcols) if mcols else None
            if func in ("SUM", "AVG") and not mcol:
                continue
            dcol = _match_synonym(cols, dim_syns) if dim_syns else None
            ncol = _match_synonym(cols, cmap.get("patient_name_columns", [])) if pname else None

            where = ["_deleted_at IS NULL"]
            params: list = []
            if pname and ncol:
                for t in [t for t in pname.lower().replace(",", " ").split() if len(t) >= 2][:5]:
                    where.append(f"lower({_quote_identifier(ncol)}) LIKE %s")
                    params.append(f"%{t}%")

            agg_expr = (
                "COUNT(*)" if func == "COUNT"
                else f"{func}(safe_numeric({_quote_identifier(mcol)}))"
            )

            if dcol:
                sql = (
                    f"SELECT {_quote_identifier(dcol)} AS dim, {agg_expr} AS val "
                    f"FROM {_quote_identifier(tbl)} WHERE {' AND '.join(where)} "
                    f"GROUP BY {_quote_identifier(dcol)} ORDER BY val DESC NULLS LAST LIMIT 100"
                )
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    for r in fetchall_dicts(cur):
                        results.append((r["dim"], r["val"]))
                used_table, measure_col_used, dim_col_used = tbl, mcol, dcol
                break  # grouped aggregation uses the first matching table
            else:
                sql = f"SELECT {agg_expr} AS val FROM {_quote_identifier(tbl)} WHERE {' AND '.join(where)}"
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    v = row[0] if row else None
                if v is not None:
                    if func == "COUNT":
                        scalar_count += int(v)
                    else:
                        scalar_total += float(v)
                    used_table, measure_col_used = tbl, mcol

    if used_table is None:
        return None

    label = (measure or "records").replace("_", " ")
    if dim_col_used:
        if not results:
            return None
        lines = [f"{func.title()} of {label} by {dim_canon}:"]
        for dim_val, val in results:
            shown = _fmt(val, func)
            lines.append(f"• {dim_val or '(unspecified)'}: {shown}")
        answer = "\n".join(lines)
    elif func == "COUNT":
        answer = f"Count: {scalar_count}"
    else:
        answer = f"{func.title()} of {label}: {_fmt(scalar_total, func)}"
    if pname:
        answer += f"  (for {pname})"

    return {
        "answer": answer,
        "documents": [],
        "intent": intent,
        "filter_used": {
            "aggregate": func, "measure": measure, "measure_column": measure_col_used,
            "group_by": dim_col_used, "patient": pname or None,
        },
        "query_path": "aggregate",
    }


def _fmt(val, func: str) -> str:
    if val is None:
        return "0"
    if func == "COUNT":
        return str(int(val))
    return f"${float(val):,.2f}"


# ── Document (invoice / order) lookup ─────────────────────────────────────────

_DOC_NUM_RE = re.compile(
    r"(?:invoice|inv|order|bill|receipt|document|doc)\s*(?:no\.?|number|num|#)?\s*[:#]?\s*"
    r"([A-Za-z0-9\-]{3,})", re.I,
)


def try_document_lookup(user_query: str, intent: dict) -> Optional[dict]:
    """
    Deterministic answers scoped to one document (invoice/order): list items,
    count items, or total — straight from the staging rows for that document
    number. Complete and exact (no truncation, no LLM arithmetic).
    """
    cmap = load_concept_map()
    ql = user_query.lower()

    m = _DOC_NUM_RE.search(user_query)
    if not m:
        return None
    docnum = m.group(1)
    dkey_syns = cmap.get("document_key_columns", [])
    if not dkey_syns:
        return None

    with get_readonly_db() as conn:
        target = None  # (table, key_col, columns)
        for tbl in _staging_tables(conn):
            cols = _table_columns(conn, tbl)
            kcol = _match_synonym(cols, dkey_syns)
            if not kcol:
                continue
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(tbl)} "
                    f"WHERE _deleted_at IS NULL AND {_quote_identifier(kcol)} = %s",
                    [docnum],
                )
                if cur.fetchone()[0] > 0:
                    target = (tbl, kcol, cols)
                    break
        if target is None:
            return None

        tbl, kcol, cols = target
        base_where = f"_deleted_at IS NULL AND {_quote_identifier(kcol)} = %s"

        # Action: count / total / list
        if "how many" in ql or "count" in ql or "number of" in ql:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(tbl)} WHERE {base_where}", [docnum]
                )
                n = cur.fetchone()[0]
            answer = f"Invoice {docnum} has {n} item(s)."
            return _doc_result(answer, intent, docnum, "count")

        measure, mcols = _longest_phrase_match(ql, cmap.get("measures", {}))
        wants_total = any(c in ql for c in ("total", "sum", "amount", "how much")) or measure == "gross"
        if wants_total:
            mcol = _match_synonym(cols, mcols) if mcols else None
            if not mcol:  # default to a gross/amount-type column
                mcol = _match_synonym(cols, cmap.get("measures", {}).get("gross", {}).get("columns", []))
            if mcol:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT SUM(safe_numeric({_quote_identifier(mcol)})) "
                        f"FROM {_quote_identifier(tbl)} WHERE {base_where}", [docnum]
                    )
                    total = cur.fetchone()[0]
                answer = f"Total ({mcol}) for invoice {docnum}: {_fmt(total, 'SUM')}"
                return _doc_result(answer, intent, docnum, "total")

        # Default: list every line item (complete — no truncation)
        display_cols = [c for c in cols if not c.startswith("_")]
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_quote_identifier(tbl)} WHERE {base_where} LIMIT 500", [docnum]
            )
            rows = fetchall_dicts(cur)

    lines = [f"Invoice {docnum} — {len(rows)} item(s):"]
    for r in rows:
        parts = [f"{c}: {r[c]}" for c in display_cols if r.get(c) not in (None, "")]
        lines.append("• " + " | ".join(parts))
    return _doc_result("\n".join(lines), intent, docnum, "list")


def _doc_result(answer: str, intent: dict, docnum: str, action: str) -> dict:
    return {
        "answer": answer,
        "documents": [],
        "intent": intent,
        "filter_used": {"document": docnum, "action": action},
        "query_path": "document_lookup",
    }
