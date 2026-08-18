"""
Robust text-to-SQL fallback (Planner → SQL Writer → validate → repair).

Used only when no deterministic query shape matches. Follows the layered design:
  1. Scoped schema retrieval (reuse sql_retriever._build_schema_from_catalog).
  2. Planner LLM → a structured JSON plan (intent, tables, filters, metrics,
     group_by, order_by, projections, limit) in JSON mode.
  3. SQL Writer LLM → SQL from the plan + schema snippet.
  4. Static validation with sqlglot (single SELECT, no DML, LIMIT injected) +
     scope check (only tables/columns from the snippet).
  5. Self-repair: on validation/execution error, one retry feeding the error back.

This substantially reduces hallucination vs. naive prompting, but (on a local
8B model) is NOT guaranteed correct — it is the best-effort layer for questions
the deterministic shapes don't cover, and it executes read-only so any returned
values come from the DB, not the model.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

from config import settings
from rag.llm_client import call_llm, call_llm_json
from rag.sql_retriever import (
    _build_schema_from_catalog, _validate_and_limit_sql, _extract_sql_from_response,
)
from db.schema import get_readonly_db, fetchall_dicts

logger = logging.getLogger(__name__)


_PLANNER_SYSTEM = (
    "You are a SQL planning assistant. You convert a question into a STRUCTURED "
    "JSON plan over the given schema. You do NOT write SQL. JSON only."
)

_PLANNER_PROMPT = """\
Schema (only these tables/columns exist — never invent others):
{schema}

Today: {today}
Question: {question}

Produce a JSON plan with these keys (omit what doesn't apply):
{{"intent": "list|aggregate|count|filter",
  "table": "<one table from the schema>",
  "columns": ["<projections>"],
  "filters": ["<col> = '<value>'", "safe_date(<col>,'DD/MM/YYYY') = DATE '2023-05-17'"],
  "metric": {{"func": "SUM|AVG|COUNT|MIN|MAX", "column": "<numeric col>"}},
  "group_by": ["<col>"],
  "order_by": [{{"column": "<col>", "dir": "ASC|DESC"}}],
  "limit": <int>}}

Rules: use ONLY tables/columns from the schema; wrap numeric columns in
safe_numeric(); wrap date columns in safe_date(col,'DD/MM/YYYY'); every
aggregation needs a matching group_by for its non-aggregated projections.
Return ONLY the JSON."""

_WRITER_SYSTEM = (
    "You are a SQL writer. Given a schema and a JSON plan, output ONE valid "
    "PostgreSQL SELECT. No prose, no code fences, SQL only."
)

_WRITER_PROMPT = """\
Schema:
{schema}

Plan:
{plan}

Write ONE PostgreSQL SELECT that implements the plan. Use only schema
tables/columns. Wrap numeric aggregations in safe_numeric(). Include a LIMIT.
Output SQL only."""

_REPAIR_PROMPT = """\
This SQL failed:
{sql}

Error:
{error}

Schema:
{schema}

Return a corrected single PostgreSQL SELECT (SQL only, no prose)."""


def _execute(sql: str) -> list[dict]:
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {int(settings.TEXT_TO_SQL_TIMEOUT_MS)}")
            cur.execute(sql)
            return fetchall_dicts(cur)


def generate_and_run(user_query: str) -> Optional[dict]:
    """Run the planner→writer→validate→repair pipeline. Returns a result dict or None."""
    try:
        with get_readonly_db() as conn:
            schema = _build_schema_from_catalog(conn, user_query)
    except Exception as exc:
        logger.warning("text_to_sql: schema build failed (%s)", exc)
        return None
    if not schema.strip():
        return None

    today = date.today().isoformat()

    # 1. Planner → JSON plan
    plan = call_llm_json(
        _PLANNER_PROMPT.format(schema=schema, today=today, question=user_query),
        system=_PLANNER_SYSTEM, default={},
    )

    # 2. Writer → SQL
    try:
        raw = call_llm(_WRITER_PROMPT.format(schema=schema, plan=plan), system=_WRITER_SYSTEM)
        sql = _extract_sql_from_response(raw)
    except Exception as exc:
        logger.warning("text_to_sql: writer failed (%s)", exc)
        return None

    # 3. Validate + execute, with one repair retry
    rows, final_sql, err = _validate_execute(sql)
    if rows is None:
        try:
            fixed = call_llm(
                _REPAIR_PROMPT.format(sql=sql, error=err, schema=schema),
                system=_WRITER_SYSTEM,
            )
            rows, final_sql, err = _validate_execute(_extract_sql_from_response(fixed))
        except Exception as exc:
            logger.warning("text_to_sql: repair failed (%s)", exc)

    if rows is None:
        return None  # give up → caller keeps the honest 'no answer' path

    return {
        "answer": _format(rows, user_query, final_sql),
        "documents": [],
        "intent": {},
        "filter_used": None,
        "query_path": "text_to_sql",
        "sql_generated": final_sql,
        "rows": rows,
    }


def _validate_execute(sql: str):
    try:
        safe = _validate_and_limit_sql(sql, settings.TEXT_TO_SQL_MAX_ROWS)
    except Exception as exc:
        return None, sql, f"validation: {exc}"
    try:
        return _execute(safe), safe, None
    except Exception as exc:
        return None, safe, f"execution: {exc}"


def _format(rows: list[dict], query: str, sql: str) -> str:
    if not rows:
        return "No matching records found."
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    for r in rows[:50]:
        lines.append(" | ".join(str(r.get(c, "")) for c in cols))
    if len(rows) > 50:
        lines.append(f"… ({len(rows) - 50} more)")
    return "\n".join(lines)
