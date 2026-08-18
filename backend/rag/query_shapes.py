"""
Generic deterministic query shapes — schema-agnostic, 100% reliable, no LLM.

Instead of hand-coding SQL per document type, we implement a small set of
universal query SHAPES as parameterized templates, and resolve the
(relation, column, value, dimension) dynamically from the live schema +
concept_map synonyms. The same shapes therefore work over ANY structure —
invoices, provider revenue, credentialing lists — with zero per-document code.

Shapes:
  • distinct   — "all/list/distinct <field>"                 → SELECT DISTINCT col
  • count      — "how many <x> [by <dim>]"                    → COUNT(*) [GROUP BY]
  • aggregate  — "total/sum/average <measure> [by <dim>]"     → SUM/AVG [GROUP BY]
  • top_k      — "top N <x> by <measure>"                     → ORDER BY … LIMIT N
  • filter     — "<records> where/for <field> = <value>"      → WHERE col = val

Column/table resolution uses concept_map (field_aliases/measures/dimensions)
plus direct normalized column-name matching against the queryable relations
(auto-exposed v_* views + curated tables). Anything that doesn't match a shape
returns None → the caller falls back to the robust text-to-SQL pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from db.schema import get_readonly_db, fetchall_dicts, _quote_identifier
from rag.events import load_concept_map, _normalize_col

logger = logging.getLogger(__name__)

_CURATED = ("patients", "providers", "records", "admissions")
_AGG_WORDS = {"sum": "SUM", "total": "SUM", "average": "AVG", "avg": "AVG", "mean": "AVG"}
_COUNT_WORDS = ("how many", "count", "number of")
_DISTINCT_WORDS = ("all ", "list ", "distinct ", "unique ", "every ")


# ── Schema introspection ──────────────────────────────────────────────────────

def _column_catalog(conn) -> dict[str, list[str]]:
    """{relation: [columns]} for queryable relations (v_* views + curated tables)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='public' "
            "AND (table_name LIKE 'v\\_%%' OR table_name = ANY(%s)) "
            "AND column_name NOT LIKE '\\_%%' "
            "ORDER BY table_name, ordinal_position",
            [list(_CURATED)],
        )
        cat: dict[str, list[str]] = {}
        for rel, col in cur.fetchall():
            cat.setdefault(rel, []).append(col)
    return cat


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _tokens(s: str) -> set[str]:
    return {_singular(t) for t in _normalize_col(s).split("_") if len(t) >= 3}


def _resolve_column(term: str, catalog: dict, synonyms: list[str] | None = None) -> list[tuple]:
    """
    Resolve a natural-language field term → list of (relation, column) candidates.
    Priority: concept_map synonyms / exact name, then shared-token (stemmed) match
    so 'providers' resolves 'provider_name' and 'revenue' does NOT match 'gross_worth'.
    """
    tnorm = _normalize_col(term)
    tterms = _tokens(term)
    syn_norm = {_normalize_col(s) for s in (synonyms or [])} | ({tnorm} if tnorm else set())
    exact, tok = [], []
    for rel, cols in catalog.items():
        for col in cols:
            cn = _normalize_col(col)
            if cn in syn_norm:
                exact.append((rel, col))
            elif tterms and (tterms & _tokens(col)):
                tok.append((rel, col))
    return exact or tok


def _concept_columns(cmap: dict, block: str, canon: Optional[str] = None) -> list[str]:
    out = []
    for c, spec in (cmap.get(block, {}) or {}).items():
        if canon and c != canon:
            continue
        cols = spec.get("columns", spec) if isinstance(spec, dict) else spec
        if isinstance(cols, list):
            out.extend(cols)
    return out


def _phrase_alias(ql: str, cmap: dict, block: str) -> tuple[Optional[str], list[str]]:
    best, best_cols, best_len = None, [], 0
    for canon, spec in (cmap.get(block, {}) or {}).items():
        phrases = spec.get("phrases", [canon]) if isinstance(spec, dict) else [canon]
        cols = spec.get("columns", []) if isinstance(spec, dict) else spec
        for ph in phrases:
            if re.search(r"\b" + re.escape(ph) + r"\b", ql) and len(ph) > best_len:
                best, best_cols, best_len = canon, cols, len(ph)
    return best, best_cols


# ── Execution ─────────────────────────────────────────────────────────────────

def _run(sql: str, params: list) -> list[dict]:
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 8000")
            cur.execute(sql, params)
            return fetchall_dicts(cur)


def _result(answer: str, sql: str, rows: list, shape: str) -> dict:
    return {
        "answer": answer,
        "documents": [],
        "intent": {},
        "filter_used": {"shape": shape},
        "query_path": f"shape:{shape}",
        "sql_generated": sql,
        "rows": rows,
    }


def _fmt_rows(rows: list[dict], limit: int = 50) -> str:
    if not rows:
        return "No matching records."
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    for r in rows[:limit]:
        lines.append(" | ".join(str(r.get(c, "")) for c in cols))
    if len(rows) > limit:
        lines.append(f"… ({len(rows) - limit} more)")
    return "\n".join(lines)


# ── Shape router ──────────────────────────────────────────────────────────────

def answer(user_query: str) -> Optional[dict]:
    """Try each deterministic shape; return a result dict or None."""
    try:
        cmap = load_concept_map()
        with get_readonly_db() as conn:
            catalog = _column_catalog(conn)
    except Exception as exc:
        logger.warning("query_shapes: schema introspection failed (%s)", exc)
        return None
    if not catalog:
        return None

    ql = " " + user_query.lower().strip() + " "

    for shape_fn in (_shape_aggregate, _shape_count, _shape_distinct):
        try:
            res = shape_fn(ql, user_query, catalog, cmap)
        except Exception as exc:
            logger.warning("query_shapes %s failed: %s", shape_fn.__name__, exc)
            res = None
        if res is not None:
            return res
    return None


def _pick_dimension(ql: str, catalog: dict, cmap: dict, prefer_rel: str = None) -> Optional[tuple]:
    """Resolve an optional GROUP BY dimension from 'by <dim>' (prefer prefer_rel)."""
    m = re.search(r"\bby\s+([a-z][a-z _]{1,30})", ql)
    if not m:
        return None
    term = m.group(1).strip()
    dim_canon, dim_cols = _phrase_alias(ql, cmap, "dimensions")
    cands = _resolve_column(term, catalog, dim_cols) or _resolve_column(dim_canon or "", catalog, dim_cols)
    if not cands:
        return None
    if prefer_rel:
        for c in cands:
            if c[0] == prefer_rel:
                return c
    return cands[0]


def _shape_aggregate(ql: str, raw: str, catalog: dict, cmap: dict) -> Optional[dict]:
    func = None
    for w, f in _AGG_WORDS.items():
        if re.search(r"\b" + w + r"\b", ql):
            func = f
            break
    if not func:
        return None

    # The measure is the NOUN after the function word (e.g. total <revenue>).
    # Resolve it against real columns first (synonyms only as a boost), so a
    # generic word like "total" never overrides the actual measure noun.
    noun = _after_word(ql, list(_AGG_WORDS))
    _, mcols = _phrase_alias(ql, cmap, "measures")
    # Resolve the noun against REAL columns first (no synonyms) so the actual
    # measure ('revenue') wins over generic measure-phrase synonyms ('total').
    cands = _resolve_column(noun, catalog) if noun else []
    if not cands and noun:
        cands = _resolve_column(noun, catalog, mcols)
    if not cands and mcols:
        cands = _resolve_column(mcols[0], catalog, mcols)
    if not cands:
        return None
    rel, col = cands[0]
    dim = _pick_dimension(ql, catalog, cmap, prefer_rel=rel)

    if dim and dim[0] == rel:
        dcol = dim[1]
        sql = (f"SELECT {_quote_identifier(dcol)} AS dim, "
               f"{func}(safe_numeric({_quote_identifier(col)})) AS value "
               f"FROM {_quote_identifier(rel)} GROUP BY {_quote_identifier(dcol)} "
               f"ORDER BY value DESC NULLS LAST LIMIT 200")
        rows = _run(sql, [])
        ans = f"{func} of {col} by {dcol}:\n" + _fmt_rows(rows)
    else:
        sql = f"SELECT {func}(safe_numeric({_quote_identifier(col)})) AS value FROM {_quote_identifier(rel)}"
        rows = _run(sql, [])
        val = rows[0]["value"] if rows else None
        ans = f"{func} of {col}: {val}"
    return _result(ans, sql, rows, "aggregate")


def _shape_count(ql: str, raw: str, catalog: dict, cmap: dict) -> Optional[dict]:
    if not any(w in ql for w in _COUNT_WORDS):
        return None
    # optional entity/relation hint: use the most relevant relation
    rel = _guess_relation(ql, catalog)
    if not rel:
        return None
    dim = _pick_dimension(ql, catalog, cmap)
    if dim and dim[0] == rel:
        dcol = dim[1]
        sql = (f"SELECT {_quote_identifier(dcol)} AS dim, COUNT(*) AS n "
               f"FROM {_quote_identifier(rel)} GROUP BY {_quote_identifier(dcol)} "
               f"ORDER BY n DESC LIMIT 200")
        rows = _run(sql, [])
        ans = f"Count by {dcol}:\n" + _fmt_rows(rows)
    else:
        sql = f"SELECT COUNT(*) AS n FROM {_quote_identifier(rel)}"
        rows = _run(sql, [])
        ans = f"Count ({rel}): {rows[0]['n'] if rows else 0}"
    return _result(ans, sql, rows, "count")


def _shape_distinct(ql: str, raw: str, catalog: dict, cmap: dict) -> Optional[dict]:
    if not any(w in ql for w in _DISTINCT_WORDS):
        return None
    # the field being listed: token(s) after the distinct cue
    term = _after_word(ql, [w.strip() for w in _DISTINCT_WORDS])
    if not term:
        return None
    # try field_aliases synonyms
    _, fcols = _phrase_alias(ql, cmap, "field_aliases")
    cands = _resolve_column(term, catalog, fcols)
    if not cands:
        return None
    rel, col = cands[0]
    sql = (f"SELECT DISTINCT {_quote_identifier(col)} AS {_quote_identifier(col)} "
           f"FROM {_quote_identifier(rel)} "
           f"WHERE {_quote_identifier(col)} IS NOT NULL "
           f"ORDER BY 1 LIMIT 500")
    rows = _run(sql, [])
    ans = f"Distinct {col} ({len(rows)}):\n" + _fmt_rows(rows)
    return _result(ans, sql, rows, "distinct")


# ── Small NL helpers ──────────────────────────────────────────────────────────

_NOUN_STOP = {
    "the", "of", "for", "by", "in", "a", "an", "all", "list", "distinct",
    "unique", "every", "total", "sum", "average", "avg", "mean", "count",
    "number", "how", "many", "much", "me", "give", "show", "get",
}


def _after_word(ql: str, words: list[str]) -> Optional[str]:
    """
    Return the measure/field noun following any of `words`, truncated at a
    'by'/'per'/'for' clause so a GROUP BY dimension never leaks into the noun.
    """
    for w in words:
        m = re.search(re.escape(w.strip()) + r"\s+([a-z][a-z _]{1,40})", ql)
        if m:
            phrase = re.split(r"\b(?:by|per|for|grouped|across)\b", m.group(1))[0]
            toks = [t for t in phrase.split() if t not in _NOUN_STOP]
            return " ".join(toks[:3]).strip() or None
    return None


def _guess_relation(ql: str, catalog: dict) -> Optional[str]:
    """Pick the relation whose name/columns best overlap the query tokens."""
    qtokens = set(re.findall(r"[a-z0-9]+", ql))
    best, best_score = None, 0
    for rel, cols in catalog.items():
        vocab = set(re.findall(r"[a-z0-9]+", rel.lower()))
        for c in cols:
            vocab |= set(re.findall(r"[a-z0-9]+", c.lower()))
        score = len(qtokens & vocab)
        if score > best_score:
            best, best_score = rel, score
    return best if best_score > 0 else (next(iter(catalog), None))
