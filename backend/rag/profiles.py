"""
Generic per-patient profile (schema-on-read).

Returns EVERY column the source files hold for a patient — name, age, phone,
address, country, DOB, claim status, account number, charges/payments, etc. —
WITHOUT naming columns in code. It reads the raw staging tables (stg_<sig>),
which already store every ingested column, so:

  • it needs no events backfill (staging is always current), and
  • a file with a different column structure just works — its columns are
    returned as-is (a new schema = a new staging table, no code change).

The only config-driven (not hardcoded) bits are *semantic*: which column is the
patient name/key, and which columns are charges/payments — all resolved through
data/concept_map.yaml synonyms. Add a synonym there for a new dataset's naming;
never edit Python.
"""
from __future__ import annotations

import logging
from typing import Optional

import hashlib

from db.schema import get_db, get_readonly_db, fetchall_dicts, _quote_identifier
from rag.events import load_concept_map, _normalize_col, _parse_number

logger = logging.getLogger(__name__)

_SYSTEM_PREFIX = "_"


def ensure_entity_indexes() -> dict:
    """
    Create the indexes that make the all-columns entity-profile path scale to
    1M+ rows, dynamically per staging table (column names vary by dataset):

      • btree index on each entity KEY column (patient/provider key)  → exact `=`
      • pg_trgm GIN index on lower(NAME column)                       → ILIKE '%…%'

    Idempotent (CREATE INDEX IF NOT EXISTS); safe to call after every ingest.
    Run AFTER ingestion so the index build doesn't contend with row writes.
    """
    cmap = load_concept_map()
    name_norm = {
        _normalize_col(c)
        for c in (cmap.get("patient_name_columns", []) + cmap.get("provider_name_columns", []))
    }
    key_norm = {
        _normalize_col(c)
        for c in (cmap.get("patient_key_columns", []) + cmap.get("provider_key_columns", [])
                  + cmap.get("document_key_columns", []))
    }

    created = 0
    with get_db() as conn:
        # pg_trgm is required for the GIN name index (defensive — init_db also does this).
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("pg_trgm extension unavailable (%s) — name index may be a scan", exc)

        tables = _staging_tables(conn)
        for tbl in tables:
            for col in _table_columns(conn, tbl):
                if col.startswith(_SYSTEM_PREFIX):
                    continue
                nc = _normalize_col(col)
                stmts = []
                if nc in key_norm:
                    idx = "ix_" + hashlib.md5(f"{tbl}.{col}.btree".encode()).hexdigest()[:16]
                    stmts.append(
                        f"CREATE INDEX IF NOT EXISTS {idx} "
                        f"ON {_quote_identifier(tbl)} ({_quote_identifier(col)})"
                    )
                if nc in name_norm:
                    idx = "ix_" + hashlib.md5(f"{tbl}.{col}.trgm".encode()).hexdigest()[:16]
                    stmts.append(
                        f"CREATE INDEX IF NOT EXISTS {idx} "
                        f"ON {_quote_identifier(tbl)} USING gin (lower({_quote_identifier(col)}) gin_trgm_ops)"
                    )
                for stmt in stmts:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(stmt)
                        conn.commit()
                        created += 1
                    except Exception as exc:
                        conn.rollback()
                        logger.warning("Entity index skipped (%s.%s): %s", tbl, col, exc)

    logger.info("ensure_entity_indexes: %d index statement(s) ensured", created)
    return {"entity_indexes_ensured": created}


def _staging_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT staging_table FROM source_catalog "
            "WHERE staging_table IS NOT NULL AND aliased_to IS NULL"
        )
        return [r[0] for r in cur.fetchall()]


def _table_columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s",
            [table],
        )
        return [r[0] for r in cur.fetchall()]


def _match_synonym(columns: list[str], synonyms: list[str]) -> Optional[str]:
    """Return the ACTUAL column whose normalised name matches a synonym (in order)."""
    norm_to_actual = {_normalize_col(c): c for c in columns}
    for syn in synonyms:
        s = _normalize_col(syn)
        if s in norm_to_actual:
            return norm_to_actual[s]
    return None


def patient_profile(name: Optional[str] = None, key: Optional[str] = None, limit: int = 5000) -> dict:
    """Backward-compatible wrapper — full profile for a PATIENT."""
    return entity_profile(name=name, key=key, role="patient", limit=limit)


def entity_profile(
    name: Optional[str] = None,
    key: Optional[str] = None,
    role: str = "patient",
    limit: int = 5000,
) -> dict:
    """
    Build a full profile for an entity (patient or provider) across all staging
    tables — EVERY column the source files hold for that entity.

    role: "patient" | "provider" — selects which concept_map name/key synonyms
    are used to resolve and match the entity.

    Returns:
      {
        "entity": str, "role": str, "n_rows": int, "source_tables": [...],
        "identity": {column: value},        # constant fields
        "totals":   {column: sum},          # summed charge/payment columns
        "line_items": [ {column: value, …}, … ],  # varying per-row fields
      }
    """
    cmap = load_concept_map()
    if role == "provider":
        name_syns = cmap.get("provider_name_columns", [])
        key_syns = cmap.get("provider_key_columns", [])
    else:
        name_syns = cmap.get("patient_name_columns", [])
        key_syns = cmap.get("patient_key_columns", [])
    charge_norm = {
        _normalize_col(c)
        for c in (cmap.get("concepts", {}).get("charge", {}).get("value_columns", []))
    }

    all_rows: list[dict] = []
    used_tables: list[str] = []

    with get_readonly_db() as conn:
        for tbl in _staging_tables(conn):
            cols = _table_columns(conn, tbl)
            name_col = _match_synonym(cols, name_syns)
            key_col = _match_synonym(cols, key_syns)

            where, params = [], []
            if key and key_col:
                where.append(f"{_quote_identifier(key_col)} = %s")
                params.append(str(key))
            elif name and name_col:
                tokens = [t for t in name.lower().replace(",", " ").split() if len(t) >= 2]
                if not tokens:
                    continue
                for tok in tokens[:5]:
                    where.append(f"lower({_quote_identifier(name_col)}) LIKE %s")
                    params.append(f"%{tok}%")
            else:
                continue

            sql = (
                f"SELECT * FROM {_quote_identifier(tbl)} "
                f"WHERE _deleted_at IS NULL AND {' AND '.join(where)} "
                f"LIMIT {int(limit)}"
            )
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = fetchall_dicts(cur)
            if rows:
                used_tables.append(tbl)
                all_rows.extend(rows)

    if not all_rows:
        return {}

    # Drop system columns
    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith(_SYSTEM_PREFIX)}

    rows = [clean(r) for r in all_rows]
    columns = sorted({c for r in rows for c in r})

    # Identity = columns with exactly one distinct non-empty value across rows
    identity, varying = {}, []
    for col in columns:
        vals = {str(r.get(col)).strip() for r in rows if r.get(col) not in (None, "")}
        if len(vals) == 1:
            identity[col] = next(iter(vals))

    varying_cols = [c for c in columns if c not in identity]

    # Totals: sum charge/payment columns (config-driven, not hardcoded)
    totals = {}
    for col in columns:
        if _normalize_col(col) in charge_norm:
            s = 0.0
            seen = False
            for r in rows:
                n = _parse_number(r.get(col))
                if n is not None:
                    s += n
                    seen = True
            if seen:
                totals[col] = round(s, 2)

    line_items = [{c: r.get(c) for c in varying_cols if r.get(c) not in (None, "")} for r in rows]

    display = identity.get(_match_synonym(columns, name_syns) or "", None) or name or str(key)

    return {
        "entity": display,
        "patient": display,   # backward-compat alias
        "role": role,
        "n_rows": len(rows),
        "source_tables": used_tables,
        "identity": identity,
        "totals": totals,
        "line_items": line_items,
    }


def profile_context(profile: dict, max_rows: int = 30) -> str:
    """
    Serialise an entity profile into compact, LLM-ready structured context for
    entity QA. Always includes identity fields + totals; caps line items so the
    context stays bounded on a local model.
    """
    if not profile:
        return ""
    lines = [f"{profile.get('role', 'patient').upper()}: {profile.get('entity')}"]
    lines.append(f"(records: {profile.get('n_rows', 0)})")
    if profile.get("identity"):
        lines.append("\nFixed fields:")
        for k, v in profile["identity"].items():
            lines.append(f"  {k}: {v}")
    if profile.get("totals"):
        lines.append("\nSummed amounts:")
        for k, v in profile["totals"].items():
            lines.append(f"  {k}: {v}")
    items = profile.get("line_items", [])
    if items:
        shown = items[:max_rows]
        lines.append(f"\nPer-record details (showing {len(shown)} of {len(items)}):")
        for i, it in enumerate(shown, 1):
            pairs = " | ".join(f"{k}: {v}" for k, v in it.items())
            lines.append(f"  [{i}] {pairs}")
    return "\n".join(lines)


def format_profile(profile: dict) -> str:
    """Render a profile as a readable text answer."""
    if not profile:
        return "No matching patient found."
    lines = [f"Profile for {profile['patient']} ({profile['n_rows']} record(s)):", ""]
    if profile["identity"]:
        lines.append("Patient details:")
        for k, v in profile["identity"].items():
            lines.append(f"  • {k}: {v}")
        lines.append("")
    if profile["totals"]:
        lines.append("Totals:")
        for k, v in profile["totals"].items():
            lines.append(f"  • {k}: {v}")
        lines.append("")
    n = len(profile["line_items"])
    if n:
        lines.append(f"Line items: {n} record(s) (per-claim detail available).")
    return "\n".join(lines)
