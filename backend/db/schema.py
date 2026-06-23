"""
PostgreSQL schema and connection management for the structured facts database.

Why PostgreSQL alongside ChromaDB?
------------------------------------
ChromaDB (vector store) holds ONLY unstructured knowledge documents
(ICD/CPT guidebooks, medical reference PDFs). It cannot reliably answer
precise factual queries like "What is Alice Johnson's claim total on 6 May?"

PostgreSQL stores structured billing/clinical rows with indexed columns so
exact lookups are instant, deterministic, and verbatim — zero hallucination.
It also supports analytical queries (aggregations, joins, GROUP BY) that the
previous SQLite + template approach could not handle.

Schema design
--------------
  patients   — one row per unique patient (deduped by name + patient_id)
  providers  — one row per unique provider (deduped by NPI or name)
  records    — every clinical/billing event (visit, bill, prescription, lab)
               details_json catches any column that has no dedicated SQL field
  admissions — MIMIC-III admission-level data (ADMISSIONS.csv)

Connection model
-----------------
  get_db()          — read-write connection (POSTGRES_DSN), used by ingestion
  get_readonly_db() — query-path connection (POSTGRES_READONLY_DSN if set,
                      else POSTGRES_DSN with a warning). All SELECT queries and
                      LLM-generated SQL execute on this connection.
"""

import logging
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

from config import settings

logger = logging.getLogger(__name__)


# ── Connection helpers ────────────────────────────────────────────────────────

def _connect(dsn: str):
    """Open a psycopg2 connection with DictCursor row factory."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


@contextmanager
def get_db():
    """
    Yield a read-write psycopg2 connection (POSTGRES_DSN).
    Used exclusively by ingestion and write operations.

    Usage:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = _connect(settings.POSTGRES_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_readonly_db():
    """
    Yield a read-only psycopg2 connection.

    Uses POSTGRES_READONLY_DSN if configured; falls back to POSTGRES_DSN
    with a warning when it is not. All query-path SELECTs and LLM-generated
    SQL execute through this connection so that when the read-only role is
    eventually set up, nothing else needs to change.
    """
    dsn = settings.POSTGRES_READONLY_DSN or settings.POSTGRES_DSN
    if not settings.POSTGRES_READONLY_DSN:
        logger.warning(
            "get_readonly_db: POSTGRES_READONLY_DSN not set — "
            "executing on full-privilege connection. "
            "Set POSTGRES_READONLY_DSN to a SELECT-only role for production."
        )
    conn = _connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def dict_row(cursor, row) -> dict:
    """Convert a psycopg2 row to a plain dict using the cursor's description."""
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def fetchall_dicts(cursor) -> list[dict]:
    """Fetch all rows from `cursor` as a list of dicts."""
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


def fetchone_dict(cursor) -> dict | None:
    """Fetch one row from `cursor` as a dict, or None."""
    row = cursor.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
-- ── Patients ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id           SERIAL PRIMARY KEY,
    name         TEXT    NOT NULL,           -- Title Case normalised
    patient_id   TEXT,                       -- MRN / chart number
    dob          TEXT,                       -- YYYY-MM-DD
    gender       TEXT,
    phone        TEXT,
    address      TEXT,
    insurance_id TEXT,
    source_files TEXT,                       -- comma-separated filenames
    subject_id   TEXT,                       -- MIMIC-III SUBJECT_ID
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_name       ON patients(name);
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_subject_id ON patients(subject_id);

-- ── Providers ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS providers (
    id             SERIAL PRIMARY KEY,
    name           TEXT,                     -- Title Case normalised
    npi            TEXT UNIQUE,              -- 10-digit NPI
    specialty      TEXT,
    dob            TEXT,                     -- YYYY-MM-DD
    phone          TEXT,
    address        TEXT,
    license_number TEXT,
    source_files   TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
CREATE INDEX IF NOT EXISTS idx_providers_npi  ON providers(npi);

-- ── Records ───────────────────────────────────────────────────────────────────
-- One row per clinical event / billing record.
-- Supports 1 M+ rows: bigint PK, targeted indexes only.
CREATE TABLE IF NOT EXISTS records (
    id              BIGSERIAL PRIMARY KEY,
    patient_id      INTEGER REFERENCES patients(id)  ON DELETE CASCADE,
    provider_id     INTEGER REFERENCES providers(id) ON DELETE SET NULL,

    -- Event classification
    record_type     TEXT NOT NULL,   -- 'visit'|'bill'|'prescription'|'lab_result'|'other'

    -- Most-queried fields as dedicated columns (fast index lookups)
    record_date     TEXT,            -- YYYY-MM-DD
    total_amount    TEXT,            -- numeric string
    claim_number    TEXT,
    diagnosis       TEXT,
    treatment       TEXT,
    medication      TEXT,
    dosage          TEXT,
    test_name       TEXT,
    test_result     TEXT,
    reference_range TEXT,

    -- MIMIC-III fields
    hadm_id         TEXT,
    icd9_code       TEXT,
    icd9_description TEXT,
    seq_num         TEXT,

    -- Catch-all overflow
    details_json    TEXT,            -- JSON object of additional key-value pairs

    -- Anti-hallucination: verbatim source text for exact quoting
    raw_text        TEXT NOT NULL,
    source_file     TEXT,
    page_number     TEXT,
    chunk_index     TEXT,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_records_patient_id   ON records(patient_id);
CREATE INDEX IF NOT EXISTS idx_records_provider_id  ON records(provider_id);
CREATE INDEX IF NOT EXISTS idx_records_type         ON records(record_type);
CREATE INDEX IF NOT EXISTS idx_records_date         ON records(record_date);
CREATE INDEX IF NOT EXISTS idx_records_claim_number ON records(claim_number);
CREATE INDEX IF NOT EXISTS idx_records_hadm_id      ON records(hadm_id);
CREATE INDEX IF NOT EXISTS idx_records_icd9_code    ON records(icd9_code);

-- ── Admissions ────────────────────────────────────────────────────────────────
-- MIMIC-III ADMISSIONS.csv: one row per hospital admission.
CREATE TABLE IF NOT EXISTS admissions (
    id             SERIAL PRIMARY KEY,
    patient_id     INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    subject_id     TEXT,
    hadm_id        TEXT UNIQUE,
    admittime      TEXT,
    dischtime      TEXT,
    deathtime      TEXT,
    admission_type TEXT,
    diagnosis      TEXT,
    source_file    TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_admissions_patient_id ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id);
CREATE INDEX IF NOT EXISTS idx_admissions_hadm_id    ON admissions(hadm_id);
"""

# Columns that may not exist on older databases — added with IF NOT EXISTS.
# PostgreSQL supports ADD COLUMN IF NOT EXISTS natively (unlike SQLite).
_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, type)  — already included in _DDL above for fresh installs;
    # these run as no-ops on fresh DBs and add the column on older ones.
    ("patients", "subject_id",       "TEXT"),
    ("records",  "hadm_id",          "TEXT"),
    ("records",  "icd9_code",        "TEXT"),
    ("records",  "icd9_description", "TEXT"),
    ("records",  "seq_num",          "TEXT"),
    ("admissions", "deathtime",      "TEXT"),
]


def _run_migrations(conn) -> None:
    """ADD COLUMN IF NOT EXISTS for each migration entry — idempotent."""
    with conn.cursor() as cur:
        for table, column, col_type in _MIGRATIONS:
            try:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.warning("Migration skipped (%s.%s): %s", table, column, exc)


def init_db() -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every startup — fully idempotent.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        _run_migrations(conn)
        with conn.cursor() as cur:
            cur.execute(_CATALOG_DDL)
        conn.commit()
    logger.info("PostgreSQL database ready — schema initialised")


# ── Identifier quoting ────────────────────────────────────────────────────────

def _quote_identifier(name: str) -> str:
    """Double-quote a PostgreSQL identifier, escaping any embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


# ── Staging catalog DDL ───────────────────────────────────────────────────────
# Kept separate from _DDL so canonical tables (patients/providers/records/admissions)
# are created first, then the catalog infrastructure.

_CATALOG_DDL = """
-- ── Safe-cast helper functions ────────────────────────────────────────────────
-- Used by per-stream VIEWs so that a single bad cell (e.g. "$1,200", "", "N/A")
-- returns NULL instead of erroring the whole query.

CREATE OR REPLACE FUNCTION safe_numeric(v TEXT) RETURNS NUMERIC
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    cleaned  TEXT;
    negative BOOLEAN;
BEGIN
    IF v IS NULL THEN RETURN NULL; END IF;
    -- Accounting notation: (100.00) → -100.00
    negative := TRIM(v) LIKE '(%' AND TRIM(v) LIKE '%)';
    -- Strip everything except digits and decimal point
    cleaned  := NULLIF(TRIM(regexp_replace(v, '[^0-9.]', '', 'g')), '');
    IF cleaned IS NULL THEN RETURN NULL; END IF;
    IF negative OR TRIM(v) LIKE '-%' THEN
        RETURN -(cleaned::NUMERIC);
    END IF;
    RETURN cleaned::NUMERIC;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION safe_date(v TEXT, fmt TEXT DEFAULT 'YYYY-MM-DD') RETURNS DATE
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    v := NULLIF(TRIM(v), '');
    IF v IS NULL THEN RETURN NULL; END IF;
    RETURN TO_DATE(v, fmt);
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION safe_timestamptz(
    v   TEXT,
    fmt TEXT DEFAULT 'YYYY-MM-DD HH24:MI:SS'
) RETURNS TIMESTAMPTZ
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    v := NULLIF(TRIM(v), '');
    IF v IS NULL THEN RETURN NULL; END IF;
    RETURN TO_TIMESTAMP(v, fmt);
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

-- ── source_catalog ────────────────────────────────────────────────────────────
-- One row per unique column-signature (stream).  Streams that route to another
-- stream's table have aliased_to set and staging_table = NULL.

CREATE TABLE IF NOT EXISTS source_catalog (
    column_signature       TEXT        PRIMARY KEY,
    staging_table          TEXT,                        -- NULL for aliased rows
    aliased_to             TEXT
        REFERENCES source_catalog(column_signature),   -- non-NULL = follow this sig
    representative_headers JSONB       NOT NULL,        -- raw header strings (first seen)
    safe_columns           JSONB       NOT NULL,        -- all sanitised cols, pre-denylist
    stored_columns         JSONB,                       -- post-denylist cols in the table
    human_label            TEXT,                        -- set by catalog_admin label
    load_mode              TEXT        NOT NULL DEFAULT 'append',   -- append | snapshot
    natural_key            JSONB,                       -- list of col names for upsert
    query_exposed          BOOLEAN     NOT NULL DEFAULT FALSE,
    column_mapping         JSONB,       -- {native_safe: {canonical, type, format?}}
    view_name              TEXT,        -- v_<human_label> set on expose
    candidate_drift_of     TEXT
        REFERENCES source_catalog(column_signature),   -- set when drift is detected
    rows_total             BIGINT      DEFAULT 0,
    rows_last_inserted     BIGINT      DEFAULT 0,
    first_ingested_at      TIMESTAMPTZ,
    last_ingested_at       TIMESTAMPTZ
);

-- Only table-owning rows (aliased_to IS NULL) may hold a staging_table name.
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_catalog_staging_table
    ON source_catalog(staging_table)
    WHERE aliased_to IS NULL AND staging_table IS NOT NULL;
"""


# ── Staging table helpers ─────────────────────────────────────────────────────

def _create_staging_table(conn, staging_table: str, stored_cols: list[str]) -> None:
    """
    CREATE TABLE IF NOT EXISTS stg_<sig8> with TEXT columns for every stored col
    plus the four system columns (_source_file, _row_hash, _ingested_at, _deleted_at).
    Idempotent — safe to call on re-ingest.
    """
    safe_table = _quote_identifier(staging_table)
    col_defs   = ",\n    ".join(
        f"{_quote_identifier(c)} TEXT" for c in stored_cols
    )
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
            _id          BIGSERIAL    PRIMARY KEY,
            {col_defs},
            _source_file TEXT         NOT NULL,
            _row_hash    TEXT         NOT NULL,
            _ingested_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            _deleted_at  TIMESTAMPTZ
        )
    """
    idx_prefix = staging_table.replace('"', "")  # safe for index names
    with conn.cursor() as cur:
        cur.execute(ddl)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{idx_prefix}_src "
            f"ON {safe_table}(_source_file)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{idx_prefix}_hash "
            f"ON {safe_table}(_row_hash)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{idx_prefix}_del "
            f"ON {safe_table}(_deleted_at)"
        )
    conn.commit()
    logger.info("Staging table ready: %s (%d cols)", staging_table, len(stored_cols))


def _sync_staging_columns(
    conn, staging_table: str, new_cols: list[str],
) -> None:
    """
    ADD COLUMN IF NOT EXISTS for any column in new_cols not already in the table.
    Skips system columns (names starting with '_').
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            [staging_table],
        )
        existing = {row[0] for row in cur.fetchall()}

    added = []
    for col in new_cols:
        if col.startswith("_") or col in existing:
            continue
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {_quote_identifier(staging_table)} "
                f"ADD COLUMN IF NOT EXISTS {_quote_identifier(col)} TEXT"
            )
        added.append(col)

    if added:
        conn.commit()
        logger.info("Added %d column(s) to %s: %s", len(added), staging_table, added)


def _build_and_create_view(conn, stream: dict) -> str:
    """
    Build and execute CREATE OR REPLACE VIEW v_<human_label>.

    SELECT clause uses safe cast helpers (safe_numeric / safe_date /
    safe_timestamptz) per column_mapping type hints so a single dirty
    cell returns NULL instead of erroring the whole query.

    column_mapping format:
        {native_safe_name: {canonical: str, type: str, format?: str}}
    Supported types: text (default), numeric, date, timestamptz.

    After CREATE, GRANTs SELECT to settings.POSTGRES_READONLY_ROLE if set.
    Validates that every column_mapping key exists in the staging table;
    raises ValueError distinguishing denylist-dropped vs. typo.

    Returns the view name.
    """
    import re
    from config import settings

    label = stream.get("human_label")
    if not label:
        raise ValueError(
            f"Stream {stream.get('column_signature', '?')[:8]}: "
            "human_label must be set before expose"
        )

    mapping = stream.get("column_mapping") or {}
    if not mapping:
        raise ValueError(
            f"Stream '{label}': column_mapping is empty — "
            "fill it in sources.yaml and run catalog_admin expose again"
        )

    staging_table  = stream["staging_table"]
    view_name      = "v_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    safe_cols_json = stream.get("safe_columns") or "[]"
    stored_json    = stream.get("stored_columns") or "[]"

    import json as _json
    try:
        safe_set   = set(_json.loads(safe_cols_json) if isinstance(safe_cols_json, str) else safe_cols_json)
        stored_set = set(_json.loads(stored_json)    if isinstance(stored_json, str)    else stored_json)
    except Exception:
        safe_set   = set()
        stored_set = set()

    # Fetch actual columns from the staging table (ground truth)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            [staging_table],
        )
        table_cols = {row[0] for row in cur.fetchall()}

    # Validate — hard fail with precise error messages (Fix B requirement)
    missing_in_table = []
    for native_safe in mapping:
        if native_safe not in table_cols:
            if native_safe in safe_set and native_safe not in stored_set:
                missing_in_table.append(
                    f"'{native_safe}' (dropped by PHI denylist — not stored)"
                )
            else:
                missing_in_table.append(
                    f"'{native_safe}' (not found — check safe-identifier spelling)"
                )
    if missing_in_table:
        raise ValueError(
            f"Stream '{label}': column_mapping keys missing from {staging_table}:\n"
            + "\n".join(f"  • {m}" for m in missing_in_table)
        )

    # Build SELECT clause with safe casts
    _DEFAULT_DATE_FMT = "YYYY-MM-DD"
    _DEFAULT_TS_FMT   = "YYYY-MM-DD HH24:MI:SS"

    select_parts = []
    for native_safe, spec in mapping.items():
        qn  = _quote_identifier(native_safe)
        qc  = _quote_identifier(spec["canonical"])
        typ = (spec.get("type") or "text").lower()
        fmt = spec.get("format") or (
            _DEFAULT_DATE_FMT if typ == "date" else _DEFAULT_TS_FMT
        )

        if typ == "numeric":
            expr = f"safe_numeric({qn})"
        elif typ == "date":
            expr = f"safe_date({qn}, '{fmt}')"
        elif typ == "timestamptz":
            expr = f"safe_timestamptz({qn}, '{fmt}')"
        else:
            expr = qn  # text — no cast

        select_parts.append(f"    {expr} AS {qc}")

    cols_sql  = ",\n".join(select_parts)
    qview     = _quote_identifier(view_name)
    qtable    = _quote_identifier(staging_table)
    view_ddl  = (
        f"CREATE OR REPLACE VIEW {qview} AS\n"
        f"SELECT\n{cols_sql}\n"
        f"FROM   {qtable}\n"
        f"WHERE  _deleted_at IS NULL"
    )

    with conn.cursor() as cur:
        cur.execute(view_ddl)
    conn.commit()
    logger.info("VIEW created: %s → %s", view_name, staging_table)

    # Grant SELECT to read-only role (Fix B)
    readonly_role = (settings.POSTGRES_READONLY_ROLE or "").strip()
    if readonly_role:
        with conn.cursor() as cur:
            cur.execute(
                f"GRANT SELECT ON {qview} TO {_quote_identifier(readonly_role)}"
            )
        conn.commit()
        logger.info("GRANTED SELECT ON %s TO %s", view_name, readonly_role)
    else:
        logger.warning(
            "POSTGRES_READONLY_ROLE not set — "
            "read-only role will not have SELECT on %s. "
            "Set POSTGRES_READONLY_ROLE in .env and re-run expose.",
            view_name,
        )

    # Update catalog
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_catalog "
            "SET query_exposed = TRUE, view_name = %s "
            "WHERE column_signature = %s",
            [view_name, stream["column_signature"]],
        )
    conn.commit()
    return view_name


def _reconcile_snapshot_deletions(
    conn,
    staging_table:   str,
    filename:        str,
    current_hashes:  list[str],
    total_rows:      int,
) -> int:
    """
    Soft-delete rows from staging_table that are absent from the current file.

    Guards:
      1. live_count == 0 → skip (first load, nothing to reconcile)
      2. total_rows < ABSOLUTE_FLOOR → skip (file suspiciously short)
      3. total_rows < live_count * MIN_FRACTION → skip (would mass-delete)

    Uses a TEMP table anti-join so no large array parameters are sent.
    Returns number of rows soft-deleted, or -1 if skipped.
    """
    from psycopg2.extras import execute_values
    from config import settings

    min_fraction   = float(settings.STAGING_RECONCILE_MIN_FRACTION)
    absolute_floor = int(settings.STAGING_RECONCILE_ABSOLUTE_FLOOR)
    qtable         = _quote_identifier(staging_table)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {qtable} "
            f"WHERE _source_file = %s AND _deleted_at IS NULL",
            [filename],
        )
        live_count = cur.fetchone()[0]

    if live_count == 0:
        logger.info(
            "Reconcile skipped for %s/%s: no live rows (first load)",
            staging_table, filename,
        )
        return -1

    if total_rows < absolute_floor:
        logger.warning(
            "Reconcile SKIPPED %s/%s: file has %d rows — "
            "below absolute floor %d. Investigate file.",
            staging_table, filename, total_rows, absolute_floor,
        )
        return -1

    if total_rows < live_count * min_fraction:
        logger.warning(
            "Reconcile SKIPPED %s/%s: %d rows in file vs %d live "
            "(%.1f%% present, threshold %.0f%%) — refusing to mass-delete.",
            staging_table, filename,
            total_rows, live_count,
            100.0 * total_rows / live_count,
            min_fraction * 100,
        )
        return -1

    # TEMP table + anti-join (Fix 5 — no large array parameter)
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _recon_hashes (_row_hash TEXT PRIMARY KEY) "
            "ON COMMIT DROP"
        )
        if current_hashes:
            execute_values(
                cur,
                "INSERT INTO _recon_hashes (_row_hash) VALUES %s "
                "ON CONFLICT DO NOTHING",
                [(h,) for h in current_hashes],
            )
        cur.execute(
            f"UPDATE {qtable} t "
            f"SET    _deleted_at = NOW() "
            f"WHERE  t._source_file = %s "
            f"  AND  t._deleted_at  IS NULL "
            f"  AND  NOT EXISTS ("
            f"      SELECT 1 FROM _recon_hashes r "
            f"      WHERE  r._row_hash = t._row_hash"
            f"  )",
            [filename],
        )
        deleted = cur.rowcount

    conn.commit()
    logger.info(
        "Reconcile: %d row(s) soft-deleted from %s/%s",
        deleted, staging_table, filename,
    )
    return deleted
