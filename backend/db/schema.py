"""
SQLite schema and connection management for the structured facts database.

Why SQLite alongside ChromaDB?
-------------------------------
ChromaDB (vector store) is excellent for semantic / similarity search but
cannot reliably answer precise factual queries like:
  "What is Alice Johnson's date of visit on 6 May 2020?"

SQLite stores the same facts as structured rows with indexed columns so
that exact lookups are instant, deterministic, and verbatim — no LLM
involved in returning the answer, so zero hallucination is possible.

Schema design
-------------
  patients   — one row per unique patient (deduped by name + patient_id)
  providers  — one row per unique provider (deduped by NPI or name)
  records    — every clinical event extracted from every document chunk
               record_type: 'visit' | 'bill' | 'prescription' | 'lab_result' | 'other'
               raw_text: verbatim chunk text — used for exact quoting

The records table carries the most-queried fields as proper columns
(record_date, total_amount, diagnosis, medication, test_name) plus a
JSON blob (details_json) for everything else so no information is lost.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """
    Yield an open SQLite connection with row_factory set to Row
    so results can be accessed by column name.

    Usage:
        with get_db() as conn:
            conn.execute(...)
    """
    db_path = Path(settings.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────

_DDL = """
-- ── Patients ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,           -- Title Case normalised
    patient_id  TEXT,                       -- MRN / chart number
    dob         TEXT,                       -- YYYY-MM-DD
    gender      TEXT,
    phone       TEXT,
    address     TEXT,
    insurance_id TEXT,
    source_files TEXT,                      -- comma-separated filenames
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patients_name      ON patients(name);
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);

-- ── Providers ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS providers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,                       -- Title Case normalised
    npi         TEXT UNIQUE,               -- 10-digit NPI (primary unique key)
    specialty   TEXT,
    dob         TEXT,                       -- YYYY-MM-DD
    phone       TEXT,
    address     TEXT,
    license_number TEXT,
    source_files TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
CREATE INDEX IF NOT EXISTS idx_providers_npi  ON providers(npi);

-- ── Records ───────────────────────────────────────────────────────────────────
-- One row per clinical event / billing record extracted from a document chunk.
CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    provider_id   INTEGER REFERENCES providers(id) ON DELETE SET NULL,

    -- Event classification
    record_type   TEXT NOT NULL,        -- 'visit' | 'bill' | 'prescription' | 'lab_result' | 'other'

    -- Most-queried fields as proper columns (fast indexed lookups)
    record_date   TEXT,                 -- YYYY-MM-DD  (visit/bill/prescription/lab date)
    total_amount  TEXT,                 -- Bills: total amount due (numeric string)
    claim_number  TEXT,                 -- Bills: insurance claim number / claim ID
    diagnosis     TEXT,                 -- Visits: primary diagnosis
    treatment     TEXT,                 -- Visits: treatment / procedure
    medication    TEXT,                 -- Prescriptions: drug name
    dosage        TEXT,                 -- Prescriptions: dosage
    test_name     TEXT,                 -- Lab results: name of the test
    test_result   TEXT,                 -- Lab results: result value
    reference_range TEXT,              -- Lab results: normal range

    -- Flexible overflow: everything else the extractor found
    details_json  TEXT,                 -- JSON object of additional key-value pairs

    -- Anti-hallucination: verbatim source text for exact quoting
    raw_text      TEXT NOT NULL,        -- exact text of the source chunk
    source_file   TEXT,
    page_number   TEXT,
    chunk_index   TEXT,

    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_records_patient_id   ON records(patient_id);
CREATE INDEX IF NOT EXISTS idx_records_provider_id  ON records(provider_id);
CREATE INDEX IF NOT EXISTS idx_records_type         ON records(record_type);
CREATE INDEX IF NOT EXISTS idx_records_date         ON records(record_date);

-- ── Admissions ───────────────────────────────────────────────────────────────
-- Admission/encounter-level data (MIMIC-III ADMISSIONS.csv: one row per
-- hospital admission, keyed by HADM_ID, linked to a patient via SUBJECT_ID).
-- Not every patient row has an admission — this table is populated only when
-- admission-level source data is ingested.
CREATE TABLE IF NOT EXISTS admissions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    subject_id      TEXT,                -- MIMIC-III SUBJECT_ID (patient identifier)
    hadm_id         TEXT UNIQUE,         -- MIMIC-III HADM_ID (admission identifier)
    admittime       TEXT,                -- admission timestamp
    dischtime       TEXT,                -- discharge timestamp
    deathtime       TEXT,                -- time of death, if patient expired during admission
    admission_type  TEXT,                -- e.g. EMERGENCY, ELECTIVE, URGENT
    diagnosis       TEXT,                -- free-text admitting diagnosis
    source_file     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_admissions_patient_id ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id);
CREATE INDEX IF NOT EXISTS idx_admissions_hadm_id    ON admissions(hadm_id);
"""

# Indexes on columns that may have been added by _MIGRATIONS (below) rather
# than present in the original CREATE TABLE. These MUST run AFTER
# _run_migrations() — if a database created before a migration was added
# still lacks the column, "CREATE INDEX ... ON records(<new column>)" would
# fail with "no such column", and (since CREATE TABLE IF NOT EXISTS is a
# no-op for an existing table) that failure would happen before the
# migration that adds the column ever runs.
_DDL_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_records_claim_number ON records(claim_number);
CREATE INDEX IF NOT EXISTS idx_patients_subject_id  ON patients(subject_id);
CREATE INDEX IF NOT EXISTS idx_records_hadm_id      ON records(hadm_id);
CREATE INDEX IF NOT EXISTS idx_records_icd9_code    ON records(icd9_code);
"""


# ── Migrations for existing databases ────────────────────────────────────────
# init_db() is called on every startup, but CREATE TABLE IF NOT EXISTS won't add
# new columns to a table that already exists from a previous version. Add any
# newly-introduced columns here so upgrades don't require wiping db_data.
_MIGRATIONS = [
    ("records", "claim_number", "TEXT"),
    # MIMIC-III migration: SUBJECT_ID identifies a patient across admissions;
    # HADM_ID/ICD9_CODE/SEQ_NUM identify a single admission-level diagnosis row
    # (DIAGNOSES_ICD.csv). icd9_description is the joined D_ICD_DIAGNOSES text.
    ("patients", "subject_id", "TEXT"),
    ("records", "hadm_id", "TEXT"),
    ("records", "icd9_code", "TEXT"),
    ("records", "icd9_description", "TEXT"),
    ("records", "seq_num", "TEXT"),
    ("admissions", "deathtime", "TEXT"),
]


def _run_migrations(conn) -> None:
    for table, column, col_type in _MIGRATIONS:
        existing_cols = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info("Migration: added column %s.%s", table, column)


def init_db() -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every startup — idempotent.

    Order matters: tables first, then migrations (which may add columns to
    tables that already existed from a previous version), then any indexes
    that depend on those migrated columns. See _DDL_POST_MIGRATION_INDEXES
    for why this ordering is required.
    """
    with get_db() as conn:
        conn.executescript(_DDL)
        _run_migrations(conn)
        conn.executescript(_DDL_POST_MIGRATION_INDEXES)
    logger.info("SQLite database ready at %s", settings.DB_PATH)
