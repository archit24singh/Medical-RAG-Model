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
"""


def init_db() -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every startup — idempotent.
    """
    with get_db() as conn:
        conn.executescript(_DDL)
    logger.info("SQLite database ready at %s", settings.DB_PATH)
