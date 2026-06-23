"""
One-time migration: copy all rows from SQLite medical_rag.db → PostgreSQL.

Reads from the old SQLite structured facts database and inserts every row into
PostgreSQL in FK-dependency order:
    patients → providers → records → admissions

Uses INSERT … ON CONFLICT (id) DO NOTHING throughout — safe to run multiple
times.  After inserting, automatically resets each table's SERIAL/BIGSERIAL
sequence so subsequent application INSERTs get non-conflicting IDs.

Requirements
------------
  pip install psycopg2-binary  (already in requirements.txt)

Usage
-----
    cd backend
    python ../scripts/migrate_sqlite_to_postgres.py \\
        [--sqlite-path PATH] \\
        [--postgres-dsn DSN] \\
        [--dry-run]

Arguments
---------
  --sqlite-path   Path to medical_rag.db  (default: ../medical_rag.db relative to script)
  --postgres-dsn  Full DSN, e.g. postgresql://user:pass@host:5432/dbname
                  (falls back to $POSTGRES_DSN env var)
  --dry-run       Count rows and report, but insert nothing

After running
-------------
Verify the row counts match:
    psql $POSTGRES_DSN -c "SELECT 'patients', COUNT(*) FROM patients
      UNION ALL SELECT 'providers', COUNT(*) FROM providers
      UNION ALL SELECT 'records',   COUNT(*) FROM records
      UNION ALL SELECT 'admissions',COUNT(*) FROM admissions;"
"""

import argparse
import logging
import os
import sqlite3
import sys

# Allow importing the project if the script is run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# Default path: one directory above this script (i.e. project root)
_DEFAULT_SQLITE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "medical_rag.db")
)

# ── Table migration specs (FK order) ─────────────────────────────────────────
#
# Each entry has:
#   name      — table name
#   select    — SELECT from SQLite (column order must match INSERT below)
#   insert    — INSERT into PostgreSQL with ON CONFLICT DO NOTHING
#   sequence  — PostgreSQL sequence to reset after migration
#   fallback  — columns to skip if the SQLite table is missing them (old schema)

_TABLES = [
    {
        "name": "patients",
        "select": """
            SELECT id, name, patient_id, dob, gender, phone, address,
                   insurance_id, source_files, subject_id, created_at, updated_at
            FROM patients
        """,
        "insert": """
            INSERT INTO patients
                (id, name, patient_id, dob, gender, phone, address,
                 insurance_id, source_files, subject_id, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """,
        "sequence": "patients_id_seq",
        "ncols": 12,
    },
    {
        "name": "providers",
        "select": """
            SELECT id, name, npi, specialty, dob, phone, address,
                   license_number, source_files, created_at, updated_at
            FROM providers
        """,
        "insert": """
            INSERT INTO providers
                (id, name, npi, specialty, dob, phone, address,
                 license_number, source_files, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """,
        "sequence": "providers_id_seq",
        "ncols": 11,
    },
    {
        "name": "records",
        "select": """
            SELECT id, patient_id, provider_id, record_type, record_date,
                   total_amount, claim_number, diagnosis, treatment,
                   medication, dosage, test_name, test_result, reference_range,
                   details_json, raw_text, source_file, page_number, chunk_index,
                   hadm_id, icd9_code, icd9_description, seq_num, created_at
            FROM records
        """,
        "insert": """
            INSERT INTO records
                (id, patient_id, provider_id, record_type, record_date,
                 total_amount, claim_number, diagnosis, treatment,
                 medication, dosage, test_name, test_result, reference_range,
                 details_json, raw_text, source_file, page_number, chunk_index,
                 hadm_id, icd9_code, icd9_description, seq_num, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """,
        "sequence": "records_id_seq",
        "ncols": 24,
    },
    {
        "name": "admissions",
        "select": """
            SELECT id, patient_id, subject_id, hadm_id, admittime, dischtime,
                   deathtime, admission_type, diagnosis, source_file, created_at
            FROM admissions
        """,
        "insert": """
            INSERT INTO admissions
                (id, patient_id, subject_id, hadm_id, admittime, dischtime,
                 deathtime, admission_type, diagnosis, source_file, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """,
        "sequence": "admissions_id_seq",
        "ncols": 11,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sqlite_batches(conn: sqlite3.Connection, query: str):
    """Yield batches of plain tuples from a SQLite SELECT query."""
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # Gracefully handle missing columns (old schema may not have all columns)
    try:
        cur.execute(query)
    except sqlite3.OperationalError as exc:
        logger.warning("SQLite query failed (%s) — skipping: %s", exc, query.split()[1])
        return
    while True:
        rows = cur.fetchmany(BATCH_SIZE)
        if not rows:
            break
        yield [tuple(row) for row in rows]


def _pad_row(row: tuple, ncols: int) -> tuple:
    """Pad a row with NULL to reach ncols (handles old schemas with fewer columns)."""
    if len(row) < ncols:
        return row + (None,) * (ncols - len(row))
    return row


def _reset_sequence(pg_cur, sequence: str, table: str) -> None:
    """Reset a PostgreSQL SERIAL/BIGSERIAL sequence to MAX(id) + 1."""
    pg_cur.execute(
        f"SELECT setval('{sequence}', COALESCE((SELECT MAX(id) FROM {table}), 1))"
    )


# ── Main migration logic ──────────────────────────────────────────────────────

def migrate(
    sqlite_path:  str,
    postgres_dsn: str,
    dry_run:      bool = False,
) -> dict[str, int]:
    """
    Migrate all rows from SQLite to PostgreSQL.

    Returns a dict of {table_name: rows_inserted}.
    """
    import psycopg2

    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    logger.info("SQLite source:    %s", sqlite_path)
    logger.info("PostgreSQL target: %s", postgres_dsn.split("@")[-1])  # hide credentials
    if dry_run:
        logger.info("Mode: DRY RUN — no rows will be inserted")

    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_conn     = psycopg2.connect(postgres_dsn)
    pg_conn.autocommit = False

    totals: dict[str, int] = {}

    try:
        for table_spec in _TABLES:
            name     = table_spec["name"]
            ncols    = table_spec["ncols"]
            sequence = table_spec["sequence"]

            # Count SQLite rows
            try:
                cur = sqlite_conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {name}")
                total = cur.fetchone()[0]
            except sqlite3.OperationalError:
                logger.info("%s: table not found in SQLite — skipping", name)
                totals[name] = 0
                continue

            if total == 0:
                logger.info("%s: 0 rows — skipping", name)
                totals[name] = 0
                continue

            logger.info("%s: %d row(s) to migrate", name, total)

            if dry_run:
                totals[name] = total
                continue

            inserted = 0
            pg_cur = pg_conn.cursor()

            for batch in _sqlite_batches(sqlite_conn, table_spec["select"]):
                padded = [_pad_row(row, ncols) for row in batch]
                pg_cur.executemany(table_spec["insert"], padded)
                inserted += len(padded)
                if inserted % 5000 == 0 or inserted == total:
                    logger.info(
                        "  %s: %d / %d row(s) inserted (%.0f%%)",
                        name, inserted, total, 100 * inserted / total,
                    )

            pg_conn.commit()

            # Reset the sequence so new application INSERTs don't collide with
            # the migrated IDs.
            _reset_sequence(pg_cur, sequence, name)
            pg_conn.commit()

            logger.info(
                "%s: migration done — %d row(s) inserted, sequence reset", name, inserted
            )
            totals[name] = inserted

    finally:
        sqlite_conn.close()
        pg_conn.close()

    return totals


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate structured medical data from SQLite → PostgreSQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sqlite-path",
        default=_DEFAULT_SQLITE,
        help=f"Path to the SQLite .db file (default: {_DEFAULT_SQLITE})",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=os.getenv("POSTGRES_DSN", ""),
        help="Full PostgreSQL DSN, e.g. postgresql://user:pass@host:5432/db "
             "(default: $POSTGRES_DSN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows in SQLite and report without inserting anything.",
    )
    args = parser.parse_args()

    if not args.postgres_dsn:
        print(
            "ERROR: supply --postgres-dsn or set the POSTGRES_DSN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    totals = migrate(
        sqlite_path=args.sqlite_path,
        postgres_dsn=args.postgres_dsn,
        dry_run=args.dry_run,
    )

    print("\n── Migration summary ─────────────────────────────────────────")
    for table, count in totals.items():
        tag = "(dry-run: would insert)" if args.dry_run else "row(s) inserted"
        print(f"  {table:<14} {count:>8}  {tag}")
    print("──────────────────────────────────────────────────────────────")

    if not args.dry_run:
        print(
            "\nVerify with:\n"
            "    psql $POSTGRES_DSN -c \\\n"
            "      \"SELECT 'patients', COUNT(*) FROM patients\n"
            "       UNION ALL SELECT 'providers', COUNT(*) FROM providers\n"
            "       UNION ALL SELECT 'records',   COUNT(*) FROM records\n"
            "       UNION ALL SELECT 'admissions',COUNT(*) FROM admissions;\""
        )


if __name__ == "__main__":
    main()
