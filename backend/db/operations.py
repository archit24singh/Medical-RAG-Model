"""
SQLite CRUD operations for patients, providers, and records.

Entity resolution
-----------------
The same patient may appear in many documents under slightly different names
("Alice Johnson", "Alice M. Johnson", "JOHNSON ALICE").  We normalise to
Title Case and then use difflib fuzzy matching to find existing rows before
creating new ones — avoiding duplicate patient / provider entries.

Verbatim quoting
----------------
Every record row stores `raw_text` — the exact chunk text from the source
document.  When answering a precise factual query the system quotes this
directly instead of letting the LLM paraphrase, eliminating hallucination.
"""

import difflib
import json
import logging
from typing import Optional

from db.schema import get_db

logger = logging.getLogger(__name__)

# Minimum similarity ratio (0-1) to consider two names the same entity.
_NAME_SIMILARITY_THRESHOLD = 0.82


# ── Patient operations ────────────────────────────────────────────────────────

def find_or_create_patient(
    name:         str,
    patient_id:   Optional[str] = None,
    dob:          Optional[str] = None,
    gender:       Optional[str] = None,
    phone:        Optional[str] = None,
    address:      Optional[str] = None,
    insurance_id: Optional[str] = None,
    source_file:  Optional[str] = None,
) -> int:
    """
    Return the id of an existing patient row that matches `name`, or insert
    a new row and return its id.

    Matching order:
      1. Exact match on patient_id (if provided)
      2. Fuzzy name match (>= 82% similarity) among existing rows
      3. Insert new row
    """
    name = _norm(name)
    if not name:
        raise ValueError("Patient name cannot be empty")

    with get_db() as conn:
        # 1. Exact patient_id match
        if patient_id:
            row = conn.execute(
                "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            if row:
                _update_patient(conn, row["id"], dob, gender, phone, address,
                                insurance_id, source_file)
                return row["id"]

        # 2. Fuzzy name match
        existing = conn.execute("SELECT id, name FROM patients").fetchall()
        best_id, best_score = _best_name_match(name, existing)
        if best_id:
            _update_patient(conn, best_id, dob, gender, phone, address,
                            insurance_id, source_file)
            return best_id

        # 3. Insert new patient
        cur = conn.execute(
            """INSERT INTO patients
               (name, patient_id, dob, gender, phone, address, insurance_id, source_files)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, patient_id, dob, gender, phone, address, insurance_id,
             source_file or ""),
        )
        logger.info("New patient created: %s (id=%d)", name, cur.lastrowid)
        return cur.lastrowid


def _update_patient(conn, patient_id_pk, dob, gender, phone, address,
                    insurance_id, source_file):
    """Fill in any null columns for an existing patient."""
    if dob:
        conn.execute("UPDATE patients SET dob=? WHERE id=? AND dob IS NULL",
                     (dob, patient_id_pk))
    if gender:
        conn.execute("UPDATE patients SET gender=? WHERE id=? AND gender IS NULL",
                     (gender, patient_id_pk))
    if phone:
        conn.execute("UPDATE patients SET phone=? WHERE id=? AND phone IS NULL",
                     (phone, patient_id_pk))
    if address:
        conn.execute("UPDATE patients SET address=? WHERE id=? AND address IS NULL",
                     (address, patient_id_pk))
    if insurance_id:
        conn.execute("UPDATE patients SET insurance_id=? WHERE id=? AND insurance_id IS NULL",
                     (insurance_id, patient_id_pk))
    if source_file:
        # Append to source_files list
        row = conn.execute("SELECT source_files FROM patients WHERE id=?",
                           (patient_id_pk,)).fetchone()
        existing = row["source_files"] or ""
        files = [f for f in existing.split(",") if f]
        if source_file not in files:
            files.append(source_file)
        conn.execute("UPDATE patients SET source_files=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (",".join(files), patient_id_pk))


# ── Provider operations ───────────────────────────────────────────────────────

def find_or_create_provider(
    name:           Optional[str] = None,
    npi:            Optional[str] = None,
    specialty:      Optional[str] = None,
    dob:            Optional[str] = None,
    phone:          Optional[str] = None,
    address:        Optional[str] = None,
    license_number: Optional[str] = None,
    source_file:    Optional[str] = None,
) -> Optional[int]:
    """
    Return provider id, creating a new row if needed.
    NPI is the primary unique key; name fuzzy-match is the fallback.
    Returns None if neither name nor NPI is given.
    """
    name = _norm(name) if name else None
    npi  = (npi or "").strip() or None

    if not name and not npi:
        return None

    with get_db() as conn:
        # 1. Exact NPI match
        if npi:
            row = conn.execute(
                "SELECT id FROM providers WHERE npi = ?", (npi,)
            ).fetchone()
            if row:
                _update_provider(conn, row["id"], name, specialty, dob,
                                 phone, address, license_number, source_file)
                return row["id"]

        # 2. Fuzzy name match
        if name:
            existing = conn.execute("SELECT id, name FROM providers").fetchall()
            best_id, _ = _best_name_match(name, existing)
            if best_id:
                _update_provider(conn, best_id, name, specialty, dob,
                                 phone, address, license_number, source_file)
                return best_id

        # 3. Insert new provider
        cur = conn.execute(
            """INSERT INTO providers
               (name, npi, specialty, dob, phone, address, license_number, source_files)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, npi, specialty, dob, phone, address, license_number,
             source_file or ""),
        )
        logger.info("New provider created: %s / NPI=%s (id=%d)",
                    name, npi, cur.lastrowid)
        return cur.lastrowid


def _update_provider(conn, provider_pk, name, specialty, dob, phone,
                     address, license_number, source_file):
    """Fill in null columns for an existing provider."""
    for col, val in [("name", name), ("specialty", specialty), ("dob", dob),
                     ("phone", phone), ("address", address),
                     ("license_number", license_number)]:
        if val:
            conn.execute(f"UPDATE providers SET {col}=? WHERE id=? AND {col} IS NULL",
                         (val, provider_pk))
    if source_file:
        row = conn.execute("SELECT source_files FROM providers WHERE id=?",
                           (provider_pk,)).fetchone()
        existing = row["source_files"] or ""
        files = [f for f in existing.split(",") if f]
        if source_file not in files:
            files.append(source_file)
        conn.execute("UPDATE providers SET source_files=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                     (",".join(files), provider_pk))


# ── Record operations ─────────────────────────────────────────────────────────

def insert_record(
    patient_id:    int,
    record_type:   str,
    raw_text:      str,
    source_file:   str,
    provider_id:   Optional[int] = None,
    record_date:   Optional[str] = None,
    total_amount:  Optional[str] = None,
    diagnosis:     Optional[str] = None,
    treatment:     Optional[str] = None,
    medication:    Optional[str] = None,
    dosage:        Optional[str] = None,
    test_name:     Optional[str] = None,
    test_result:   Optional[str] = None,
    reference_range: Optional[str] = None,
    details:       Optional[dict] = None,
    page_number:   Optional[str] = None,
    chunk_index:   Optional[str] = None,
) -> int:
    """Insert a clinical record and return its id."""
    details_json = json.dumps(details) if details else None
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO records
               (patient_id, provider_id, record_type, record_date,
                total_amount, diagnosis, treatment, medication, dosage,
                test_name, test_result, reference_range,
                details_json, raw_text, source_file, page_number, chunk_index)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (patient_id, provider_id, record_type, record_date,
             total_amount, diagnosis, treatment, medication, dosage,
             test_name, test_result, reference_range,
             details_json, raw_text, source_file, page_number, chunk_index),
        )
        return cur.lastrowid


# ── Query operations ──────────────────────────────────────────────────────────

def query_patient_records(
    patient_name:  Optional[str] = None,
    patient_id_str: Optional[str] = None,
    record_type:   Optional[str] = None,
    record_date:   Optional[str] = None,
    limit:         int = 20,
) -> list[dict]:
    """
    Return records for a patient, optionally filtered by type and date.
    Uses fuzzy name matching so minor variations are handled.
    """
    with get_db() as conn:
        # Resolve patient row(s)
        patient_pks = _resolve_patient_pks(conn, patient_name, patient_id_str)
        if not patient_pks:
            return []

        placeholders = ",".join("?" * len(patient_pks))
        params: list = list(patient_pks)

        sql = f"""
            SELECT r.*, p.name AS patient_name, p.dob AS patient_dob,
                   p.patient_id AS patient_mrn,
                   pv.name AS provider_name, pv.npi AS provider_npi
            FROM records r
            JOIN patients p ON r.patient_id = p.id
            LEFT JOIN providers pv ON r.provider_id = pv.id
            WHERE r.patient_id IN ({placeholders})
        """

        if record_type:
            sql += " AND r.record_type = ?"
            params.append(record_type)

        if record_date:
            sql += " AND r.record_date = ?"
            params.append(record_date)

        sql += " ORDER BY r.record_date DESC, r.id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def query_provider(
    provider_name: Optional[str] = None,
    provider_npi:  Optional[str] = None,
) -> list[dict]:
    """Return provider rows matching name (fuzzy) or exact NPI."""
    with get_db() as conn:
        if provider_npi:
            rows = conn.execute(
                "SELECT * FROM providers WHERE npi = ?", (provider_npi,)
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]

        if provider_name:
            norm = _norm(provider_name)
            all_rows = conn.execute("SELECT * FROM providers").fetchall()
            scored = []
            for row in all_rows:
                score = difflib.SequenceMatcher(
                    None, norm.lower(), (row["name"] or "").lower()
                ).ratio()
                if score >= _NAME_SIMILARITY_THRESHOLD:
                    scored.append((score, dict(row)))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored]

    return []


def query_all_patient_records(
    patient_name:   Optional[str] = None,
    patient_id_str: Optional[str] = None,
    limit:          int = 50,
) -> list[dict]:
    """Return all records for a patient across all record types."""
    return query_patient_records(
        patient_name=patient_name,
        patient_id_str=patient_id_str,
        limit=limit,
    )


def get_patient_info(
    patient_name:   Optional[str] = None,
    patient_id_str: Optional[str] = None,
) -> Optional[dict]:
    """Return the patients row for a given name / ID."""
    with get_db() as conn:
        pks = _resolve_patient_pks(conn, patient_name, patient_id_str)
        if not pks:
            return None
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (pks[0],)
        ).fetchone()
        return dict(row) if row else None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _norm(name: Optional[str]) -> str:
    """Normalise a name: strip whitespace, Title Case."""
    if not name:
        return ""
    return " ".join(name.strip().title().split())


def _best_name_match(
    name: str,
    existing_rows,  # list of sqlite3.Row with (id, name)
) -> tuple[Optional[int], float]:
    """
    Find the best fuzzy name match among existing rows.
    Returns (row_id, score) or (None, 0.0) if nothing is above threshold.
    """
    best_id, best_score = None, 0.0
    for row in existing_rows:
        score = difflib.SequenceMatcher(
            None, name.lower(), (row["name"] or "").lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_id = row["id"]

    if best_score >= _NAME_SIMILARITY_THRESHOLD:
        return best_id, best_score
    return None, 0.0


def _resolve_patient_pks(
    conn,
    name:       Optional[str],
    patient_id: Optional[str],
) -> list[int]:
    """Return a list of patient primary keys matching name and/or patient_id."""
    pks = set()

    if patient_id:
        rows = conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchall()
        pks.update(r["id"] for r in rows)

    if name:
        norm = _norm(name)
        all_rows = conn.execute("SELECT id, name FROM patients").fetchall()
        for row in all_rows:
            score = difflib.SequenceMatcher(
                None, norm.lower(), (row["name"] or "").lower()
            ).ratio()
            if score >= _NAME_SIMILARITY_THRESHOLD:
                pks.add(row["id"])

    return list(pks)
