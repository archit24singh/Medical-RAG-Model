"""
PostgreSQL CRUD operations for patients, providers, records, and admissions.

Entity resolution
-----------------
The same patient may appear under slightly different names across documents
("Alice Johnson", "Alice M. Johnson", "JOHNSON ALICE"). We normalise to
Title Case and use difflib fuzzy matching to find existing rows before
creating new ones — avoiding duplicate patient/provider entries.

Verbatim quoting
----------------
Every record row stores `raw_text` — the exact chunk text from the source
document. Precise factual queries quote this directly instead of letting the
LLM paraphrase, eliminating hallucination on the SQL path.

Placeholder syntax
------------------
psycopg2 uses %s (not ?) for all parameter placeholders. All queries here
use %s — never string-format the SQL with user data.
"""

import difflib
import json
import logging
import re
from typing import Optional

from db.schema import get_db, fetchall_dicts, fetchone_dict

logger = logging.getLogger(__name__)

# Minimum similarity ratio (0-1) to consider two names the same entity.
_NAME_SIMILARITY_THRESHOLD = 0.82


# ── Name matching helpers ─────────────────────────────────────────────────────

def _name_tokens(name: str) -> set[str]:
    """Tokenize a name into a set of lowercase tokens, stripping punctuation."""
    if not name:
        return set()
    cleaned = re.sub(r"[.,]", " ", name.lower())
    return {t for t in cleaned.split() if t}


def _canonical_name(name: str) -> str:
    """Canonical form: lowercase tokens, punctuation stripped, sorted."""
    return " ".join(sorted(_name_tokens(name)))


def _name_match_score(query: str, candidate: str) -> float:
    """
    Similarity score (0-1) robust to token reordering and partial names.
    Returns 1.0 if one token set is a non-empty subset of the other
    (handles "Last, First M" vs "First M Last"), otherwise SequenceMatcher
    on canonical forms.
    """
    query_tokens = _name_tokens(query)
    candidate_tokens = _name_tokens(candidate)

    if not query_tokens or not candidate_tokens:
        return 0.0

    smaller, larger = (
        (query_tokens, candidate_tokens)
        if len(query_tokens) <= len(candidate_tokens)
        else (candidate_tokens, query_tokens)
    )
    if len(smaller) >= 2 and smaller <= larger:
        return 1.0

    return difflib.SequenceMatcher(
        None, _canonical_name(query), _canonical_name(candidate)
    ).ratio()


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
    subject_id:   Optional[str] = None,
) -> int:
    """
    Return the id of an existing patient row or insert a new one.

    Matching order:
      1. Exact subject_id match (MIMIC-III)
      2. Exact patient_id match
      3. Fuzzy name match (>= 82% similarity)
      4. Insert new row
    """
    name = _norm(name)
    if not name:
        raise ValueError("Patient name cannot be empty")

    with get_db() as conn:
        with conn.cursor() as cur:
            # 1. subject_id
            if subject_id:
                cur.execute("SELECT id FROM patients WHERE subject_id = %s", (subject_id,))
                row = fetchone_dict(cur)
                if row:
                    _update_patient(conn, row["id"], dob, gender, phone, address,
                                    insurance_id, source_file, subject_id)
                    return row["id"]

            # 2. patient_id
            if patient_id:
                cur.execute("SELECT id FROM patients WHERE patient_id = %s", (patient_id,))
                row = fetchone_dict(cur)
                if row:
                    _update_patient(conn, row["id"], dob, gender, phone, address,
                                    insurance_id, source_file, subject_id)
                    return row["id"]

            # 3. fuzzy name
            cur.execute("SELECT id, name FROM patients")
            existing = fetchall_dicts(cur)
            best_id, best_score = _best_name_match(name, existing)
            if best_id:
                _update_patient(conn, best_id, dob, gender, phone, address,
                                insurance_id, source_file, subject_id)
                return best_id

            # 4. insert
            cur.execute(
                """INSERT INTO patients
                   (name, patient_id, dob, gender, phone, address, insurance_id,
                    source_files, subject_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (name, patient_id, dob, gender, phone, address, insurance_id,
                 source_file or "", subject_id),
            )
            new_id = cur.fetchone()[0]
            logger.info("New patient created: %s (id=%d)", name, new_id)
            return new_id


def _update_patient(conn, patient_pk, dob, gender, phone, address,
                    insurance_id, source_file, subject_id=None):
    """Fill in any NULL columns for an existing patient."""
    with conn.cursor() as cur:
        for col, val in [("dob", dob), ("gender", gender), ("phone", phone),
                         ("address", address), ("insurance_id", insurance_id),
                         ("subject_id", subject_id)]:
            if val:
                cur.execute(
                    f"UPDATE patients SET {col}=%s WHERE id=%s AND {col} IS NULL",
                    (val, patient_pk)
                )
        if source_file:
            cur.execute("SELECT source_files FROM patients WHERE id=%s", (patient_pk,))
            row = cur.fetchone()
            existing = (row[0] or "") if row else ""
            files = [f for f in existing.split(",") if f]
            if source_file not in files:
                files.append(source_file)
            cur.execute(
                "UPDATE patients SET source_files=%s, updated_at=NOW() WHERE id=%s",
                (",".join(files), patient_pk)
            )


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
    """Return provider id, creating a new row if needed. Returns None if no name/NPI."""
    name = _norm(name) if name else None
    npi  = (npi or "").strip() or None

    if not name and not npi:
        return None

    with get_db() as conn:
        with conn.cursor() as cur:
            # 1. exact NPI
            if npi:
                cur.execute("SELECT id FROM providers WHERE npi = %s", (npi,))
                row = fetchone_dict(cur)
                if row:
                    _update_provider(conn, row["id"], name, specialty, dob,
                                     phone, address, license_number, source_file)
                    return row["id"]

            # 2. fuzzy name
            if name:
                cur.execute("SELECT id, name FROM providers")
                existing = fetchall_dicts(cur)
                best_id, _ = _best_name_match(name, existing)
                if best_id:
                    _update_provider(conn, best_id, name, specialty, dob,
                                     phone, address, license_number, source_file)
                    return best_id

            # 3. insert
            cur.execute(
                """INSERT INTO providers
                   (name, npi, specialty, dob, phone, address, license_number, source_files)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (name, npi, specialty, dob, phone, address, license_number,
                 source_file or ""),
            )
            new_id = cur.fetchone()[0]
            logger.info("New provider created: %s / NPI=%s (id=%d)", name, npi, new_id)
            return new_id


def _update_provider(conn, provider_pk, name, specialty, dob, phone,
                     address, license_number, source_file):
    """Fill in NULL columns for an existing provider."""
    with conn.cursor() as cur:
        for col, val in [("name", name), ("specialty", specialty), ("dob", dob),
                         ("phone", phone), ("address", address),
                         ("license_number", license_number)]:
            if val:
                cur.execute(
                    f"UPDATE providers SET {col}=%s WHERE id=%s AND {col} IS NULL",
                    (val, provider_pk)
                )
        if source_file:
            cur.execute("SELECT source_files FROM providers WHERE id=%s", (provider_pk,))
            row = cur.fetchone()
            existing = (row[0] or "") if row else ""
            files = [f for f in existing.split(",") if f]
            if source_file not in files:
                files.append(source_file)
            cur.execute(
                "UPDATE providers SET source_files=%s, updated_at=NOW() WHERE id=%s",
                (",".join(files), provider_pk)
            )


# ── Record operations ─────────────────────────────────────────────────────────

def insert_record(
    patient_id:       int,
    record_type:      str,
    raw_text:         str,
    source_file:      str,
    provider_id:      Optional[int] = None,
    record_date:      Optional[str] = None,
    total_amount:     Optional[str] = None,
    claim_number:     Optional[str] = None,
    diagnosis:        Optional[str] = None,
    treatment:        Optional[str] = None,
    medication:       Optional[str] = None,
    dosage:           Optional[str] = None,
    test_name:        Optional[str] = None,
    test_result:      Optional[str] = None,
    reference_range:  Optional[str] = None,
    details:          Optional[dict] = None,
    page_number:      Optional[str] = None,
    chunk_index:      Optional[str] = None,
    hadm_id:          Optional[str] = None,
    icd9_code:        Optional[str] = None,
    icd9_description: Optional[str] = None,
    seq_num:          Optional[str] = None,
) -> int:
    """Insert a clinical record and return its id."""
    details_json = json.dumps(details) if details else None
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO records
                   (patient_id, provider_id, record_type, record_date,
                    total_amount, claim_number, diagnosis, treatment,
                    medication, dosage, test_name, test_result, reference_range,
                    details_json, raw_text, source_file, page_number, chunk_index,
                    hadm_id, icd9_code, icd9_description, seq_num)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id""",
                (patient_id, provider_id, record_type, record_date,
                 total_amount, claim_number, diagnosis, treatment,
                 medication, dosage, test_name, test_result, reference_range,
                 details_json, raw_text, source_file, page_number, chunk_index,
                 hadm_id, icd9_code, icd9_description, seq_num),
            )
            return cur.fetchone()[0]


# ── Admission operations ──────────────────────────────────────────────────────

def find_or_create_admission(
    patient_pk:     Optional[int],
    subject_id:     Optional[str],
    hadm_id:        str,
    admittime:      Optional[str] = None,
    dischtime:      Optional[str] = None,
    deathtime:      Optional[str] = None,
    admission_type: Optional[str] = None,
    diagnosis:      Optional[str] = None,
    source_file:    Optional[str] = None,
) -> int:
    """Return id of an existing admissions row for hadm_id, or insert a new one."""
    if not hadm_id:
        raise ValueError("hadm_id cannot be empty")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM admissions WHERE hadm_id = %s", (hadm_id,))
            row = fetchone_dict(cur)
            if row:
                for col, val in [("patient_id", patient_pk), ("subject_id", subject_id),
                                  ("admittime", admittime), ("dischtime", dischtime),
                                  ("deathtime", deathtime),
                                  ("admission_type", admission_type),
                                  ("diagnosis", diagnosis)]:
                    if val is not None:
                        cur.execute(
                            f"UPDATE admissions SET {col}=%s WHERE id=%s AND {col} IS NULL",
                            (val, row["id"])
                        )
                return row["id"]

            cur.execute(
                """INSERT INTO admissions
                   (patient_id, subject_id, hadm_id, admittime, dischtime,
                    deathtime, admission_type, diagnosis, source_file)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id""",
                (patient_pk, subject_id, hadm_id, admittime, dischtime,
                 deathtime, admission_type, diagnosis, source_file or ""),
            )
            new_id = cur.fetchone()[0]
            logger.info("New admission created: hadm_id=%s (id=%d)", hadm_id, new_id)
            return new_id


def get_admission_info(hadm_id: str) -> Optional[dict]:
    """Return an admissions row for a given hadm_id, or None."""
    from db.schema import get_readonly_db
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM admissions WHERE hadm_id = %s", (hadm_id,))
            return fetchone_dict(cur)


def query_records_by_hadm_id(hadm_id: str, limit: int = 30) -> list[dict]:
    """Return all records rows tied to a given hadm_id."""
    from db.schema import get_readonly_db
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*, p.name AS patient_name, p.dob AS patient_dob,
                       p.patient_id AS patient_mrn,
                       p.subject_id AS patient_subject_id,
                       pv.name AS provider_name, pv.npi AS provider_npi
                FROM records r
                JOIN patients p ON r.patient_id = p.id
                LEFT JOIN providers pv ON r.provider_id = pv.id
                WHERE r.hadm_id = %s
                ORDER BY
                    CASE WHEN r.seq_num ~ '^[0-9]+$' THEN r.seq_num::int END ASC,
                    r.id ASC
                LIMIT %s
                """,
                (hadm_id, limit),
            )
            return fetchall_dicts(cur)


# ── Maintenance ───────────────────────────────────────────────────────────────

def delete_records_by_source_file(source_file: str) -> int:
    """
    Delete all records rows for source_file before re-ingesting.
    Makes ingestion idempotent: re-ingest always produces exactly one
    current copy of each row, with no stale duplicates from older runs.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM records WHERE source_file = %s", (source_file,))
            return cur.rowcount


# ── Query operations ──────────────────────────────────────────────────────────

def query_patient_records(
    patient_name:   Optional[str] = None,
    patient_id_str: Optional[str] = None,
    record_type:    Optional[str] = None,
    record_date:    Optional[str] = None,
    limit:          int = 20,
    subject_id:     Optional[str] = None,
) -> list[dict]:
    """Return records for a patient, optionally filtered by type and date."""
    from db.schema import get_readonly_db
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            patient_pks = _resolve_patient_pks(conn, patient_name, patient_id_str, subject_id)
            if not patient_pks:
                return []

            placeholders = ",".join(["%s"] * len(patient_pks))
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
                sql += " AND r.record_type = %s"
                params.append(record_type)
            if record_date:
                sql += " AND r.record_date = %s"
                params.append(record_date)
            sql += " ORDER BY r.record_date DESC, r.id DESC LIMIT %s"
            params.append(limit)

            cur.execute(sql, params)
            return fetchall_dicts(cur)


def query_provider(
    provider_name: Optional[str] = None,
    provider_npi:  Optional[str] = None,
) -> list[dict]:
    """Return provider rows matching name (fuzzy) or exact NPI."""
    from db.schema import get_readonly_db
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            if provider_npi:
                cur.execute("SELECT * FROM providers WHERE npi = %s", (provider_npi,))
                rows = fetchall_dicts(cur)
                if rows:
                    return rows

            if provider_name:
                norm = _norm(provider_name)
                cur.execute("SELECT * FROM providers")
                all_rows = fetchall_dicts(cur)
                scored = []
                for row in all_rows:
                    score = _name_match_score(norm, row.get("name") or "")
                    if score >= _NAME_SIMILARITY_THRESHOLD:
                        scored.append((score, row))
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
    subject_id:     Optional[str] = None,
) -> Optional[dict]:
    """Return a patients row, merging non-null fields across fuzzy duplicate matches."""
    from db.schema import get_readonly_db
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            pks = _resolve_patient_pks(conn, patient_name, patient_id_str, subject_id)
            if not pks:
                return None

            merged: Optional[dict] = None
            for pk in pks:
                cur.execute("SELECT * FROM patients WHERE id = %s", (pk,))
                row = fetchone_dict(cur)
                if not row:
                    continue
                if merged is None:
                    merged = row
                else:
                    for k, v in row.items():
                        if v not in (None, "", "Unknown") and not merged.get(k):
                            merged[k] = v
            return merged


# ── Internal helpers ──────────────────────────────────────────────────────────

def _norm(name: Optional[str]) -> str:
    if not name:
        return ""
    return " ".join(name.strip().title().split())


def _best_name_match(
    name: str,
    existing_rows: list[dict],
) -> tuple[Optional[int], float]:
    best_id, best_score = None, 0.0
    for row in existing_rows:
        score = _name_match_score(name, row.get("name") or "")
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
    subject_id: Optional[str] = None,
) -> list[int]:
    """Return list of patient primary keys matching name, patient_id, and/or subject_id."""
    pks: set[int] = set()
    with conn.cursor() as cur:
        if subject_id:
            cur.execute("SELECT id FROM patients WHERE subject_id = %s", (subject_id,))
            pks.update(r[0] for r in cur.fetchall())

        if patient_id:
            cur.execute("SELECT id FROM patients WHERE patient_id = %s", (patient_id,))
            pks.update(r[0] for r in cur.fetchall())

        if name:
            norm = _norm(name)
            cur.execute("SELECT id, name FROM patients")
            for row in cur.fetchall():
                score = _name_match_score(norm, row[1] or "")
                if score >= _NAME_SIMILARITY_THRESHOLD:
                    pks.add(row[0])

    return list(pks)
