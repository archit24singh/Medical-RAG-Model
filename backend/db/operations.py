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
import re
from typing import Optional

from db.schema import get_db

logger = logging.getLogger(__name__)

# Minimum similarity ratio (0-1) to consider two names the same entity.
_NAME_SIMILARITY_THRESHOLD = 0.82


# ── Name matching helpers ───────────────────────────────────────────────────
#
# Names arrive in inconsistent orderings depending on the source document and
# how the LLM extracted them, e.g.:
#   "Hill, Susan L"   (spreadsheet "Last, First M" format)
#   "Susan L Hill"    (LLM's default "First Last" Title Case)
#   "Susan L"         (user query, missing last name)
#
# difflib.SequenceMatcher is order-sensitive, so "Hill, Susan L" vs
# "Susan L Hill" scores well below the similarity threshold even though they
# refer to the same person. To handle this we compare names as an
# order-independent set of tokens (with punctuation stripped) in addition to
# the plain string ratio.

def _name_tokens(name: str) -> set[str]:
    """Tokenize a name into a set of lowercase tokens, stripping punctuation."""
    if not name:
        return set()
    cleaned = re.sub(r"[.,]", " ", name.lower())
    return {t for t in cleaned.split() if t}


def _canonical_name(name: str) -> str:
    """
    Canonical form of a name: lowercase tokens, punctuation stripped, sorted.
    Produces the same string for "Hill, Susan L" and "Susan L Hill".
    """
    return " ".join(sorted(_name_tokens(name)))


def _name_match_score(query: str, candidate: str) -> float:
    """
    Similarity score (0-1) between two names that is robust to token
    reordering ("Last, First M" vs "First M Last") and partial names
    (e.g. "Susan L" vs "Hill, Susan L").

    Returns 1.0 if one name's token set is a non-empty subset of the
    other's (handles partial / reordered names), otherwise falls back to
    a SequenceMatcher ratio on the canonical (sorted-token) forms.
    """
    query_tokens = _name_tokens(query)
    candidate_tokens = _name_tokens(candidate)

    if not query_tokens or not candidate_tokens:
        return 0.0

    # Require at least 2 tokens on the (shorter) side before allowing a
    # subset match — a single shared token (e.g. just "Susan") is too
    # ambiguous to treat as a confident match.
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
    Return the id of an existing patient row that matches `name`, or insert
    a new row and return its id.

    Matching order:
      1. Exact match on subject_id (if provided) — MIMIC-III SUBJECT_ID
      2. Exact match on patient_id (if provided)
      3. Fuzzy name match (>= 82% similarity) among existing rows
      4. Insert new row
    """
    name = _norm(name)
    if not name:
        raise ValueError("Patient name cannot be empty")

    with get_db() as conn:
        # 1. Exact subject_id match (MIMIC-III)
        if subject_id:
            row = conn.execute(
                "SELECT id FROM patients WHERE subject_id = ?", (subject_id,)
            ).fetchone()
            if row:
                _update_patient(conn, row["id"], dob, gender, phone, address,
                                insurance_id, source_file, subject_id)
                return row["id"]

        # 2. Exact patient_id match
        if patient_id:
            row = conn.execute(
                "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            if row:
                _update_patient(conn, row["id"], dob, gender, phone, address,
                                insurance_id, source_file, subject_id)
                return row["id"]

        # 3. Fuzzy name match
        existing = conn.execute("SELECT id, name FROM patients").fetchall()
        best_id, best_score = _best_name_match(name, existing)
        if best_id:
            _update_patient(conn, best_id, dob, gender, phone, address,
                            insurance_id, source_file, subject_id)
            return best_id

        # 4. Insert new patient
        cur = conn.execute(
            """INSERT INTO patients
               (name, patient_id, dob, gender, phone, address, insurance_id,
                source_files, subject_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, patient_id, dob, gender, phone, address, insurance_id,
             source_file or "", subject_id),
        )
        logger.info("New patient created: %s (id=%d)", name, cur.lastrowid)
        return cur.lastrowid


def _update_patient(conn, patient_id_pk, dob, gender, phone, address,
                    insurance_id, source_file, subject_id=None):
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
    if subject_id:
        conn.execute("UPDATE patients SET subject_id=? WHERE id=? AND subject_id IS NULL",
                     (subject_id, patient_id_pk))
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
    claim_number:  Optional[str] = None,
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
    hadm_id:          Optional[str] = None,
    icd9_code:        Optional[str] = None,
    icd9_description: Optional[str] = None,
    seq_num:          Optional[str] = None,
) -> int:
    """Insert a clinical record and return its id."""
    details_json = json.dumps(details) if details else None
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO records
               (patient_id, provider_id, record_type, record_date,
                total_amount, claim_number, diagnosis, treatment, medication, dosage,
                test_name, test_result, reference_range,
                details_json, raw_text, source_file, page_number, chunk_index,
                hadm_id, icd9_code, icd9_description, seq_num)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (patient_id, provider_id, record_type, record_date,
             total_amount, claim_number, diagnosis, treatment, medication, dosage,
             test_name, test_result, reference_range,
             details_json, raw_text, source_file, page_number, chunk_index,
             hadm_id, icd9_code, icd9_description, seq_num),
        )
        return cur.lastrowid


# ── Admission operations (MIMIC-III) ─────────────────────────────────────────

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
    """
    Return the id of an existing admissions row for `hadm_id`, or insert a
    new row and return its id. hadm_id is unique, so this is idempotent
    across re-ingests.
    """
    if not hadm_id:
        raise ValueError("hadm_id cannot be empty")

    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM admissions WHERE hadm_id = ?", (hadm_id,)
        ).fetchone()
        if row:
            # Fill in any null columns for an existing admission
            for col, val in [("patient_id", patient_pk), ("subject_id", subject_id),
                              ("admittime", admittime), ("dischtime", dischtime),
                              ("deathtime", deathtime),
                              ("admission_type", admission_type), ("diagnosis", diagnosis)]:
                if val is not None:
                    conn.execute(f"UPDATE admissions SET {col}=? WHERE id=? AND {col} IS NULL",
                                 (val, row["id"]))
            return row["id"]

        cur = conn.execute(
            """INSERT INTO admissions
               (patient_id, subject_id, hadm_id, admittime, dischtime, deathtime,
                admission_type, diagnosis, source_file)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient_pk, subject_id, hadm_id, admittime, dischtime, deathtime,
             admission_type, diagnosis, source_file or ""),
        )
        logger.info("New admission created: hadm_id=%s (id=%d)", hadm_id, cur.lastrowid)
        return cur.lastrowid


def get_admission_info(hadm_id: str) -> Optional[dict]:
    """Return an admissions row for a given hadm_id, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM admissions WHERE hadm_id = ?", (hadm_id,)
        ).fetchone()
        return dict(row) if row else None


def query_records_by_hadm_id(hadm_id: str, limit: int = 30) -> list[dict]:
    """Return all `records` rows (e.g. diagnoses) tied to a given hadm_id."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT r.*, p.name AS patient_name, p.dob AS patient_dob,
                   p.patient_id AS patient_mrn, p.subject_id AS patient_subject_id,
                   pv.name AS provider_name, pv.npi AS provider_npi
            FROM records r
            JOIN patients p ON r.patient_id = p.id
            LEFT JOIN providers pv ON r.provider_id = pv.id
            WHERE r.hadm_id = ?
            ORDER BY CAST(r.seq_num AS INTEGER) ASC, r.id ASC
            LIMIT ?
            """,
            (hadm_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Maintenance operations ────────────────────────────────────────────────────

def delete_records_by_source_file(source_file: str) -> int:
    """
    Delete all `records` rows previously written from `source_file`.

    Why: ingestion (tabular and unstructured) can run repeatedly for the same
    file — e.g. AUTO_INGEST re-runs ingest_directory() on every container
    restart. Without this, each re-ingest APPENDS a fresh copy of every row
    on top of whatever was already there, so the same patient ends up with
    duplicate records — some from an older code version (missing fields like
    claim_number) and some from the latest version (fully populated). Since
    queries are ORDER BY date DESC, id DESC LIMIT N, the results end up an
    interleaved mix of old/incomplete and new/complete rows for the same
    patient — which looks like "some fields show up, some don't."

    Calling this at the start of re-ingesting a file makes ingestion
    idempotent: old rows for that file are cleared out before the fresh
    ones are written, so there's only ever one (current) copy per file.

    Returns the number of rows deleted.
    """
    with get_db() as conn:
        cur = conn.execute("DELETE FROM records WHERE source_file = ?", (source_file,))
        return cur.rowcount


# ── Query operations ──────────────────────────────────────────────────────────

def query_patient_records(
    patient_name:  Optional[str] = None,
    patient_id_str: Optional[str] = None,
    record_type:   Optional[str] = None,
    record_date:   Optional[str] = None,
    limit:         int = 20,
    subject_id:    Optional[str] = None,
) -> list[dict]:
    """
    Return records for a patient, optionally filtered by type and date.
    Uses fuzzy name matching so minor variations are handled.
    """
    with get_db() as conn:
        # Resolve patient row(s)
        patient_pks = _resolve_patient_pks(conn, patient_name, patient_id_str, subject_id)
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
                score = _name_match_score(norm, row["name"] or "")
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
    subject_id:     Optional[str] = None,
) -> Optional[dict]:
    """
    Return a patients row for a given name / ID.

    If a name matches multiple fragmented patient rows (e.g. the same
    person was ingested under slightly different name formats before the
    matching logic was improved), merge the non-null demographic fields
    across all matches so a field stored on one row (e.g. address) is
    still surfaced even if a different row is the "primary" match.
    """
    with get_db() as conn:
        pks = _resolve_patient_pks(conn, patient_name, patient_id_str, subject_id)
        if not pks:
            return None

        merged: Optional[dict] = None
        for pk in pks:
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (pk,)
            ).fetchone()
            if not row:
                continue
            d = dict(row)
            if merged is None:
                merged = d
            else:
                for k, v in d.items():
                    if v not in (None, "", "Unknown") and not merged.get(k):
                        merged[k] = v
        return merged


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
        score = _name_match_score(name, row["name"] or "")
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
    """Return a list of patient primary keys matching name, patient_id, and/or subject_id."""
    pks = set()

    if subject_id:
        rows = conn.execute(
            "SELECT id FROM patients WHERE subject_id = ?", (subject_id,)
        ).fetchall()
        pks.update(r["id"] for r in rows)

    if patient_id:
        rows = conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchall()
        pks.update(r["id"] for r in rows)

    if name:
        norm = _norm(name)
        all_rows = conn.execute("SELECT id, name FROM patients").fetchall()
        for row in all_rows:
            score = _name_match_score(norm, row["name"] or "")
            if score >= _NAME_SIMILARITY_THRESHOLD:
                pks.add(row["id"])

    return list(pks)
