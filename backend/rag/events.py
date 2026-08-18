"""
Canonical event adapter (Phase 1) — the dataset-agnostic foundation for the
ETHER / AIR / DER pillars.

Every source schema (Campbell claims, MIMIC-IV, EHRSHOT, …) is mapped into one
canonical shape:

    Event(patient_key, concept, concept_type, value, event_time, source_file, raw_text)

The mapping is driven entirely by ``data/concept_map.yaml`` — adding a new
dataset is a config edit (new column synonyms), never a code change. The
pillars downstream only ever see canonical concepts, so they stay portable.

Data source
-----------
Events are built FROM the existing schema-on-read staging tables
(``stg_<sig>``) rather than by intercepting the ingest path. This means it
works on data that is *already ingested* (e.g. the Campbell file) with no
re-ingestion, and stays decoupled from the (complex) tabular ingest pipeline.

Public API
----------
  load_concept_map()                      → dict (cached)
  extract_events_from_row(row, src, …)    → list[Event]
  backfill_events(limit=None)             → int   (populate events table)
  numeric_trajectories(patient_key)       → {concept: [(value, date), …]}
  textual_events(patient_key)             → list[Event]
  list_patient_keys()                     → list[str]
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import yaml

from config import settings
from db.schema import get_db, get_readonly_db, fetchall_dicts

logger = logging.getLogger(__name__)

# ── Canonical event model ─────────────────────────────────────────────────────


@dataclass
class Event:
    patient_key: Optional[str]
    concept: str
    concept_type: str            # 'numeric' | 'textual'
    value_num: Optional[float]
    value_text: Optional[str]
    event_time: Optional[date]
    event_time_raw: Optional[str]
    source_file: Optional[str]
    raw_text: str
    patient_name: Optional[str] = None

    def row_hash(self) -> str:
        h = hashlib.sha256()
        for part in (
            self.patient_key, self.patient_name, self.concept, self.concept_type,
            str(self.value_num), self.value_text,
            self.event_time_raw, self.source_file,
        ):
            h.update((part or "").encode())
            h.update(b"\x00")
        return h.hexdigest()


# ── Concept-map loading ───────────────────────────────────────────────────────

_CONCEPT_MAP: Optional[dict] = None
_CONCEPT_MAP_PATH = getattr(settings, "CONCEPT_MAP_FILE", None) or "/app/data/concept_map.yaml"


def _normalize_col(name: str) -> str:
    """Lowercase, collapse spaces/dashes/dots to single underscore, strip."""
    n = (name or "").strip().lower()
    n = re.sub(r"[\s\-./]+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n.strip("_")


def load_concept_map(path: Optional[str] = None, force: bool = False) -> dict:
    """Load and cache concept_map.yaml. Falls back to a sensible default path."""
    global _CONCEPT_MAP
    if _CONCEPT_MAP is not None and not force:
        return _CONCEPT_MAP

    candidates = [
        path,
        _CONCEPT_MAP_PATH,
        "/app/data/concept_map.yaml",
        # local-dev path (repo layout)
        __file__.rsplit("/backend/", 1)[0] + "/data/concept_map.yaml"
        if "/backend/" in __file__ else None,
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            with open(cand, "r") as f:
                _CONCEPT_MAP = yaml.safe_load(f) or {}
                logger.info("Loaded concept_map from %s", cand)
                return _CONCEPT_MAP
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to parse concept_map %s: %s", cand, exc)
            continue

    logger.warning("No concept_map.yaml found — event extraction will be empty")
    _CONCEPT_MAP = {}
    return _CONCEPT_MAP


# ── Value / date parsing ──────────────────────────────────────────────────────

def _parse_number(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        num = float(cleaned)
        return -abs(num) if neg else num
    except ValueError:
        return None


_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y",
    "%Y/%m/%d", "%m-%d-%Y", "%d/%m/%Y", "%Y%m%d",
)


def _parse_date(val: Any) -> tuple[Optional[date], Optional[str]]:
    if val is None:
        return None, None
    raw = str(val).strip()
    if not raw:
        return None, None
    if isinstance(val, (datetime, date)):
        d = val.date() if isinstance(val, datetime) else val
        return d, raw
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw[:len(datetime.now().strftime(fmt)) + 4], fmt).date(), raw
        except ValueError:
            continue
    # last resort: leading ISO date
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])), raw
        except ValueError:
            pass
    return None, raw


def _clean_text(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "unknown", ""):
        return None
    return s


# ── Extraction ────────────────────────────────────────────────────────────────

def _first_present(norm_row: dict, synonyms: list[str]) -> Optional[str]:
    """Return the value of the first synonym present (and non-empty) in the row."""
    for syn in synonyms:
        if syn in norm_row:
            v = _clean_text(norm_row[syn])
            if v is not None:
                return v
    return None


def _paired_label(norm_row: dict, value_col: str, label_cols: list[str]) -> Optional[str]:
    """
    Find a description paired with a *_code column.
    Positional pairing: icd1_code → icd1_name / icd1_description.
    Falls back to the first present label column.
    """
    if value_col.endswith("_code"):
        stem = value_col[:-5]
        for suffix in ("_name", "_description", "_desc"):
            cand = stem + suffix
            v = _clean_text(norm_row.get(cand))
            if v is not None:
                return v
    return _first_present(norm_row, label_cols)


def extract_events_from_row(
    row: dict,
    source_file: Optional[str] = None,
    patient_key_override: Optional[str] = None,
) -> list[Event]:
    """Map a single raw row (dict) into a list of canonical Events."""
    cmap = load_concept_map()
    if not cmap:
        return []

    norm_row = {_normalize_col(k): v for k, v in row.items()}

    # Patient key + display name
    patient_key = patient_key_override
    if patient_key is None:
        patient_key = _first_present(norm_row, cmap.get("patient_key_columns", []))
    patient_name = _first_present(norm_row, cmap.get("patient_name_columns", []))

    # Event time
    event_time, event_time_raw = None, None
    for ts_col in cmap.get("timestamp_columns", []):
        if ts_col in norm_row:
            et, raw = _parse_date(norm_row[ts_col])
            if raw is not None:
                event_time, event_time_raw = et, raw
                if et is not None:
                    break  # prefer a parseable timestamp

    events: list[Event] = []
    for concept, spec in (cmap.get("concepts") or {}).items():
        ctype = (spec.get("type") or "textual").lower()
        value_cols = [_normalize_col(c) for c in spec.get("value_columns", [])]
        label_cols = [_normalize_col(c) for c in spec.get("label_columns", [])]

        for vcol in value_cols:
            if vcol not in norm_row:
                continue
            raw_val = _clean_text(norm_row[vcol])
            if raw_val is None:
                continue

            if ctype == "numeric":
                num = _parse_number(raw_val)
                if num is None:
                    continue
                label = _paired_label(norm_row, vcol, label_cols) or concept
                raw_text = f"{label}: {num}"
                events.append(Event(
                    patient_key=patient_key, concept=concept, concept_type="numeric",
                    value_num=num, value_text=label,
                    event_time=event_time, event_time_raw=event_time_raw,
                    source_file=source_file, raw_text=raw_text,
                    patient_name=patient_name,
                ))
            else:
                label = _paired_label(norm_row, vcol, label_cols)
                text = f"{raw_val} — {label}" if label else raw_val
                raw_text = f"{concept}: {text}"
                events.append(Event(
                    patient_key=patient_key, concept=concept, concept_type="textual",
                    value_num=None, value_text=text,
                    event_time=event_time, event_time_raw=event_time_raw,
                    source_file=source_file, raw_text=raw_text,
                    patient_name=patient_name,
                ))

    return events


# ── Persistence ───────────────────────────────────────────────────────────────

def persist_events(events: list[Event]) -> int:
    """Insert events idempotently (ON CONFLICT (row_hash) DO NOTHING)."""
    if not events:
        return 0
    from psycopg2.extras import execute_values

    rows = [
        (
            e.patient_key, e.patient_name, e.concept, e.concept_type,
            e.value_num, e.value_text,
            e.event_time, e.event_time_raw,
            e.source_file, e.raw_text, e.row_hash(),
        )
        for e in events
    ]
    with get_db() as conn:
        with conn.cursor() as cur:
            # fetch=True makes execute_values return rows from EVERY page, so the
            # count is accurate. (cur.rowcount only reflects the last page and
            # under-reports when the insert spans multiple pages.)
            returned = execute_values(
                cur,
                "INSERT INTO events "
                "(patient_key, patient_name, concept, concept_type, value_num, value_text, "
                " event_time, event_time_raw, source_file, raw_text, row_hash) "
                "VALUES %s ON CONFLICT (row_hash) DO NOTHING RETURNING 1",
                rows,
                fetch=True,
            )
            inserted = len(returned) if returned else 0
    return inserted


def _list_staging_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT staging_table FROM source_catalog "
            "WHERE staging_table IS NOT NULL AND aliased_to IS NULL"
        )
        return [r[0] for r in cur.fetchall()]


def backfill_events(limit: Optional[int] = None) -> int:
    """
    Build the canonical events table FROM all staging tables.

    Idempotent: re-running only inserts events not already present (row_hash).
    Returns the number of newly-inserted events.
    """
    total_inserted = 0
    with get_readonly_db() as conn:
        try:
            staging_tables = _list_staging_tables(conn)
        except Exception as exc:
            logger.warning("backfill_events: could not list staging tables (%s)", exc)
            staging_tables = []

    for tbl in staging_tables:
        with get_readonly_db() as conn:
            with conn.cursor() as cur:
                q = f'SELECT * FROM "{tbl}" WHERE _deleted_at IS NULL'
                if limit:
                    q += f" LIMIT {int(limit)}"
                cur.execute(q)
                rows = fetchall_dicts(cur)

        batch: list[Event] = []
        for row in rows:
            src = row.get("_source_file") or tbl
            # Drop system columns before mapping
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            batch.extend(extract_events_from_row(clean, source_file=src))
            if len(batch) >= 2000:
                total_inserted += persist_events(batch)
                batch = []
        if batch:
            total_inserted += persist_events(batch)
        logger.info("backfill_events: processed staging table %s (%d rows)", tbl, len(rows))

    logger.info("backfill_events: inserted %d new event(s)", total_inserted)
    return total_inserted


# ── Query helpers (used by ETHER) ─────────────────────────────────────────────

def list_patient_keys() -> list[str]:
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT patient_key FROM events "
                "WHERE patient_key IS NOT NULL ORDER BY patient_key"
            )
            return [r[0] for r in cur.fetchall()]


def numeric_trajectories(patient_key: Optional[str] = None) -> dict[str, list[tuple]]:
    """
    Return {concept_label: [(value, date), …]} for numeric events, time-ordered.
    The 'indicator' is value_text (e.g. lab/test name) when present, else concept.
    """
    where = "WHERE concept_type = 'numeric'"
    params: list[Any] = []
    if patient_key is not None:
        where += " AND patient_key = %s"
        params.append(patient_key)

    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COALESCE(value_text, concept) AS indicator, value_num, event_time "
                f"FROM events {where} ORDER BY indicator, event_time NULLS LAST",
                params,
            )
            rows = cur.fetchall()

    traj: dict[str, list[tuple]] = {}
    for indicator, value, ev_time in rows:
        traj.setdefault(indicator, []).append((float(value) if value is not None else None, ev_time))
    return traj


def patient_rollup(
    patient_name: Optional[str] = None,
    patient_key: Optional[str] = None,
    concept: Optional[str] = None,
    limit: int = 500,
) -> dict[str, list[dict]]:
    """
    Deterministic, COMPLETE per-patient rollup straight from the events table —
    not semantic top-k. Returns {concept: [{value, first, last, count}, …]}.

    Scales: matches on the indexed patient_key (exact) when given, else on a
    pg_trgm-indexed case-insensitive name match, so it reads one patient's events
    rather than scanning the whole (growing) table.
    """
    clauses, params = [], []
    if patient_key:
        clauses.append("patient_key = %s")
        params.append(str(patient_key))
    elif patient_name:
        # Token-AND match so "Mcgee Patricia L" matches stored "McGee, Patricia L"
        # regardless of punctuation/order. Each clause uses the trigram index.
        tokens = [t for t in re.split(r"[^a-z0-9]+", patient_name.lower()) if len(t) >= 2]
        if not tokens:
            return {}
        for tok in tokens[:5]:
            clauses.append("lower(patient_name) LIKE %s")
            params.append(f"%{tok}%")
    else:
        return {}
    if concept:
        clauses.append("concept = %s")
        params.append(concept)

    where = " AND ".join(clauses)
    sql = (
        "SELECT concept, COALESCE(value_text, value_num::text) AS val, "
        "       MIN(event_time) AS first_seen, MAX(event_time) AS last_seen, COUNT(*) AS n "
        f"FROM events WHERE {where} "
        "GROUP BY concept, val ORDER BY concept, n DESC "
        f"LIMIT {int(limit)}"
    )
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = fetchall_dicts(cur)

    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["concept"], []).append({
            "value": r["val"],
            "first": r["first_seen"].isoformat() if r["first_seen"] else None,
            "last": r["last_seen"].isoformat() if r["last_seen"] else None,
            "count": int(r["n"]),
        })
    return out


def textual_events(patient_key: Optional[str] = None, concepts: Optional[list[str]] = None) -> list[Event]:
    """Return textual events (optionally filtered by patient / concept), time-ordered."""
    where = "WHERE concept_type = 'textual'"
    params: list[Any] = []
    if patient_key is not None:
        where += " AND patient_key = %s"
        params.append(patient_key)
    if concepts:
        where += " AND concept = ANY(%s)"
        params.append(concepts)

    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT patient_key, patient_name, concept, value_text, event_time, "
                f"event_time_raw, source_file, raw_text FROM events {where} "
                f"ORDER BY event_time NULLS LAST",
                params,
            )
            rows = fetchall_dicts(cur)

    return [
        Event(
            patient_key=r["patient_key"], concept=r["concept"], concept_type="textual",
            value_num=None, value_text=r["value_text"],
            event_time=r["event_time"], event_time_raw=r["event_time_raw"],
            source_file=r["source_file"], raw_text=r["raw_text"],
            patient_name=r.get("patient_name"),
        )
        for r in rows
    ]
