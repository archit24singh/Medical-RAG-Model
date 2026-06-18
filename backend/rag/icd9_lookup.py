"""
ICD9 code → description lookup.

MIMIC-III's DIAGNOSES_ICD.csv (and PROCEDURES_ICD.csv) store only a bare
ICD9_CODE per row — the human-readable description lives in a separate
lookup table, D_ICD_DIAGNOSES.csv (icd9_code, short_title, long_title).

This module lazily loads D_ICD_DIAGNOSES.csv (searched for under
settings.BUCKET_DIR) into an in-memory dict the first time a lookup is
requested, and caches it for the lifetime of the process. If the file
can't be found, lookups simply return None and ingestion proceeds with
diagnosis codes only (no description) — this is a soft dependency, not
a hard requirement.
"""

import csv
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_CACHE: dict[str, str] | None = None


def _load_d_icd_diagnoses() -> dict[str, str]:
    """
    Search settings.BUCKET_DIR recursively for D_ICD_DIAGNOSES.csv and build
    a {icd9_code: short_title} lookup dict. Returns {} if not found.
    """
    bucket = Path(settings.BUCKET_DIR)
    if not bucket.exists():
        logger.info("ICD9 lookup: bucket dir %s does not exist", bucket)
        return {}

    matches = list(bucket.rglob("D_ICD_DIAGNOSES.csv"))
    if not matches:
        logger.info("ICD9 lookup: D_ICD_DIAGNOSES.csv not found under %s", bucket)
        return {}

    csv_path = matches[0]
    lookup: dict[str, str] = {}
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            # Normalize header names (case-insensitive)
            fieldnames = {name.lower().strip(): name for name in (reader.fieldnames or [])}
            code_col  = fieldnames.get("icd9_code")
            title_col = fieldnames.get("short_title") or fieldnames.get("long_title")
            if not code_col or not title_col:
                logger.warning(
                    "ICD9 lookup: %s missing icd9_code/short_title columns — got %s",
                    csv_path, reader.fieldnames,
                )
                return {}
            for row in reader:
                code = (row.get(code_col) or "").strip()
                title = (row.get(title_col) or "").strip()
                if code and title:
                    lookup[code] = title
    except Exception as exc:
        logger.warning("ICD9 lookup: failed to read %s: %s", csv_path, exc)
        return {}

    logger.info("ICD9 lookup: loaded %d code(s) from %s", len(lookup), csv_path)
    return lookup


def lookup_icd9(code: str | None) -> str | None:
    """
    Return the short title/description for an ICD9 code, or None if the
    code is empty or no lookup table was found.

    MIMIC-III stores ICD9_CODE without a decimal point (e.g. "99591" for
    "995.91"); D_ICD_DIAGNOSES.csv uses the same undotted form, so no
    reformatting is needed for an exact match.
    """
    global _CACHE
    if not code:
        return None

    if _CACHE is None:
        _CACHE = _load_d_icd_diagnoses()

    return _CACHE.get(code.strip())
