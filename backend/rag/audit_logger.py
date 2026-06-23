"""
Structured JSON-lines audit logger — one record per query.

HIPAA note
----------
This log records query text, route, any generated SQL, and source document IDs.
It does NOT log the full LLM-generated answer or retrieved document content.
Treat the log file as PHI-adjacent and secure accordingly (restricted file
permissions, encrypted at-rest storage, log rotation).

Rotation
--------
RotatingFileHandler replaces the file every AUDIT_LOG_MAX_BYTES bytes,
keeping AUDIT_LOG_BACKUP_COUNT old files.  With defaults of 10 MB × 10 files
that is 100 MB total retention.

Record format (one JSON object per line)
-----------------------------------------
{
  "timestamp":  "2025-06-01T12:34:56Z",
  "query":      "What is Alice Johnson's total billed amount for May 2025?",
  "route":      "sql",          -- sql | rag | hybrid | analytical | error
  "sql":        null,           -- LLM-generated SQL (text-to-SQL path only)
  "source_ids": ["sql:42"],     -- document / record IDs used in the answer
  "latency_ms": 312.4,          -- end-to-end query latency
  "error":      null            -- error message if the query failed
}
"""

import json
import logging
import logging.handlers
import os
import time
from typing import Optional

from config import settings

# Module-level audit logger singleton — initialised lazily on first write.
_audit_log: Optional[logging.Logger] = None


def _get_audit_log() -> logging.Logger:
    """Return the audit logger, initialising it on first call."""
    global _audit_log
    if _audit_log is not None:
        return _audit_log

    log = logging.getLogger("rag.audit")
    log.setLevel(logging.INFO)
    log.propagate = False  # Don't bubble up to the root application logger

    # Ensure the log directory exists
    log_dir = os.path.dirname(settings.AUDIT_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=settings.AUDIT_LOG_PATH,
        maxBytes=settings.AUDIT_LOG_MAX_BYTES,
        backupCount=settings.AUDIT_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    # Raw JSON-lines: formatter passes the message through unchanged.
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)

    _audit_log = log
    return _audit_log


def log_query_event(
    *,
    query:         str,
    route:         str,
    sql_generated: Optional[str] = None,
    source_ids:    Optional[list] = None,
    latency_ms:    Optional[float] = None,
    error:         Optional[str] = None,
) -> None:
    """
    Write one JSON-lines audit record to the rotating log file.

    Parameters
    ----------
    query         : The user's raw query string (may contain PHI).
    route         : Query path — 'sql', 'rag', 'hybrid', 'analytical', 'error'.
    sql_generated : LLM-generated SQL statement (text-to-SQL path only).
    source_ids    : Document / record IDs that contributed to the answer.
    latency_ms    : End-to-end query latency in milliseconds.
    error         : Error message if the query pipeline raised an exception.

    This function is intentionally infallible — a logging failure must never
    propagate to the query path and cause a user-visible 500 error.
    """
    record = {
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query":      query,
        "route":      route,
        "sql":        sql_generated,
        "source_ids": source_ids or [],
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "error":      error,
    }
    try:
        _get_audit_log().info(json.dumps(record, ensure_ascii=False))
    except Exception as exc:
        # Last resort: write to the application logger so we at least see it.
        logging.getLogger(__name__).warning("Audit log write failed: %s", exc)
