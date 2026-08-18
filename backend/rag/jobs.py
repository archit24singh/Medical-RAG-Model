"""
Background ingestion jobs (Phase 6 — scale).

The synchronous /ingest endpoint blocks the HTTP request for the entire ingest,
which times out on large multi-dataset loads. This module runs ingestion in a
background thread and tracks progress, so callers submit a job and poll status.

Kept deliberately lightweight (in-process thread + status dict) — no Redis /
Celery dependency. For a true multi-worker deployment this is the seam to swap
in a real task queue; the API (submit_directory_job / get_job) stays the same.

Additive & idempotent: the underlying ingest path already dedupes by row hash
and upserts by stable id, so a new dataset is added without touching existing
data, and re-running a job is safe.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _new_job(kind: str, target: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id, "kind": kind, "target": target,
            "status": "queued", "started_at": time.time(),
            "finished_at": None, "result": None, "error": None,
        }
    return job_id


def _update(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def get_job(job_id: str) -> Optional[dict]:
    with _JOBS_LOCK:
        return dict(_JOBS[job_id]) if job_id in _JOBS else None


def list_jobs() -> list[dict]:
    with _JOBS_LOCK:
        return [dict(j) for j in _JOBS.values()]


def submit_directory_job(bucket_dir: Optional[str] = None, build_events: bool = True) -> str:
    """
    Start a background job that ingests a directory and (optionally) builds the
    canonical events + event index afterwards. Returns a job_id to poll.
    """
    from config import settings
    target = bucket_dir or settings.BUCKET_DIR
    job_id = _new_job("ingest_directory", target)

    def _run():
        _update(job_id, status="running")
        try:
            from rag.ingestion import ingest_directory
            res = ingest_directory(target)
            result = {
                "success_count": len(res.get("success", [])),
                "error_count": len(res.get("errors", [])),
                "skipped_count": len(res.get("skipped", [])),
            }
            if build_events:
                from rag.events import backfill_events
                from rag.event_chunker import index_all_events
                result["events_inserted"] = backfill_events()
                result["event_chunks_indexed"] = index_all_events()
            _update(job_id, status="completed", result=result, finished_at=time.time())
            logger.info("Job %s completed: %s", job_id, result)
        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            _update(job_id, status="failed", error=str(exc), finished_at=time.time())

    threading.Thread(target=_run, daemon=True).start()
    return job_id
