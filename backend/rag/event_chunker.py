"""
Timestamp-preserving event chunker (Phase 1).

Replaces the old blind character-window chunker for event data. Instead of
splitting text without regard to time, it groups a patient's *textual* canonical
events into time-ordered chunks, and stamps each chunk with its temporal
metadata (``event_time`` = latest event in the chunk, ``event_time_start`` =
earliest). ETHER's U-shaped time-aware scoring (Phase 2) reads ``event_time``
off each retrieved chunk, which is only possible because the chunk carries it.

Chunks are written to the existing ChromaDB collection with stable IDs so
re-indexing is idempotent (upsert), and metadata ``doc_type='event'`` so the
event index can be filtered apart from free-text documents when needed.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Optional

from config import settings
from rag.events import Event, textual_events, list_patient_keys
from rag.vectorstore import add_documents_batch

logger = logging.getLogger(__name__)


def _chunk_id(patient_key: Optional[str], idx: int) -> str:
    h = hashlib.sha256(f"event::{patient_key}::{idx}".encode()).hexdigest()[:24]
    return f"evt_{h}"


def _events_to_chunks(events: list[Event], chunk_size: int) -> list[dict]:
    """
    Group time-ordered textual events into chunks of ~chunk_size characters,
    preserving order and capturing each chunk's time span.
    """
    chunks: list[dict] = []
    buf: list[Event] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        times = [e.event_time for e in buf if e.event_time is not None]
        start = min(times) if times else None
        end = max(times) if times else None

        # Self-describing header so name-based Q&A can find the patient and the
        # LLM can see dates. Each event line is prefixed with its date.
        name = next((e.patient_name for e in buf if e.patient_name), None)
        key = buf[0].patient_key
        who = name or (f"patient {key}" if key else "patient")
        span = ""
        if start and end:
            span = f" Records {start.isoformat()} to {end.isoformat()}."
        header = f"Patient: {who}.{span}"

        lines = [header]
        for e in buf:
            d = e.event_time.isoformat() if isinstance(e.event_time, date) else "date unknown"
            lines.append(f"[{d}] {e.raw_text}")
        text = "\n".join(lines)

        chunks.append({
            "text": text,
            "event_time": end.isoformat() if isinstance(end, date) else None,
            "event_time_start": start.isoformat() if isinstance(start, date) else None,
            "source_file": buf[0].source_file,
            "patient_key": key,
            "patient_name": name,
            "n_events": len(buf),
        })
        buf, buf_len = [], 0

    for e in events:
        ln = len(e.raw_text) + 1
        if buf and buf_len + ln > chunk_size:
            flush()
        buf.append(e)
        buf_len += ln
    flush()
    return chunks


def index_patient_events(patient_key: Optional[str], chunk_size: Optional[int] = None) -> int:
    """Build timestamp-preserving chunks for one patient and upsert to the vector store."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    events = textual_events(patient_key=patient_key)
    if not events:
        return 0

    chunks = _events_to_chunks(events, chunk_size)
    ids, texts, metas = [], [], []
    for i, ch in enumerate(chunks):
        ids.append(_chunk_id(patient_key, i))
        texts.append(ch["text"])
        metas.append({
            "doc_type": "event",
            "patient_key": ch["patient_key"] or "",
            "patient_name": ch["patient_name"] or "",
            "event_time": ch["event_time"] or "",
            "event_time_start": ch["event_time_start"] or "",
            "file_name": ch["source_file"] or "events",
            "chunk_number": i,
            "n_events": ch["n_events"],
        })
    added = add_documents_batch(ids, texts, metas)
    return added


def index_all_events(chunk_size: Optional[int] = None) -> int:
    """Index timestamped event chunks for every patient. Returns total chunks added."""
    total = 0
    for pk in list_patient_keys():
        total += index_patient_events(pk, chunk_size)
    logger.info("index_all_events: upserted %d event chunk(s)", total)
    return total
