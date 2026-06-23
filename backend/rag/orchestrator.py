"""
Query orchestrator — the single entry point for all RAG queries.

This thin dispatch layer sits above both the SQL retriever and the RAG
retriever.  Its two jobs are:

  1. Dispatch: route to the single-pass pipeline (default) or the compound
     loop (behind ENABLE_COMPOUND_LOOP flag).
  2. Audit: emit one JSON-lines audit record per query via audit_logger.

Normal mode (ENABLE_COMPOUND_LOOP=false, default)
--------------------------------------------------
  orchestrator.query(user_query)
      → retriever.query(user_query)      — existing intent → route → answer pipeline

Compound loop mode (ENABLE_COMPOUND_LOOP=true, NOT YET IMPLEMENTED)
-------------------------------------------------------------------
  Contract:
    Step 1. SQL exact-lookup to ground the query in structured facts.
    Step 2. Observe the returned rows; identify what additional context is needed.
    Step 3. Build a targeted RAG sub-query from the SQL observations.
    Step 4. Retrieve unstructured documents for the sub-query.
    Step 5. Synthesize an answer grounded in both SQL facts and RAG context.
            Fail hard (return raw sources) if grounding check fails.

  Enable only after Steps 1–5 are built and tested.  The flag exists as a
  clean feature seam — no other code needs to change when the loop ships.
"""

import logging
import time
from typing import Any

from config import settings
from rag.audit_logger import log_query_event

logger = logging.getLogger(__name__)


def query(user_query: str) -> dict[str, Any]:
    """
    Main entry point — call this from main.py instead of retriever.query().

    Returns the same dict shape as retriever.query():
      {
        "answer":      str,
        "documents":   list[dict],
        "intent":      dict,
        "filter_used": dict | None,
        "query_path":  str,
        ...optional fields...
      }
    """
    t0 = time.monotonic()
    result: dict[str, Any] = {}
    error_msg: str | None = None

    try:
        if settings.ENABLE_COMPOUND_LOOP:
            result = _compound_query(user_query)
        else:
            from rag.retriever import query as _retriever_query
            result = _retriever_query(user_query)
    except Exception as exc:
        logger.error("Orchestrator query failed: %s", exc, exc_info=True)
        error_msg = str(exc)
        result = {
            "answer":     f"Query failed: {exc}",
            "documents":  [],
            "intent":     {},
            "query_path": "error",
        }
    finally:
        latency_ms = (time.monotonic() - t0) * 1000
        log_query_event(
            query=user_query,
            route=result.get("query_path", "unknown"),
            sql_generated=result.get("sql_generated"),
            source_ids=[
                d.get("id") for d in result.get("documents", []) if d.get("id")
            ],
            latency_ms=latency_ms,
            error=error_msg,
        )

    return result


def _compound_query(user_query: str) -> dict[str, Any]:
    """
    Compound query loop — SQL → observe rows → build RAG query → retrieve → synthesize.

    NOT YET IMPLEMENTED.  Raises NotImplementedError so the flag being
    accidentally enabled surfaces immediately rather than silently falling back.

    When implemented this function must:
      1. Run sql_retriever.lookup() to ground the query in structured facts.
      2. Examine returned rows; decide what unstructured context is needed.
      3. Build a targeted sub-query string (e.g. "ICD coding guidelines for 410.0").
      4. Call retriever._hybrid_search() with the sub-query.
      5. Merge SQL facts + RAG chunks; synthesize + hallucination-check.
      6. On grounding failure: return raw sources + hard failure message.
    """
    raise NotImplementedError(
        "Compound query loop is not yet implemented. "
        "Set ENABLE_COMPOUND_LOOP=false (the default) to use the single-pass pipeline."
    )
