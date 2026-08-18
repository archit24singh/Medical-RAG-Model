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
from typing import Any, Optional

from config import settings
from rag.audit_logger import log_query_event

logger = logging.getLogger(__name__)


# ── Prediction-task registry ──────────────────────────────────────────────────

_PRED_TASKS: Optional[dict] = None


def _load_prediction_tasks() -> dict:
    global _PRED_TASKS
    if _PRED_TASKS is not None:
        return _PRED_TASKS
    import yaml
    candidates = [
        settings.PREDICTION_TASKS_FILE,
        "/app/data/prediction_tasks.yaml",
        __file__.rsplit("/backend/", 1)[0] + "/data/prediction_tasks.yaml"
        if "/backend/" in __file__ else None,
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            with open(cand) as f:
                _PRED_TASKS = (yaml.safe_load(f) or {}).get("tasks", {})
                return _PRED_TASKS
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to parse prediction_tasks %s: %s", cand, exc)
    _PRED_TASKS = {}
    return _PRED_TASKS


def predict(task_name: str, patient_key: Optional[str] = None,
            prediction_time: Optional[str] = None) -> dict[str, Any]:
    """
    Run a clinical prediction (DER) for a named task from prediction_tasks.yaml.

    Args:
        task_name:        key in prediction_tasks.yaml (e.g. 'readmission_30d')
                          OR a free-text task description (treated ad-hoc, binary).
        patient_key:      restrict evidence to one patient (recommended).
        prediction_time:  optional ISO date anchor (τ*).

    Returns the DER result dict plus the task definition used.
    """
    from datetime import date
    from rag.der import predict as der_predict

    tasks = _load_prediction_tasks()
    spec = tasks.get(task_name)
    if spec is None:
        # Ad-hoc binary task from a free-text description
        spec = {"task": task_name, "outcome": "meets the described outcome",
                "labels": ["yes", "no"]}

    tau_star = None
    if prediction_time:
        try:
            tau_star = date.fromisoformat(prediction_time[:10])
        except ValueError:
            logger.warning("Bad prediction_time %r — ignoring", prediction_time)

    t0 = time.monotonic()
    result = der_predict(
        task_query=spec["task"],
        outcome_phrase=spec["outcome"],
        labels=spec["labels"],
        patient_key=patient_key,
        tau_star=tau_star,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    result["task_name"] = task_name
    result["task"] = spec
    result["latency_ms"] = latency_ms

    log_query_event(
        query=f"PREDICT {task_name} patient={patient_key}",
        route="predict",
        sql_generated=None,
        source_ids=[],
        latency_ms=latency_ms,
        error=None,
    )
    return result


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
        if settings.USE_SQL_AGENT:
            # Structured questions go through the LangChain SQL agent (schema
            # introspection + tool-using loop over Postgres). RAG is the
            # semantic fallback for reference questions the agent can't answer.
            from rag.sql_agent import answer as _sql_agent_answer
            agent_res = _sql_agent_answer(user_query)
            if agent_res is not None:
                result = agent_res
            else:
                from rag.retriever import query as _retriever_query
                result = _retriever_query(user_query)
        else:
            # Legacy path: deterministic query shapes + custom text-to-SQL fallback.
            structured = _try_structured(user_query)
            profile = None if structured is not None else _try_patient_profile(user_query)
            rollup = None if (structured is not None or profile is not None) else _try_event_rollup(user_query)
            entity = None if (structured is not None or profile is not None or rollup is not None) \
                else _try_entity_qa(user_query)

            if structured is not None:
                result = structured
            elif profile is not None:
                result = profile
            elif rollup is not None:
                result = rollup
            elif entity is not None:
                result = entity
            elif settings.ENABLE_COMPOUND_LOOP:
                result = _compound_query(user_query)
            else:
                from rag.retriever import query as _retriever_query
                result = _retriever_query(user_query)

            if _looks_like_failure(result):
                sql_result = _try_text_to_sql(user_query)
                if sql_result is not None:
                    result = sql_result
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


# ── Structured per-patient event rollup ──────────────────────────────────────
# Deterministic "list all <concept> for patient X" answers straight from the
# events table — COMPLETE results, not semantic top-k. Triggers only when the
# query clearly asks for a patient's set of a known concept; otherwise returns
# None and the normal retrieval path handles it.

_ROLLUP_CONCEPTS = {
    "diagnosis":  ["diagnos", "icd", "condition"],
    "procedure":  ["procedure", "cpt", "surger", "operation"],
    "medication": ["medication", "drug", "prescri", " rx"],
    "visit_type": ["visit"],
    "charge":     ["charge", "billed", "payment", "cost"],
}
_ROLLUP_CUES = ("all ", "list", "every", "which", "associated", "history",
                "summary", "what diagnos", "what procedure", "what medication",
                "does ", "has ", "have ")


def _try_event_rollup(user_query: str) -> Optional[dict[str, Any]]:
    q = user_query.lower()

    concept = None
    for c, kws in _ROLLUP_CONCEPTS.items():
        if any(kw in q for kw in kws):
            concept = c
            break
    if concept is None:
        return None
    if not any(cue in q for cue in _ROLLUP_CUES):
        return None

    # Identify the patient (name ≥2 tokens, or numeric key)
    from rag.intent_parser import parse_intent
    intent = parse_intent(user_query)
    name = intent.get("patient_name") or ""
    key = intent.get("patient_id") or intent.get("subject_id")
    if len(name.split()) < 2 and not key:
        return None

    try:
        from rag.events import patient_rollup
        roll = patient_rollup(patient_name=name or None, patient_key=key, concept=concept)
    except Exception as exc:
        logger.warning("Event rollup failed (%s) — falling back to retrieval", exc)
        return None

    items = roll.get(concept, [])
    who = name or f"patient {key}"
    if not items:
        return None  # let normal retrieval try (e.g. data is in a PDF, not events)

    lines = [f"{concept.replace('_', ' ').title()} for {who} ({len(items)} distinct):"]
    for it in items:
        span = ""
        if it["first"] and it["last"]:
            span = f"  ({it['first']}" + (f" → {it['last']}" if it["last"] != it["first"] else "") + ")"
        cnt = f"  ×{it['count']}" if it["count"] > 1 else ""
        lines.append(f"• {it['value']}{span}{cnt}")

    return {
        "answer": "\n".join(lines),
        "documents": [],
        "intent": intent,
        "filter_used": {"concept": concept, "patient": who},
        "query_path": "event_rollup",
    }


# ── Full per-patient profile (all columns, schema-on-read) ────────────────────
_PROFILE_CUES = (
    "profile", "all details", "all information", "all info", "everything about",
    "all columns", "full record", "complete record", "all the details",
    "all fields", "every detail", "summary of",
    "patient details", "full patient", "patient detail", "details for",
    "full details", "all patient", "complete details",
)


def _try_patient_profile(user_query: str) -> Optional[dict[str, Any]]:
    q = user_query.lower()
    if not any(cue in q for cue in _PROFILE_CUES):
        return None

    from rag.intent_parser import parse_intent
    intent = parse_intent(user_query)
    name = intent.get("patient_name") or ""
    key = intent.get("patient_id") or intent.get("subject_id")
    if len(name.split()) < 2 and not key:
        return None

    try:
        from rag.profiles import patient_profile, format_profile
        prof = patient_profile(name=name or None, key=key)
    except Exception as exc:
        logger.warning("Patient profile failed (%s) — falling back", exc)
        return None

    if not prof:
        return None

    return {
        "answer": format_profile(prof),
        "documents": [],
        "intent": intent,
        "filter_used": {"patient": prof["patient"]},
        "query_path": "patient_profile",
        "profile": prof,
    }


# ── Text-to-SQL fallback (bridge) ─────────────────────────────────────────────

def _looks_like_failure(result: dict) -> bool:
    """True when the primary path could not confidently answer (→ try SQL)."""
    if not result:
        return True
    if result.get("hallucination_flag"):
        return True
    ans = (result.get("answer") or "").strip()
    if not ans or ans.startswith("⛔"):
        return True
    low = ans.lower()
    return (
        "no matching records" in low
        or "insufficient grounding" in low
        or "no analytical data" in low
    )


def _try_text_to_sql(user_query: str) -> Optional[dict[str, Any]]:
    """
    Robust text-to-SQL fallback (Planner → Writer → validate → repair) over the
    auto-exposed views. Best-effort only — used when no deterministic shape matched.
    """
    try:
        from rag.text_to_sql import generate_and_run
        return generate_and_run(user_query)
    except Exception as exc:
        logger.warning("Text-to-SQL fallback failed (%s)", exc)
        return None


# ── Deterministic structured QA (aggregation + field lookup) ──────────────────
# Aggregates are computed in SQL and factual fields are read straight from the
# column — the LLM never does arithmetic or picks a value, so it can't answer a
# different question than the one asked (e.g. a payment sum for a DOB query).

def _try_structured(user_query: str) -> Optional[dict[str, Any]]:
    try:
        from rag.intent_parser import parse_intent
        from rag.structured_qa import try_field_lookup, try_document_lookup
        intent = parse_intent(user_query)
        # 1. Document (invoice/order) lookup — scoped, complete, deterministic.
        doc = try_document_lookup(user_query, intent)
        if doc is not None:
            return doc
        # 2. Generic deterministic query shapes (aggregate/count/distinct) —
        #    schema-agnostic, resolves table+column dynamically. 100% reliable.
        from rag.query_shapes import answer as shapes_answer
        shaped = shapes_answer(user_query)
        if shaped is not None:
            return shaped
        # 3. Single-field lookup for a named entity.
        return try_field_lookup(user_query, intent)
    except Exception as exc:
        logger.warning("Structured QA failed (%s) — falling back", exc)
        return None


# ── Entity QA — answer ANY question about a named patient/provider from ───────
# whatever column holds the value (schema-on-read). This is the default path for
# any query naming a specific patient or provider that isn't an explicit
# concept rollup or full-profile dump. The LLM answers strictly from the
# entity's own structured records, so any column is reachable without hardcoding.

_ENTITY_QA_PROMPT = """\
You are answering a question about a specific {role} using ONLY the structured \
records below. Do not use outside knowledge.

Question: {question}

{role} records:
{data}

Rules:
- Use ONLY the records above. Quote values verbatim (dates, amounts, codes, names).
- If the question asks for one field, answer it directly and concisely.
- If several records are relevant, summarise them briefly.
- If no field in the records answers the question, say exactly:
  "That information is not available in the records for this {role}."
- Format money with a $ sign; keep dates exactly as shown.

Answer:"""


def _try_entity_qa(user_query: str) -> Optional[dict[str, Any]]:
    from rag.intent_parser import parse_intent
    intent = parse_intent(user_query)

    role = "provider" if intent.get("query_type") == "provider" else "patient"
    if role == "provider":
        name = intent.get("provider_name") or ""
        key = intent.get("provider_npi")
    else:
        name = intent.get("patient_name") or ""
        key = intent.get("patient_id") or intent.get("subject_id")

    toks = name.split()
    has_name = len(toks) >= 2 or (len(toks) == 1 and len(toks[0]) >= 3)
    if not has_name and not key:
        return None  # no specific entity → let reference/analytical retrieval handle it

    try:
        from rag.profiles import entity_profile, profile_context
        prof = entity_profile(name=name or None, key=key, role=role)
        if not prof:
            # name may belong to the other role (e.g. matched a provider not a patient)
            other = "patient" if role == "provider" else "provider"
            prof = entity_profile(name=name or None, key=key, role=other)
            if prof:
                role = other
    except Exception as exc:
        logger.warning("Entity QA profile lookup failed (%s) — falling back", exc)
        return None

    if not prof:
        return None

    context = profile_context(prof, max_rows=30)
    try:
        from rag.llm_client import call_llm
        answer = call_llm(_ENTITY_QA_PROMPT.format(role=role, question=user_query, data=context))
    except Exception as exc:
        logger.warning("Entity QA LLM call failed (%s) — falling back", exc)
        return None

    return {
        "answer": answer,
        "documents": [],
        "intent": intent,
        "filter_used": {"entity": prof.get("entity"), "role": role},
        "query_path": "entity_qa",
    }


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
