"""
AIR — Adaptive Iterative Retrieval (paper §3.3).

Single-pass retrieval misses evidence that is temporally dispersed or only
indirectly related to the initial query. AIR progressively expands coverage in
a controlled way:

  1. Retrieve initial evidence with ETHER for the task query.
  2. Ask the LLM whether the current evidence is sufficient to answer the task.
  3. If not, the LLM emits ONE focused, non-redundant refinement query targeting
     a single missing clinical dimension.
  4. Retrieve for the refined query and MERGE (dedup + temporal order) into the
     running evidence set.
  5. Repeat until sufficient or AIR_MAX_ITERATIONS / AIR_MAX_EVIDENCE_CHUNKS hit.

All LLM steps use JSON mode and degrade gracefully (a malformed/again-failed
response just stops the loop) so a local model never hard-fails the pipeline.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from config import settings
from rag.llm_client import call_llm_json
from rag.ether import gather_evidence, _to_date

logger = logging.getLogger(__name__)


_SUFFICIENCY_SYSTEM = (
    "You are a clinical evidence auditor. Given a prediction task and retrieved "
    "evidence, decide whether the evidence is sufficient to make a confident "
    "prediction. Respond ONLY with JSON."
)

_SUFFICIENCY_PROMPT = """\
Prediction task:
{task}

Currently retrieved evidence:
{evidence}

Is this evidence sufficient to answer the task confidently?
Reply with JSON exactly: {{"sufficient": true|false, "missing": "<one short phrase naming the single most important missing clinical aspect, or empty if sufficient>"}}"""

_REFINE_SYSTEM = (
    "You refine clinical retrieval queries. Produce ONE concise query focused on "
    "a single missing clinical dimension, non-redundant with what is already "
    "retrieved. Respond ONLY with JSON."
)

_REFINE_PROMPT = """\
Prediction task:
{task}

Already retrieved (do not repeat these):
{evidence}

Missing aspect to target: {missing}

Produce one short, focused retrieval query for ONLY that missing aspect.
Reply with JSON exactly: {{"query": "<the refined query>"}}"""


def _merge_textual(acc: list[dict], new: list[dict], cap: int) -> list[dict]:
    """Merge by id (dedup), re-sort chronologically, cap total size."""
    by_id = {c["id"]: c for c in acc}
    for c in new:
        by_id.setdefault(c["id"], c)
    merged = list(by_id.values())
    merged.sort(key=lambda c: (_to_date(c["metadata"].get("event_time")) or date.min))
    return merged[:cap]


def adaptive_retrieve(
    task_query: str,
    patient_key: Optional[str] = None,
    tau_star: Optional[date] = None,
    max_iterations: Optional[int] = None,
) -> dict:
    """
    Run the AIR loop. Returns a fused evidence bundle:
      {
        "textual":      [chunk dict, …],
        "numeric":      {indicator: [(value, date), …]},
        "evidence_text": str,
        "iterations":   int,
        "queries":      [str, …],   # task query + refinements used
      }
    """
    max_iterations = max_iterations or settings.AIR_MAX_ITERATIONS
    cap = settings.AIR_MAX_EVIDENCE_CHUNKS

    bundle = gather_evidence(task_query, patient_key, tau_star)
    textual = bundle["textual"]
    numeric = dict(bundle["numeric"])
    queries = [task_query]

    iterations = 0
    for _ in range(max_iterations):
        evidence_text = _serialize(textual, numeric)
        verdict = call_llm_json(
            _SUFFICIENCY_PROMPT.format(task=task_query, evidence=evidence_text or "(none)"),
            system=_SUFFICIENCY_SYSTEM,
            default={"sufficient": True, "missing": ""},
        )
        iterations += 1
        if verdict.get("sufficient", True):
            break
        if len(textual) >= cap:
            logger.info("AIR: evidence cap (%d) reached — stopping", cap)
            break

        missing = (verdict.get("missing") or "").strip()
        refined = call_llm_json(
            _REFINE_PROMPT.format(
                task=task_query, evidence=evidence_text or "(none)",
                missing=missing or "any additional relevant clinical evidence",
            ),
            system=_REFINE_SYSTEM,
            default={},
        )
        rq = (refined.get("query") or "").strip()
        if not rq or rq in queries:
            logger.info("AIR: no usable new refinement query — stopping")
            break
        queries.append(rq)

        new_bundle = gather_evidence(rq, patient_key, tau_star)
        textual = _merge_textual(textual, new_bundle["textual"], cap)
        for ind, vals in new_bundle["numeric"].items():
            numeric.setdefault(ind, vals)

    return {
        "textual": textual,
        "numeric": numeric,
        "evidence_text": _serialize(textual, numeric),
        "iterations": iterations,
        "queries": queries,
    }


def _serialize(textual: list[dict], numeric: dict) -> str:
    lines = []
    if numeric:
        lines.append("## Numeric indicators (most recent values)")
        for ind, vals in numeric.items():
            series = ", ".join(f"{v}{(' @' + str(t)) if t else ''}" for v, t in vals)
            lines.append(f"- {ind}: {series}")
        lines.append("")
    if textual:
        lines.append("## Clinical events (chronological)")
        for c in textual:
            t = c["metadata"].get("event_time") or "?"
            lines.append(f"[{t}] {c['content']}")
    return "\n".join(lines).strip()
