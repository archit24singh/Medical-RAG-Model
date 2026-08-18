"""
ETHER — Event- and Time-Aware Hybrid EHR Retrieval (paper §3.2).

Two complementary evidence paths over the canonical events table:

  Textual path (U-shaped time-aware retrieval)
  --------------------------------------------
  Dense candidate chunks are re-scored by combining semantic similarity with a
  U-shaped temporal weight that emphasises events near the prediction time τ*
  AND near trajectory onset τ_first (early disease onset is often as informative
  as recent events — paper §3.2):

      s_time(τc) = max( exp(-(τ*-τc)/τ_recent), exp(-(τc-τ_first)/τ_early) )
      s(c)       = α·s_sem + (1-α)·s_time

  Numeric path (indicator-wise aggregation + coarse-to-fine selection)
  --------------------------------------------------------------------
  Numeric events are grouped into per-indicator trajectories. A coarse step
  ranks indicators by relevance to the task query (CrossEncoder — a cheap,
  deterministic local stand-in for the paper's LLM reranker), a fine step keeps
  the top N_fine indicators, and the N_recent most-recent values per indicator
  form the numeric evidence.

Output: a fused evidence bundle (textual chunks + numeric trajectories) plus a
serialised ``evidence_text`` string ready to drop into an LLM prompt.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Optional

from config import settings
from rag.events import numeric_trajectories
from rag.vectorstore import search

logger = logging.getLogger(__name__)


# ── Time helpers ──────────────────────────────────────────────────────────────

def _to_date(val) -> Optional[date]:
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:19] if len(s) >= 19 and fmt.endswith("%S") else s[:10], fmt).date()
        except ValueError:
            continue
    return None


def u_shape_time_score(
    tau_c: Optional[date],
    tau_star: Optional[date],
    tau_first: Optional[date],
) -> float:
    """U-shaped temporal relevance in [0,1] (paper Eq. 2). 0.5 if time unknown."""
    if tau_c is None or tau_star is None or tau_first is None:
        return 0.5
    recent = max(settings.ETHER_TAU_RECENT_DAYS, 1e-6)
    early = max(settings.ETHER_TAU_EARLY_DAYS, 1e-6)
    d_recent = (tau_star - tau_c).days       # ≥0 for past events
    d_early = (tau_c - tau_first).days        # ≥0
    s_recent = math.exp(-abs(d_recent) / recent)
    s_early = math.exp(-abs(d_early) / early)
    return max(s_recent, s_early)


# ── Textual path ──────────────────────────────────────────────────────────────

def _event_where(patient_key: Optional[str]) -> Optional[dict]:
    clauses = [{"doc_type": {"$eq": "event"}}]
    if patient_key:
        clauses.append({"patient_key": {"$eq": str(patient_key)}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_textual_evidence(
    query: str,
    patient_key: Optional[str] = None,
    tau_star: Optional[date] = None,
    k_final: Optional[int] = None,
    candidate_k: int = 40,
) -> list[dict]:
    """
    Retrieve textual event chunks and re-score with the U-shaped time weight.
    Returns the top-k_final chunks in chronological order, each annotated with
    ether_score / s_sem / s_time.
    """
    k_final = k_final or settings.ETHER_K_FINAL
    alpha = settings.ETHER_ALPHA

    candidates = search(query, where_filter=_event_where(patient_key), n_results=candidate_k)
    if not candidates:
        # Fall back to unfiltered semantic search (e.g. free-text docs / no events yet)
        candidates = search(query, where_filter=None, n_results=candidate_k)
    if not candidates:
        return []

    times = [_to_date(c["metadata"].get("event_time")) for c in candidates]
    times = [t for t in times if t is not None]
    tau_first = min(times) if times else None
    if tau_star is None:
        tau_star = max(times) if times else None

    for c in candidates:
        s_sem = float(c.get("relevance_score", 0.0))
        tau_c = _to_date(c["metadata"].get("event_time"))
        s_time = u_shape_time_score(tau_c, tau_star, tau_first)
        c["s_sem"] = s_sem
        c["s_time"] = s_time
        c["ether_score"] = alpha * s_sem + (1.0 - alpha) * s_time

    ranked = sorted(candidates, key=lambda c: c["ether_score"], reverse=True)[:k_final]
    # Final evidence is temporally ordered (paper §3.2)
    ranked.sort(key=lambda c: (_to_date(c["metadata"].get("event_time")) or date.min))
    return ranked


# ── Numeric path ──────────────────────────────────────────────────────────────

def _score_indicators(query: str, indicators: list[str]) -> list[tuple[str, float]]:
    """Rank indicator names by relevance to the query via CrossEncoder; fallback token overlap."""
    if not indicators:
        return []
    try:
        from rag.retriever import _get_reranker
        reranker = _get_reranker()
        if reranker is not None:
            scores = reranker.predict([(query, ind) for ind in indicators])
            return sorted(zip(indicators, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    except Exception as exc:
        logger.debug("Indicator reranker unavailable (%s) — token-overlap fallback", exc)

    q_tokens = set(query.lower().split())
    scored = []
    for ind in indicators:
        overlap = len(q_tokens & set(ind.lower().split()))
        scored.append((ind, float(overlap)))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def retrieve_numeric_evidence(
    query: str,
    patient_key: Optional[str] = None,
    tau_star: Optional[date] = None,
) -> dict[str, list[tuple]]:
    """
    Coarse-to-fine indicator selection over numeric trajectories.
    Returns {indicator: [(value, date), …]} with the N_recent most-recent values
    (≤ τ* when known) for the top N_fine indicators.
    """
    traj = numeric_trajectories(patient_key)
    if not traj:
        return {}

    ranked = _score_indicators(query, list(traj.keys()))
    coarse = [ind for ind, _ in ranked[: settings.ETHER_N_COARSE]]
    fine = coarse[: settings.ETHER_N_FINE]

    out: dict[str, list[tuple]] = {}
    for ind in fine:
        values = traj[ind]
        # keep only values at/before the prediction time when known
        if tau_star is not None:
            values = [(v, t) for (v, t) in values if (_to_date(t) is None or _to_date(t) <= tau_star)]
        # most-recent N_recent, returned chronologically
        values = sorted(values, key=lambda vt: (_to_date(vt[1]) or date.min))
        out[ind] = values[-settings.ETHER_N_RECENT:]
    return out


# ── Fused evidence bundle ─────────────────────────────────────────────────────

def gather_evidence(
    query: str,
    patient_key: Optional[str] = None,
    tau_star: Optional[date] = None,
    k_final: Optional[int] = None,
) -> dict:
    """
    Run both ETHER paths and return a fused evidence bundle:
      {
        "textual":      [chunk dict, …],
        "numeric":      {indicator: [(value, date), …]},
        "evidence_text": str,   # serialised for an LLM prompt
      }
    """
    textual = retrieve_textual_evidence(query, patient_key, tau_star, k_final)
    numeric = retrieve_numeric_evidence(query, patient_key, tau_star)

    lines = []
    if numeric:
        lines.append("## Numeric indicators (most recent values)")
        for ind, vals in numeric.items():
            series = ", ".join(
                f"{v}{(' @' + str(t)) if t else ''}" for v, t in vals
            )
            lines.append(f"- {ind}: {series}")
        lines.append("")
    if textual:
        lines.append("## Clinical events (chronological)")
        for c in textual:
            t = c["metadata"].get("event_time") or "?"
            lines.append(f"[{t}] {c['content']}")

    return {
        "textual": textual,
        "numeric": numeric,
        "evidence_text": "\n".join(lines).strip(),
    }
