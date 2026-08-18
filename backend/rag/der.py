"""
DER — Dual-Path Evidence Retrieval and Reasoning (paper §3.4).

Reasoning along a single evidence pathway is prone to confirmation bias. DER
retrieves and reasons along TWO complementary paths and fuses them:

  q+  (factual)        — seeks evidence supporting the target outcome
  q-  (counterfactual) — seeks evidence supporting its absence

Each path is expanded with AIR, yielding E+ and E-. The shared numeric evidence
E_num is reused. The model forms a positive hypothesis h+ from (E+, E_num) and a
negative hypothesis h- from (E-, E_num), the evidence is fused
(E_fuse = E+ ∪ E- ∪ E_num), and a final comparative decision weighs the strength
and directness of evidence for each hypothesis to produce the prediction.

This is what turns the system from retrieval-Q/A into a *predictive* model: the
output is a class label + rationale, grounded in dual-path evidence.

All LLM steps use JSON mode and degrade gracefully on a local model.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from config import settings
from rag.llm_client import call_llm_json
from rag.air import adaptive_retrieve, _merge_textual, _serialize

logger = logging.getLogger(__name__)


# ── Query construction ────────────────────────────────────────────────────────

def _factual_query(task: str, outcome: str) -> str:
    return f"{task} Evidence indicating the patient {outcome} (supporting a positive outcome)."


def _counterfactual_query(task: str, outcome: str) -> str:
    return f"{task} Evidence indicating the patient does NOT {outcome} (supporting a negative outcome)."


# ── Hypothesis + decision prompts ─────────────────────────────────────────────

_HYPOTHESIS_SYSTEM = (
    "You are a clinical reasoner. From the evidence, state a concise hypothesis "
    "for the given stance and cite the strongest supporting findings. JSON only."
)

_HYPOTHESIS_PROMPT = """\
Prediction task:
{task}

Stance to argue: {stance}

Evidence:
{evidence}

Give a brief hypothesis for this stance with its strongest support.
Reply with JSON exactly: {{"hypothesis": "<1-3 sentences>", "support": "<key findings>", "strength": "weak|moderate|strong"}}"""

_DECISION_SYSTEM = (
    "You are a clinical decision model. Compare the positive and negative "
    "hypotheses against the fused evidence and output a final prediction. "
    "Weigh the strength, directness, and clinical relevance of each side. "
    "JSON only."
)

_DECISION_PROMPT = """\
Prediction task:
{task}

Allowed labels: {labels}

Positive hypothesis (outcome present):
{h_pos}

Negative hypothesis (outcome absent):
{h_neg}

Fused evidence:
{evidence}

Compare both hypotheses against the evidence and decide.
Reply with JSON exactly: {{"prediction": "<one of the allowed labels>", "confidence": 0.0-1.0, "rationale": "<why, citing the deciding evidence>"}}"""


def predict(
    task_query: str,
    outcome_phrase: str,
    labels: list[str],
    patient_key: Optional[str] = None,
    tau_star: Optional[date] = None,
) -> dict:
    """
    Run the full DER prediction.

    Args:
        task_query:      the clinical prediction task (natural language).
        outcome_phrase:  the positive outcome, e.g. "is readmitted within 30 days".
        labels:          allowed output labels, e.g. ["readmitted", "not readmitted"].
        patient_key:     restrict evidence to one patient (recommended).
        tau_star:        prediction time anchor.

    Returns:
        {
          "prediction": str, "confidence": float, "rationale": str,
          "hypotheses": {"positive": {...}, "negative": {...}},
          "evidence_text": str,
          "paths": {"factual_queries": [...], "counterfactual_queries": [...]},
        }
    """
    # ── Dual-path retrieval (AIR on each path) ────────────────────────────────
    pos = adaptive_retrieve(_factual_query(task_query, outcome_phrase), patient_key, tau_star)

    if settings.ENABLE_DER:
        neg = adaptive_retrieve(_counterfactual_query(task_query, outcome_phrase), patient_key, tau_star)
    else:
        neg = {"textual": [], "numeric": {}, "queries": []}

    # ── Hypotheses ────────────────────────────────────────────────────────────
    h_pos = call_llm_json(
        _HYPOTHESIS_PROMPT.format(task=task_query, stance="outcome IS present", evidence=pos["evidence_text"] or "(none)"),
        system=_HYPOTHESIS_SYSTEM,
        default={"hypothesis": "", "support": "", "strength": "weak"},
    )
    h_neg = call_llm_json(
        _HYPOTHESIS_PROMPT.format(task=task_query, stance="outcome is ABSENT", evidence=neg["evidence_text"] or "(none)"),
        system=_HYPOTHESIS_SYSTEM,
        default={"hypothesis": "", "support": "", "strength": "weak"},
    )

    # ── Evidence fusion (E_fuse = E+ ∪ E- ∪ E_num) ────────────────────────────
    cap = settings.AIR_MAX_EVIDENCE_CHUNKS
    fused_textual = _merge_textual(pos["textual"], neg["textual"], cap)
    fused_numeric = dict(pos["numeric"])
    for ind, vals in neg["numeric"].items():
        fused_numeric.setdefault(ind, vals)
    fused_text = _serialize(fused_textual, fused_numeric)

    # ── Comparative decision ──────────────────────────────────────────────────
    decision = call_llm_json(
        _DECISION_PROMPT.format(
            task=task_query,
            labels=", ".join(labels),
            h_pos=h_pos.get("hypothesis", ""),
            h_neg=h_neg.get("hypothesis", ""),
            evidence=fused_text or "(none)",
        ),
        system=_DECISION_SYSTEM,
        default={"prediction": labels[0] if labels else "unknown", "confidence": 0.0,
                 "rationale": "Insufficient evidence for a confident prediction."},
    )

    pred = decision.get("prediction", "")
    # Snap to nearest allowed label (local models sometimes paraphrase)
    if labels and pred not in labels:
        pred = _snap_label(pred, labels)

    return {
        "prediction": pred,
        "confidence": _safe_float(decision.get("confidence"), 0.0),
        "rationale": decision.get("rationale", ""),
        "hypotheses": {"positive": h_pos, "negative": h_neg},
        "evidence_text": fused_text,
        "paths": {
            "factual_queries": pos.get("queries", []),
            "counterfactual_queries": neg.get("queries", []),
        },
    }


def _snap_label(pred: str, labels: list[str]) -> str:
    p = (pred or "").strip().lower()
    # Exact match first
    for lab in labels:
        if lab.lower() == p:
            return lab
    # Substring match, preferring the longest (most specific) label so that
    # e.g. "not readmitted" wins over "readmitted" when both appear.
    for lab in sorted(labels, key=len, reverse=True):
        if lab.lower() in p or p in lab.lower():
            return lab
    return labels[0]


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
