"""
Query router — decides which path answers a user query:

  'analytical' — text-to-SQL via LLM for aggregate / trend questions
                 Checked FIRST.  Triggered when the intent has is_analytical=True.
                 Example: "How many claims by payer this quarter?"
                 Example: "Top 10 ICD-9 codes by frequency"

  'sql'        — precise factual lookup (zero hallucination)
                 Used when the query names a specific entity AND requests a
                 specific fact or event (date, amount, DOB, NPI, medication …).
                 Example: "What is Alice Johnson's total bill for 6 May 2025?"

  'hybrid'     — SQL first for grounding, RAG for enrichment
                 Used when there is a named entity but no specific field.
                 Example: "Show me everything about Alice Johnson"

  'rag'        — semantic / similarity search via ChromaDB + LLM synthesis
                 Used for open-ended, summary, or exploratory questions.
                 Example: "What ICD-10 code covers essential hypertension?"

Decision logic
--------------
The router works from the structured intent dict produced by
intent_parser.parse_intent().  No additional LLM call is needed.

  is_analytical                → analytical  (checked first)
  has_entity + has_specific    → sql
  has_entity only              → hybrid
  neither                      → rag
"""

import logging

logger = logging.getLogger(__name__)

# Intent fields that indicate a specific fact is being asked for
_SPECIFIC_FIELD_INDICATORS = {
    "specific_field",   # already parsed by intent_parser ("NPI number", "date of birth" …)
    "date",             # asking about a specific date
}

# doc_types that map to precise record lookups (as opposed to open retrieval)
_EXACT_DOC_TYPES = {"bill", "prescription", "lab_result", "provider_info"}


def route(intent: dict) -> str:
    """
    Return 'analytical', 'sql', 'hybrid', or 'rag' based on the parsed intent.

    Args:
        intent: dict produced by intent_parser.parse_intent()

    Returns:
        One of 'analytical', 'sql', 'hybrid', 'rag'
    """
    # ── Analytical (checked first) ────────────────────────────────────────────
    # Aggregate / trending queries bypass the exact-lookup and RAG paths and
    # go straight to LLM text-to-SQL generation.
    if intent.get("is_analytical"):
        logger.info(
            "Query router → analytical  (is_analytical=True, doc_type=%s)",
            intent.get("doc_type"),
        )
        return "analytical"

    # ── Entity-based exact / hybrid lookups ───────────────────────────────────
    has_entity = bool(
        intent.get("patient_name") or
        intent.get("patient_id") or
        intent.get("provider_name") or
        intent.get("provider_npi") or
        intent.get("subject_id") or
        intent.get("hadm_id")
    )

    has_specific = bool(
        intent.get("specific_field") or
        intent.get("date") or
        intent.get("doc_type") in _EXACT_DOC_TYPES
    )

    if has_entity and has_specific:
        decision = "sql"
    elif has_entity:
        decision = "hybrid"
    else:
        decision = "rag"

    logger.info(
        "Query router → %s  (entity=%s, specific=%s, doc_type=%s)",
        decision, has_entity, has_specific, intent.get("doc_type"),
    )
    return decision
