"""
Query router — decides whether a user query should be answered by:

  'sql'    — precise factual lookup from SQLite (zero hallucination)
             Used when the query names a specific entity AND asks for a
             specific fact or event (date, amount, DOB, NPI, medication …).
             Example: "What is Alice's visit date on 6 May 2020?"

  'rag'    — semantic / similarity search via ChromaDB + LLM synthesis
             Used for open-ended, summary, or exploratory questions where
             exact match is less important than relevance.
             Example: "Summarise Alice's medical history"

  'hybrid' — SQL first for grounding, RAG for enrichment
             Used when there is a named entity but no specific field.
             Example: "Show me everything about Alice Johnson"

Decision logic
--------------
The router works from the structured intent dict already produced by
intent_parser.parse_intent().  No additional LLM call is needed.

  has_entity   = patient_name / patient_id / provider_name / provider_npi present
  has_specific = specific_field OR date requested

  has_entity + has_specific  → sql
  has_entity only            → hybrid
  neither                    → rag
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
    Return 'sql', 'rag', or 'hybrid' based on the parsed intent.

    Args:
        intent: dict produced by intent_parser.parse_intent()

    Returns:
        One of 'sql', 'rag', 'hybrid'
    """
    has_entity = bool(
        intent.get("patient_name") or
        intent.get("patient_id") or
        intent.get("provider_name") or
        intent.get("provider_npi")
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
