"""
Contextual enricher — prepends situating context to every chunk before storage.

Inspired by Anthropic's "Contextual Retrieval" blog post and the identical
strategy presented by Microsoft Research.

Why it works
------------
  A raw chunk like "Total due: $1,250.00" is ambiguous — who is the patient?
  what bill? what date? With contextual enrichment the stored text becomes:

    "This chunk is from Alice Johnson's hospital bill dated 2025-10-27.
     It appears in the billing summary section and lists the final amount owed.
     The bill was issued by City Medical Center for inpatient services."

    Total due: $1,250.00

  When the user asks "What is the total on Alice's bill?", the enriched chunk
  is much more likely to surface because the context echoes the query terms.

Implementation
--------------
  For each chunk:
    1. Build a document excerpt (first 3 000 chars of the full document).
    2. Send (excerpt + chunk) to Mistral and ask for 2-3 context sentences.
    3. Prepend those sentences to the chunk text.

  If the LLM call fails for any chunk, that chunk is stored without context
  so the pipeline never stalls.

Output
------
  Each returned chunk dict has two extra keys:
    "original_text" : str   — the chunk before enrichment
    "has_context"   : bool  — whether context was successfully generated
"""

import logging

from rag.llm_client import call_llm

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """\
You are a medical document analyst. Write exactly 2-3 concise sentences that \
situate the following chunk within its source document.

Cover all of these that are relevant:
- Which section or topic the chunk belongs to (e.g. billing summary, lab results, diagnosis)
- What specific information it contains
- The patient's name, date, or provider if they can be inferred from context

<document_excerpt>
{document_excerpt}
</document_excerpt>

<chunk>
{chunk_text}
</chunk>

Write ONLY the 2-3 context sentences. No labels, no preamble, no explanation."""


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_chunks(chunks: list[dict], pages: list[dict], filename: str = "") -> list[dict]:
    """
    Prepend contextual sentences to each chunk.

    Args:
        chunks:   Output of chunker.chunk_pages().
        pages:    Raw pages from document_parser (used to build the document excerpt).
        filename: Used only for log messages.

    Returns:
        The same list of chunk dicts with 'text' prepended with context,
        plus 'original_text' and 'has_context' keys added.
    """
    if not chunks:
        return []

    # Build a document excerpt from the full text (capped at 3 000 chars)
    full_text    = "\n\n".join(p["text"] for p in pages)
    doc_excerpt  = full_text[:3000]
    if len(full_text) > 3000:
        doc_excerpt += "\n...[document continues]"

    enriched = []
    total    = len(chunks)

    for i, chunk in enumerate(chunks):
        try:
            context = _generate_context(doc_excerpt, chunk["text"])
            enriched_text = f"{context}\n\n{chunk['text']}"
            enriched.append({
                **chunk,
                "text":          enriched_text,
                "original_text": chunk["text"],
                "has_context":   True,
            })
            logger.debug("Enriched chunk %d/%d for '%s'", i + 1, total, filename)

        except Exception as exc:
            logger.warning(
                "Context generation failed for chunk %d/%d of '%s': %s — storing plain",
                i + 1, total, filename, exc,
            )
            enriched.append({
                **chunk,
                "original_text": chunk["text"],
                "has_context":   False,
            })

    return enriched


# ── Internal ──────────────────────────────────────────────────────────────────

def _generate_context(document_excerpt: str, chunk_text: str) -> str:
    """Call the LLM and return 2-3 context sentences for one chunk."""
    prompt = _CONTEXT_PROMPT.format(
        document_excerpt=document_excerpt,
        # Cap the chunk passed to the LLM to avoid overflowing the context window
        chunk_text=chunk_text[:1200],
    )
    response = call_llm(prompt).strip()

    # Strip any accidental label prefixes the LLM might add
    for prefix in ("context:", "Context:", "CONTEXT:"):
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    return response
