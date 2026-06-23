"""
Main RAG retrieval pipeline — dual-track: SQL exact lookup + hybrid RAG.

Query paths
-----------
  Analytical path
    → Aggregate / trending queries (counts, totals, breakdowns by dimension)
    → LLM generates SQL from schema YAML → sqlglot AST check → execute on read-only PG

  SQL path  (sql / hybrid)
    → Precise factual questions routed to PostgreSQL by the query router
    → Results come from structured rows, never LLM-generated facts
    → raw_text (verbatim source) is quoted directly → zero hallucination
    → LLM only formats the presentation, not the facts

  RAG path  (rag / hybrid fallback)
    → Open-ended / summary questions answered by hybrid vector+BM25 search
    → CrossEncoder reranking (retrieve 20, rerank, keep chunks above score threshold)
    → LLM synthesises an answer from the surviving chunks
    → Hallucination checker validates grounding; failure → raw sources + hard message

Decision tree
-------------
  parse_intent (regex-only) → query_router.route() →
    'analytical' → sql_retriever.analytical_lookup() → format aggregate answer
    'sql'        → sql_retriever.lookup() → format verbatim answer
    'hybrid'     → sql first, then RAG if needed
    'rag'        → hybrid vector+BM25 → rerank → threshold-filter → LLM → grounding-check

Anti-hallucination measures
----------------------------
  1. SQL results are returned verbatim — the LLM cannot alter the facts
  2. CrossEncoder reranker + score threshold: off-topic chunks are dropped
     without any LLM call (replaces the old per-chunk LLM relevance grader)
  3. RAG answer prompt: "use ONLY the retrieved documents"
  4. Hallucination checker: verifies the answer is grounded in the docs
  5. Grounding failure: hard "insufficient grounding" message + raw sources
     (no warned passthrough — prevents confident but ungrounded answers)
  6. "No results" path: says "not found" rather than guessing
"""
import logging

from rag.intent_parser import parse_intent, build_where_filter
from rag.query_router import route
from rag.sql_retriever import lookup as sql_lookup, analytical_lookup
from rag.vectorstore import search, search_bm25
from rag.llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

# ── CrossEncoder reranker (lazy singleton) ────────────────────────────────────
_reranker = None


def _get_reranker():
    """Load the CrossEncoder once and cache it for the process lifetime."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.RERANKER_MODEL)
            logger.info("CrossEncoder loaded: %s", settings.RERANKER_MODEL)
        except Exception as exc:
            logger.warning("CrossEncoder unavailable (%s) — reranking disabled", exc)
    return _reranker


def _rerank(query: str, documents: list[dict], top_k: int) -> list[dict]:
    """
    Rerank `documents` using the CrossEncoder and return the top `top_k`.
    Falls back to score-ordered input if the model is unavailable.
    """
    if not documents:
        return documents

    reranker = _get_reranker()
    if reranker is None:
        return documents[:top_k]

    try:
        pairs  = [(query, doc["content"]) for doc in documents]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = []
        for score, doc in ranked[:top_k]:
            doc = dict(doc)
            doc["rerank_score"] = float(score)
            reranked.append(doc)
        logger.info(
            "Reranked %d → %d doc(s): top score %.3f",
            len(documents), len(reranked),
            reranked[0]["rerank_score"] if reranked else 0.0,
        )
        return reranked
    except Exception as exc:
        logger.warning("Reranking failed (%s) — using original order", exc)
        return documents[:top_k]

# ── CrossEncoder score-threshold filter ──────────────────────────────────────
# Replaces the old per-chunk LLM relevance grader.  After reranking, any chunk
# whose CrossEncoder score falls below RERANKER_SCORE_THRESHOLD is dropped.
# This eliminates O(N) sequential Ollama calls per query and gives a
# deterministic, sub-millisecond filter instead of an LLM yes/no judgment.

def _filter_by_threshold(documents: list[dict]) -> list[dict]:
    """
    Drop reranked documents whose CrossEncoder score is below the configured
    threshold.  Documents with no rerank_score (reranker was unavailable) are
    always kept so the pipeline degrades gracefully rather than returning nothing.

    If ALL reranked docs fall below the threshold, the top-1 doc is kept as a
    last-resort fallback — it is better to send the LLM the single best chunk
    than to produce a "no results" answer when relevant content exists.
    """
    threshold = settings.RERANKER_SCORE_THRESHOLD
    if threshold <= 0.0:
        # Threshold disabled — pass everything through
        return documents

    filtered = [
        doc for doc in documents
        if doc.get("rerank_score", threshold + 1.0) >= threshold
    ]

    if not filtered:
        logger.info(
            "Score threshold %.2f filtered all %d docs — keeping top-1 as fallback",
            threshold, len(documents),
        )
        return documents[:1] if documents else []

    logger.info(
        "Score threshold %.2f: %d/%d doc(s) passed",
        threshold, len(filtered), len(documents),
    )
    return filtered


# ── Hallucination checker ─────────────────────────────────────────────────────

_HALLUCINATION_GRADE_PROMPT = """\
You are checking whether a generated answer is grounded in the provided documents.

Documents:
{documents}

Generated answer:
{answer}

Is the answer grounded in and supported by the documents above?
Reply with ONLY "yes" or "no" — no explanation, no punctuation, nothing else."""


def _check_hallucination(answer: str, documents: list[dict]) -> bool:
    """
    Returns True if the answer appears grounded, False if it looks hallucinated.
    Falls back to True (pass) if the LLM call fails.
    """
    if not documents or not answer.strip():
        return True  # nothing to check

    docs_text = "\n\n".join(doc.get("content", "")[:500] for doc in documents[:5])
    try:
        result = call_llm(
            _HALLUCINATION_GRADE_PROMPT.format(
                documents=docs_text,
                answer=answer[:800],
            )
        )
        grounded = result.strip().lower().startswith("yes")
        if not grounded:
            logger.warning("Hallucination checker flagged the answer as ungrounded")
        return grounded
    except Exception as exc:
        logger.warning("Hallucination check failed (%s) — passing answer through", exc)
        return True


# ── Answer prompts ────────────────────────────────────────────────────────────

_GROUNDING_FAILURE_MESSAGE = (
    "⛔ Insufficient grounding — the retrieved documents do not adequately "
    "support a confident answer to this query.\n\n"
    "The raw source documents retrieved are shown in the 'documents' field. "
    "Please review them directly or rephrase your question to be more specific."
)

_ANALYTICAL_ANSWER_PROMPT = """\
You are a medical billing analyst presenting aggregate query results.

User question: {query}

Query results ({row_count} row(s)):
{results_text}

Instructions:
- Present the data clearly: use a table if there are multiple rows, or a sentence for a single value.
- Do NOT add any information not present in the results above.
- Round monetary amounts to 2 decimal places and prefix with $.
- If the result is empty, say "No data found for this query."

Answer:"""

_SQL_ANSWER_PROMPT = """\
You are a medical records assistant presenting exact information retrieved \
from a structured database.

User question: {query}

Retrieved records:
{records}

STRICT RULES:
- Present ONLY the information shown in the retrieved records above
- Quote values exactly as they appear — do not round, abbreviate, or paraphrase
- If multiple records match, list all of them clearly
- Do NOT add any information not present in the records above
- Format dates as DD-MM-YYYY
- Format monetary amounts with the $ sign

Answer:"""

_RAG_ANSWER_PROMPT = """\
You are a medical records assistant. Answer using ONLY the documents retrieved below.

User query: {query}

Retrieved documents:
{documents}

Instructions:
- If a specific field was requested (NPI, date of birth, total amount, etc.), highlight it clearly.
- If multiple documents match, present all relevant information.
- If no relevant documents are found, say "No matching records found."
- Be concise and factual. Never add information not in the documents.
- Format monetary amounts with the $ sign.
- Display dates as DD-MM-YYYY.

Answer:"""


# ── Public API ────────────────────────────────────────────────────────────────

def query(user_query: str) -> dict:
    """
    Full retrieval pipeline — called from the FastAPI /query endpoint.

    Returns:
        {
          "answer":      str  — human-readable answer,
          "documents":   list — source records with metadata + relevance score,
          "intent":      dict — structured criteria from the query,
          "filter_used": dict — the ChromaDB where-filter (if RAG path used),
          "query_path":  str  — 'sql', 'rag', or 'hybrid' (for debugging),
        }
    """
    logger.info("Query: %r", user_query)

    # ── Step 1: parse intent ──────────────────────────────────────────────────
    intent = parse_intent(user_query)
    logger.info("Intent: %s", intent)

    # ── Step 2: route query ───────────────────────────────────────────────────
    path = route(intent)

    # ── Step 2a: analytical path (LLM text-to-SQL) ───────────────────────────
    if path == "analytical":
        analytical_docs = analytical_lookup(user_query)
        if analytical_docs:
            answer = _format_analytical_answer(analytical_docs, user_query)
        else:
            answer = "No analytical data found for this query."
        return {
            "answer":        answer,
            "documents":     analytical_docs,
            "intent":        intent,
            "filter_used":   None,
            "query_path":    "analytical",
            "sql_generated": analytical_docs[0].get("sql_generated") if analytical_docs else None,
        }

    # ── Step 3a: SQL exact lookup ─────────────────────────────────────────────
    sql_documents = []
    if path in ("sql", "hybrid"):
        sql_documents = sql_lookup(intent)
        logger.info("SQL lookup returned %d result(s)", len(sql_documents))

    if sql_documents and path == "sql":
        # Pure SQL path — format verbatim answer
        answer = _format_sql_answer(sql_documents, user_query)
        return {
            "answer":      answer,
            "documents":   sql_documents,
            "intent":      intent,
            "filter_used": None,
            "query_path":  "sql",
        }

    # ── Step 3b: RAG path (hybrid search → rerank → grade) ───────────────────
    where_filter = build_where_filter(intent)
    has_entity   = bool(
        intent.get("patient_name") or intent.get("patient_id") or
        intent.get("provider_name") or intent.get("provider_npi") or
        intent.get("subject_id") or intent.get("hadm_id")
    )
    # Retrieve RERANKER_INITIAL_K candidates; reranker will trim to RERANKER_TOP_K
    n_results = settings.RERANKER_INITIAL_K

    rag_documents = _hybrid_search(user_query, where_filter, n_results)
    if not rag_documents and where_filter:
        # Only broaden to an unfiltered search if the filter was NOT tied to
        # a specific named entity (e.g. it was just a doc_type/date filter).
        # If the user asked about a specific patient/provider and nothing
        # matched, broadening to "everything" risks pulling in another
        # patient's records and the LLM hallucinating that they belong to
        # the requested person. In that case, report no results instead.
        entity_claimed = bool(
            (intent.get("patient_name") and len(intent["patient_name"].split()) >= 2) or
            intent.get("patient_id") or
            (intent.get("provider_name") and len(intent["provider_name"].split()) >= 2) or
            intent.get("provider_npi") or
            intent.get("subject_id") or
            intent.get("hadm_id")
        )

        # An "entity" the intent parser extracted is only trustworthy if SQL
        # actually found a matching patient/provider/admission for it. If
        # sql_documents is empty, the extracted ID isn't a real identifier in
        # this system — it might be an ICD code, a CPT code, a typo, or any
        # other token the LLM mistook for an ID. In that case don't let it
        # suppress the search: broaden to an unfiltered query so reference
        # material (e.g. coding-guideline PDFs) can still be found. This is
        # deliberately format-agnostic — it doesn't matter what kind of code
        # or ID the parser thought it saw.
        entity_filtered = entity_claimed and bool(sql_documents)

        if not entity_filtered:
            if entity_claimed:
                logger.info(
                    "Entity-like field(s) in intent (%s) were not confirmed "
                    "by SQL — broadening to an unfiltered search anyway",
                    {k: v for k, v in intent.items() if k in (
                        "patient_name", "patient_id", "provider_name",
                        "provider_npi", "subject_id", "hadm_id") and v},
                )
            else:
                logger.info("No RAG results with filter — retrying without filter")
            rag_documents = _hybrid_search(user_query, None, n_results)
        else:
            logger.info(
                "No RAG results for the specified patient/provider — "
                "not broadening to an unfiltered search to avoid mixing "
                "in other patients' records"
            )

    # ── Step 4: rerank + score-threshold filter RAG results ──────────────────
    if rag_documents and path != "sql":
        # CrossEncoder reranker: retrieve INITIAL_K → rerank → top TOP_K
        rag_documents = _rerank(user_query, rag_documents, settings.RERANKER_TOP_K)
        # Score threshold filter: drop chunks below RERANKER_SCORE_THRESHOLD
        # (replaces the old per-chunk LLM relevance grader — eliminates N
        # sequential Ollama calls and gives a deterministic filter instead)
        rag_documents = _filter_by_threshold(rag_documents)

    # ── Step 5: combine SQL + RAG for hybrid path ─────────────────────────────
    if path == "hybrid":
        # SQL results are exact — put them first, RAG enriches the answer
        all_documents = sql_documents + [
            d for d in rag_documents
            if d["id"] not in {s["id"] for s in sql_documents}
        ]
    else:
        all_documents = rag_documents

    # ── Step 7: generate answer ───────────────────────────────────────────────
    hallucination_flag = False

    if not all_documents:
        answer = (
            "No matching records found in the database. "
            "Please ensure the relevant files have been ingested, "
            "or try rephrasing your query."
        )
    elif sql_documents and path == "hybrid":
        # Have SQL grounding — use the more constrained SQL prompt
        answer = _format_sql_answer(sql_documents, user_query)
    else:
        # Pure RAG answer — runs through reranker → grader → LLM → hallucination check
        answer, hallucination_flag = _format_rag_answer(rag_documents, user_query)

    return {
        "answer":             answer,
        "documents":          all_documents,
        "intent":             intent,
        "filter_used":        where_filter,
        "query_path":         path,
        "hallucination_flag": hallucination_flag,
    }


# ── Answer formatters ─────────────────────────────────────────────────────────

def _format_analytical_answer(documents: list[dict], user_query: str) -> str:
    """
    Format an answer from analytical (text-to-SQL) results.
    The LLM presents the aggregate table; it cannot alter the values.
    """
    rows_text = "\n".join(doc["content"] for doc in documents)
    try:
        return call_llm(_ANALYTICAL_ANSWER_PROMPT.format(
            query=user_query,
            row_count=len(documents),
            results_text=rows_text,
        ))
    except Exception as exc:
        logger.warning("LLM analytical formatting failed: %s — using raw rows", exc)
        return f"Query results ({len(documents)} row(s)):\n\n{rows_text}"


def _format_sql_answer(documents: list[dict], user_query: str) -> str:
    """
    Format an answer from SQL results.
    The LLM presents the facts but cannot alter them — values are quoted verbatim.
    """
    records_text = "\n\n".join(
        f"Record {i + 1}:\n{doc['content']}"
        for i, doc in enumerate(documents)
    )

    try:
        return call_llm(_SQL_ANSWER_PROMPT.format(
            query=user_query,
            records=records_text,
        ))
    except Exception as exc:
        logger.warning("LLM formatting failed: %s — using raw content", exc)
        return records_text


def _format_rag_answer(documents: list[dict], user_query: str) -> tuple[str, bool]:
    """
    Format an answer from RAG (ChromaDB) results via LLM synthesis.
    Returns (answer_text, hallucination_flag).
    hallucination_flag is True when the checker flagged the answer as ungrounded.
    """
    docs_text = "\n\n".join(
        f"Document {i + 1} — {doc['metadata'].get('file_name', 'unknown')}"
        f"  (chunk {doc['metadata'].get('chunk_number', '?')})\n"
        f"Relevance: {doc.get('relevance_score', 0.0):.0%}\n"
        f"{doc['content'][:1500]}"
        for i, doc in enumerate(documents)
    )

    try:
        answer = call_llm(_RAG_ANSWER_PROMPT.format(
            query=user_query,
            documents=docs_text,
        ))
    except Exception as exc:
        logger.warning("LLM RAG answer failed: %s — using fallback", exc)
        return _fallback_answer(documents), False

    # Hallucination / grounding check
    grounded = _check_hallucination(answer, documents)
    if not grounded:
        # Hard failure: return the raw sources and a clear refusal instead of
        # a warned passthrough.  A confident but ungrounded answer is worse
        # than an honest "I cannot verify this" message.
        logger.warning("Grounding check failed — returning raw sources with failure message")
        return _GROUNDING_FAILURE_MESSAGE, True

    return answer, False


# ── Hybrid vector + BM25 search ───────────────────────────────────────────────

def _hybrid_search(
    query_text:   str,
    where_filter: dict,
    n_results:    int,
) -> list[dict]:
    alpha   = settings.HYBRID_SEARCH_ALPHA
    fetch_n = min(n_results * 3, 60)

    vector_results: list[dict] = []
    bm25_results:   list[dict] = []

    if alpha > 0.0:
        vector_results = search(query_text, where_filter=where_filter, n_results=fetch_n)
    if alpha < 1.0:
        bm25_results = search_bm25(query_text, where_filter=where_filter, n_results=fetch_n)

    if not bm25_results:
        return vector_results[:n_results]
    if not vector_results:
        return bm25_results[:n_results]

    return _reciprocal_rank_fusion(vector_results, bm25_results, n_results)


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results:   list[dict],
    n_results:      int,
    k:              int = 60,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    doc_map:    dict[str, dict]  = {}

    for rank, doc in enumerate(vector_results):
        did = doc["id"]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        doc_map[did] = doc

    for rank, doc in enumerate(bm25_results):
        did = doc["id"]
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        if did not in doc_map:
            doc_map[did] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    merged = []
    for did in sorted_ids[:n_results]:
        doc = dict(doc_map[did])
        doc["relevance_score"] = rrf_scores[did]
        merged.append(doc)
    return merged


# ── Plain-text fallback (no LLM) ──────────────────────────────────────────────

def _fallback_answer(documents: list) -> str:
    if not documents:
        return "No matching records found."
    lines = [f"Found {len(documents)} result(s):\n"]
    for i, doc in enumerate(documents, 1):
        m = doc.get("metadata", {})
        lines.append(f"── Result {i} ──────────────────")
        lines.append(doc.get("content", "")[:600])
        lines.append(f"Source: {m.get('file_name','?')} p{m.get('page_number','?')}\n")
    return "\n".join(lines)
