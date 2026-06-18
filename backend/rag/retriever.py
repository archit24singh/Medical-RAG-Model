"""
Main RAG retrieval pipeline — dual-track: SQL exact lookup + hybrid RAG.

Query paths
-----------
  SQL path  (sql / hybrid)
    → Precise factual questions routed to SQLite by the query router
    → Results come from structured rows, never LLM-generated facts
    → raw_text (verbatim source) is quoted directly → zero hallucination
    → LLM only formats the presentation, not the facts

  RAG path  (rag / hybrid fallback)
    → Open-ended / summary questions answered by hybrid vector+BM25 search
    → CrossEncoder reranking (retrieve 20, rerank, use top 5)
    → Relevance grader (binary yes/no per chunk, removes irrelevant docs)
    → LLM synthesises an answer from the surviving relevant chunks
    → Hallucination checker validates the final answer

Decision tree
-------------
  parse_intent (regex-only) → query_router.route() →
    'sql'    → sql_retriever.lookup() → format verbatim answer
    'hybrid' → sql first, then RAG if needed
    'rag'    → hybrid vector+BM25 → rerank → grade → LLM → hallucination-check

Anti-hallucination measures
----------------------------
  1. SQL results are returned verbatim — the LLM cannot alter the facts
  2. CrossEncoder reranker: only the 5 most relevant chunks reach the LLM
  3. Relevance grader: LLM yes/no filter removes off-topic chunks
  4. RAG answer prompt: "use ONLY the retrieved documents"
  5. Hallucination checker: verifies the answer is grounded in the docs
  6. "No results" path: says "not found" rather than guessing
"""
import logging

from rag.intent_parser import parse_intent, build_where_filter
from rag.query_router import route
from rag.sql_retriever import lookup as sql_lookup
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

# ── Relevance grader ─────────────────────────────────────────────────────────

_RELEVANCE_GRADE_PROMPT = """\
You are grading whether a retrieved document is relevant to the user's question.

User question: {query}

Retrieved document:
{document}

Is this document relevant to the question? Reply with ONLY "yes" or "no" — \
no explanation, no punctuation, nothing else."""


def _grade_relevance(query: str, document: dict) -> bool:
    """
    Binary relevance check for one retrieved chunk.
    Returns True (relevant) or False (irrelevant).
    Falls back to True (keep) if the LLM call fails.
    """
    try:
        answer = call_llm(
            _RELEVANCE_GRADE_PROMPT.format(
                query=query,
                document=document.get("content", "")[:800],
            )
        )
        return answer.strip().lower().startswith("yes")
    except Exception as exc:
        logger.warning("Relevance grading failed (%s) — keeping document", exc)
        return True  # keep on error


def _filter_relevant(query: str, documents: list[dict]) -> list[dict]:
    """
    Run the relevance grader over all documents and return only the relevant ones.
    If all are filtered out, return the original list (safety fallback).
    """
    if not documents:
        return documents

    relevant = [doc for doc in documents if _grade_relevance(query, doc)]
    if not relevant:
        logger.info("Relevance grader removed all docs — using originals as fallback")
        return documents  # don't leave the LLM with nothing
    logger.info("Relevance grader: %d/%d doc(s) kept", len(relevant), len(documents))
    return relevant


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

    # ── Step 4: rerank + relevance-grade RAG results ─────────────────────────
    if rag_documents and path != "sql":
        # CrossEncoder reranker: retrieve INITIAL_K → rerank → top TOP_K
        rag_documents = _rerank(user_query, rag_documents, settings.RERANKER_TOP_K)
        # Relevance grader: binary yes/no filter (from reliable_rag pattern)
        rag_documents = _filter_relevant(user_query, rag_documents)

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

    # Hallucination check
    grounded = _check_hallucination(answer, documents)
    if not grounded:
        logger.warning("Hallucination detected — appending disclaimer to answer")
        answer = (
            answer
            + "\n\n⚠️ *Note: this answer may not be fully supported by the "
            "retrieved documents — please verify against the source material.*"
        )

    return answer, not grounded


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
