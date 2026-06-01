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
    → LLM synthesises an answer from retrieved chunks
    → Used when SQL returns no results or question is exploratory

Decision tree
-------------
  parse_intent → query_router.route() →
    'sql'    → sql_retriever.lookup() → format verbatim answer
    'hybrid' → sql first, then RAG if needed
    'rag'    → hybrid vector+BM25 search → LLM synthesis

Anti-hallucination measures
----------------------------
  1. SQL results are returned verbatim — the LLM cannot alter the facts
  2. RAG answer prompt instructs the LLM to use ONLY retrieved documents
  3. If neither source returns results, the system says "not found"
     rather than guessing
"""
import logging

from rag.intent_parser import parse_intent, build_where_filter
from rag.query_router import route
from rag.sql_retriever import lookup as sql_lookup
from rag.vectorstore import search, search_bm25
from rag.llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

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

    # ── Step 3b: RAG path (hybrid search) ─────────────────────────────────────
    where_filter = build_where_filter(intent)
    has_entity   = bool(
        intent.get("patient_name") or intent.get("patient_id") or
        intent.get("provider_name") or intent.get("provider_npi")
    )
    n_results = 20 if has_entity else 8

    rag_documents = _hybrid_search(user_query, where_filter, n_results)
    if not rag_documents and where_filter:
        logger.info("No RAG results with filter — retrying without filter")
        rag_documents = _hybrid_search(user_query, None, n_results)

    # ── Step 4: combine SQL + RAG for hybrid path ─────────────────────────────
    if path == "hybrid":
        # SQL results are exact — put them first, RAG enriches the answer
        all_documents = sql_documents + [
            d for d in rag_documents
            if d["id"] not in {s["id"] for s in sql_documents}
        ]
    else:
        all_documents = rag_documents

    # ── Step 5: generate answer ───────────────────────────────────────────────
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
        # Pure RAG answer
        answer = _format_rag_answer(rag_documents, user_query)

    return {
        "answer":      answer,
        "documents":   all_documents,
        "intent":      intent,
        "filter_used": where_filter,
        "query_path":  path,
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


def _format_rag_answer(documents: list[dict], user_query: str) -> str:
    """Format an answer from RAG (ChromaDB) results via LLM synthesis."""
    docs_text = "\n\n".join(
        f"Document {i + 1} — {doc['metadata'].get('file_name', 'unknown')}"
        f"  (page {doc['metadata'].get('page_number', '?')},"
        f" chunk {doc['metadata'].get('chunk_number', '?')})\n"
        f"Relevance: {doc['relevance_score']:.0%}\n"
        f"{doc['content'][:1500]}"
        for i, doc in enumerate(documents)
    )

    try:
        return call_llm(_RAG_ANSWER_PROMPT.format(
            query=user_query,
            documents=docs_text,
        ))
    except Exception as exc:
        logger.warning("LLM RAG answer failed: %s — using fallback", exc)
        return _fallback_answer(documents)


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
