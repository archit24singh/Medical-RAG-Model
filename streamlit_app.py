"""
Streamlit Community Cloud entry point for the Medical RAG demo.

This is a single-process deployment: no FastAPI, no PostgreSQL, no separate
ChromaDB service. It loads the pre-built (baked) Chroma index committed in the
repo, embeds the query + reranks in-process on CPU, and calls Groq (with an
OpenRouter fallback) for the answer.

Configuration comes from Streamlit secrets (Settings → Secrets) plus the
RAG-only defaults set below. See DEPLOY_STREAMLIT.md.
"""
import os
import sys

import streamlit as st

st.set_page_config(
    page_title="ICD-10 RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")


def _configure_environment() -> None:
    """
    Set runtime config BEFORE importing the backend (pydantic Settings reads
    environment variables at import time).

    Defaults put the app in pure-RAG mode with in-process embeddings and the
    embedded (baked) Chroma index. API keys and any overrides come from
    Streamlit secrets.
    """
    os.environ.setdefault("RAG_ONLY_MODE",       "true")
    os.environ.setdefault("LLM_PROVIDER",        "openai")
    os.environ.setdefault("EMBEDDING_BACKEND",   "sentence-transformers")
    os.environ.setdefault("CHROMA_MODE",         "persistent")
    os.environ.setdefault("CHROMA_PERSIST_DIR",  os.path.join(BASE_DIR, "chroma_index"))
    os.environ.setdefault("CHROMA_COLLECTION",   "medical_docs")
    os.environ.setdefault("AUTO_INGEST",         "false")   # index is pre-built
    # Save Groq quota on the demo: one LLM call per question, not two.
    os.environ.setdefault("ENABLE_HALLUCINATION_CHECK", "false")

    # Copy any secrets the user set into the environment so config picks them up.
    # (GROQ_API_KEY is required; the rest are optional overrides.)
    passthrough = [
        "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
        "LLM_FALLBACK_ENABLED", "ENABLE_HALLUCINATION_CHECK",
        "ST_EMBEDDING_MODEL", "RERANKER_TOP_K", "RERANKER_SCORE_THRESHOLD",
    ]
    try:
        for key in passthrough:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # st.secrets raises if no secrets file exists — fine for local runs.
        pass


_configure_environment()
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@st.cache_resource(show_spinner="Loading models and vector index (first load only)…")
def _load_backend():
    """Import the retriever and warm the collection once per container."""
    from rag.retriever import query as rag_query
    from rag.vectorstore import get_document_count
    count = get_document_count()   # triggers embedder + index load
    return rag_query, count


def _has_groq_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ICD-10 RAG")
    st.caption("Ask questions about the ICD-10-CM FY25 coding guidelines.")
    st.divider()
    if not _has_groq_key():
        st.error(
            "No API key configured. Add **GROQ_API_KEY** in the app's "
            "Settings → Secrets, then rerun."
        )
    else:
        st.success("✅ Ready")
    st.caption(
        "Pre-ingested demo — the document set is fixed and uploads are disabled."
    )

# ── Load backend (models + index) ─────────────────────────────────────────────
try:
    rag_query, doc_count = _load_backend()
    with st.sidebar:
        st.metric("Chunks indexed", doc_count)
except Exception as exc:
    st.error(f"Failed to load the vector index / models: {exc}")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 ICD-10 Coding Assistant")
st.caption("Ask about ICD-10-CM codes and guidelines in plain English.")

with st.expander("💡 Example questions", expanded=False):
    EXAMPLES = [
        "What is the Z code for acquired absence of limb?",
        "What does Z12 mean?",
        "What is the code for hypertensive heart and chronic kidney disease?",
    ]
    for ex in EXAMPLES:
        if st.button(f"→  {ex}", key=f"ex_{ex}"):
            st.session_state["prefill"] = ex


def _render_sources(documents: list) -> None:
    if not documents:
        return
    with st.expander(f"📎 {len(documents)} source passage(s)"):
        for i, doc in enumerate(documents, 1):
            m = doc.get("metadata", {})
            score = doc.get("rerank_score")
            score_txt = f" · relevance {score:.2f}" if isinstance(score, (int, float)) else ""
            st.markdown(
                f"**{i}. {m.get('file_name', 'document')}** "
                f"(page {m.get('page_number', '?')}{score_txt})"
            )
            st.caption(doc.get("content", "")[:600] + "…")
            if i < len(documents):
                st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("documents"):
            _render_sources(msg["documents"])

# ── Input ─────────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about an ICD-10 code or guideline…") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the guidelines…"):
            try:
                result = rag_query(user_input)
            except Exception as exc:
                result = None
                st.error(f"Query failed: {exc}")

        if result:
            answer    = result.get("answer", "No answer generated.")
            documents = result.get("documents", [])
            st.markdown(answer)
            _render_sources(documents)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "documents": documents,
            })

if st.session_state.messages and st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()
