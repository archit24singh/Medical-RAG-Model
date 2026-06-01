"""
Streamlit chat UI for the Medical RAG System.

Features:
  - Natural-language chat to query patient bills, records, and provider info
  - Sidebar: system status, ingest trigger, file upload, document index view
  - Source documents panel with relevance scores for each answer
  - "What we understood" header so users can see how their query was interpreted
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ── All helper functions (defined before any UI code) ─────────────────────────

def api(method: str, endpoint: str, **kwargs) -> dict | None:
    url = f"{BACKEND_URL}/{endpoint.lstrip('/')}"
    try:
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend. Is Docker running?")
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out.")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def relevance_label(score: float) -> str:
    pct = score * 100
    if pct >= 75:
        return f'<span class="relevance-high">🟢 {pct:.0f}%</span>'
    elif pct >= 50:
        return f'<span class="relevance-mid">🟡 {pct:.0f}%</span>'
    return f'<span class="relevance-low">🔴 {pct:.0f}%</span>'


def render_sources(documents: list) -> None:
    """Render the source documents panel with relevance scores."""
    if not documents:
        return
    with st.expander(f"📎 {len(documents)} source document(s)"):
        for i, doc in enumerate(documents, 1):
            m = doc.get("metadata", {})
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{i}. {m.get('file_name', 'Unknown')}**")
                tags = []
                if m.get("patient_name"):
                    tags.append(f"Patient: {m['patient_name']}")
                if m.get("provider_name"):
                    tags.append(f"Provider: {m['provider_name']}")
                if m.get("doc_type"):
                    tags.append(f"Type: {m['doc_type']}")
                if m.get("date"):
                    tags.append(f"Date: {m['date']}")
                if m.get("provider_npi"):
                    tags.append(f"NPI: {m['provider_npi']}")
                if m.get("total_amount"):
                    tags.append(f"Amount: ${m['total_amount']}")
                st.caption("  |  ".join(tags))
            with c2:
                score = doc.get("relevance_score", 0)
                st.markdown(relevance_label(score), unsafe_allow_html=True)

            if st.checkbox("View raw content", key=f"raw_{i}_{doc.get('id', i)}"):
                st.code(doc.get("content", "")[:1000], language=None)

            if i < len(documents):
                st.divider()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .understood-badge {
    background: #e8f4fd;
    border-left: 3px solid #1f77b4;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 0.85rem;
    margin-bottom: 8px;
  }
  .relevance-high { color: #2ca02c; font-weight: 600; }
  .relevance-mid  { color: #ff7f0e; font-weight: 600; }
  .relevance-low  { color: #d62728; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 Medical RAG")
    st.caption("Intelligent Medical Document Retrieval")
    st.divider()

    # System status
    st.subheader("System Status")
    col_r, col_s = st.columns([2, 1])
    with col_s:
        if st.button("🔄", help="Refresh status"):
            st.rerun()

    health = api("get", "/health") or {}
    if health.get("status") == "healthy":
        st.success("✅ Online")
        st.metric("Documents indexed", health.get("documents_indexed", 0))
        st.caption(
            f"LLM: **{health.get('llm_provider','?')}** / "
            f"`{health.get('llm_model','?')}`"
        )
    else:
        st.error("⚠️ Backend unavailable")
        if health.get("error"):
            st.caption(health["error"])

    st.divider()

    # Ingest
    st.subheader("📂 Ingest Documents")
    st.caption(
        "Place files in the `data/patients/` or `data/providers/` "
        "folder, then click **Ingest**."
    )
    if st.button("⚡ Ingest All Files", use_container_width=True, type="primary"):
        with st.spinner("Indexing files…"):
            res = api("post", "/ingest")
        if res:
            st.success(f"✅ {res['success_count']} file(s) indexed")
            if res["error_count"]:
                st.warning(f"⚠️ {res['error_count']} error(s) — check backend logs")

    st.caption("Or upload a file directly:")
    upload = st.file_uploader(
        "Upload",
        type=["pdf", "json", "csv", "xlsx", "txt"],
        label_visibility="collapsed",
    )
    if upload:
        if st.button("📤 Upload & Ingest", use_container_width=True):
            with st.spinner(f"Ingesting {upload.name}…"):
                res = api(
                    "post", "/ingest/upload",
                    files={"file": (upload.name, upload.getvalue(), upload.type)},
                )
            if res and res.get("status") == "success":
                st.success(f"✅ {upload.name} ingested")
                m = res.get("metadata", {})
                if m.get("patient_name"):
                    st.caption(f"Patient: {m['patient_name']}")
                if m.get("doc_type"):
                    st.caption(f"Type: {m['doc_type']}")

    st.divider()

    # Document index
    st.subheader("📋 Indexed Documents")
    if st.button("View Index", use_container_width=True):
        with st.spinner("Loading…"):
            res = api("get", "/documents")
        if res:
            if res["total"] == 0:
                st.info("No documents indexed yet.")
            else:
                for doc in res["documents"]:
                    m = doc.get("metadata", {})
                    label = m.get("file_name") or doc["id"][:10]
                    with st.expander(f"📄 {label}"):
                        if m.get("patient_name"):
                            st.write(f"**Patient:** {m['patient_name']}")
                        if m.get("provider_name"):
                            st.write(f"**Provider:** {m['provider_name']}")
                        if m.get("doc_type"):
                            st.write(f"**Type:** {m['doc_type']}")
                        if m.get("date"):
                            st.write(f"**Date:** {m['date']}")
                        if m.get("provider_npi"):
                            st.write(f"**NPI:** {m['provider_npi']}")
                        st.caption(f"ID: `{doc['id']}`")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🏥 Medical Document Retrieval")
st.caption(
    "Ask about patient bills, medical records, or provider information "
    "in plain English."
)

# Example queries
with st.expander("💡 Example queries — click to use", expanded=False):
    examples = [
        "Get patient Alice Johnson's bill for 27-10-2025",
        "What is the NPI number for Dr. Robert Chen?",
        "Show me all records for patient P001",
        "What is the total amount on Alice's latest bill?",
        "Get provider information for NPI 9876543210",
        "What is the date of birth for Dr. Sarah Williams?",
    ]
    for ex in examples:
        if st.button(f"→  {ex}", key=f"ex_{ex}"):
            st.session_state["prefill"] = ex


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("documents"):
            render_sources(msg["documents"])


# ── Chat input ────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about a patient bill, record, or provider…")

if prefill and not user_input:
    user_input = prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical records…"):
            result = api("post", "/query", json={"query": user_input})

        if result:
            answer    = result.get("answer", "No answer generated.")
            documents = result.get("documents", [])
            intent    = result.get("intent", {})

            # ── "Understood" badge ────────────────────────────────────────────
            INTENT_LABELS = {
                "patient_name":   "Patient",
                "patient_id":     "ID",
                "provider_name":  "Provider",
                "provider_npi":   "NPI",
                "provider_dob":   "Provider DOB",
                "date":           "Date",
                "doc_type":       "Type",
                "specific_field": "Looking for",
                "query_type":     None,  # internal — don't surface
            }
            tags = []
            for field, label in INTENT_LABELS.items():
                val = intent.get(field)
                if val and label:
                    tags.append(f"{label}: **{val}**")

            if tags:
                st.markdown(
                    '<div class="understood-badge">🔍 Understood: '
                    + "  |  ".join(tags) + "</div>",
                    unsafe_allow_html=True,
                )
            elif not documents:
                st.info(
                    "ℹ️ Could not match your query to any document in the database. "
                    "Try rephrasing, or check that the relevant files have been ingested."
                )

            st.markdown(answer)
            render_sources(documents)

            st.session_state.messages.append({
                "role":      "assistant",
                "content":   answer,
                "documents": documents,
                "intent":    intent,
            })
        else:
            err = "❌ Could not reach the backend. Check that Docker is running."
            st.error(err)
            st.session_state.messages.append({
                "role": "assistant", "content": err, "documents": [],
            })

# ── Clear chat ────────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
