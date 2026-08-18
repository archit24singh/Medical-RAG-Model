"""
LangChain SQL Database Agent over Postgres.

Replaces the deterministic query shapes + custom text-to-SQL fallback: the agent
connects to Postgres via SQLAlchemy, introspects the catalog (tables, columns,
sample rows) itself, and runs a tool-using loop (list_tables → get_schema →
write query → check → run → retry) to answer structured questions. Powered by
the configured LLM (default: NVIDIA Nemotron 3 Ultra via OpenRouter, using
langchain-openai's OpenAI-compatible ChatOpenAI client; langchain-ollama is
used instead when LLM_PROVIDER=ollama).

Scope: exposed to the curated tables + auto-exposed v_* views only — the raw
stg_* staging tables and catalog internals are hidden. Executes on the
read-only DSN when configured.

Best-effort: any import/build/run failure returns None so the caller falls back
to the RAG path. Requires the local Ollama daemon to be signed into Ollama Cloud.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_AGENT = None
_BUILD_FAILED = False


def _readonly_uri() -> Optional[str]:
    dsn = settings.POSTGRES_READONLY_DSN or settings.POSTGRES_DSN
    if not dsn:
        return None
    # SQLAlchemy needs the psycopg2 driver in the URI.
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+psycopg2://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+psycopg2://", 1)
    return dsn


def _queryable_tables() -> Optional[list[str]]:
    """
    Tables/views exposed to the agent. When SQL_AGENT_EXPOSE_ALL_TABLES is True
    (current decision), return None so LangChain includes EVERY public table —
    incl. raw stg_* staging. Otherwise restrict to curated tables + v_* views.
    """
    if getattr(settings, "SQL_AGENT_EXPOSE_ALL_TABLES", True):
        return None  # None → SQLDatabase includes all tables
    from db.schema import get_readonly_db
    keep: list[str] = []
    with get_readonly_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' "
                "AND (table_name LIKE 'v\\_%%' "
                "     OR table_name IN ('patients','providers','records','admissions','documents'))"
            )
            keep = [r[0] for r in cur.fetchall()]
    return keep


def _agent_model_name() -> str:
    """Model id the agent runs on, for logs/health."""
    return settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.OPENAI_MODEL


def _build_llm():
    """
    LangChain chat model for the SQL agent.

    LLM_PROVIDER=openai → ChatOpenAI pointed at OPENAI_BASE_URL. This covers
    OpenRouter (default, serving NVIDIA Nemotron 3 Ultra), NVIDIA NIM and
    OpenAI itself, since all three speak the OpenAI chat-completions protocol
    (including the tool-calling schema the "tool-calling" agent type needs).
    LLM_PROVIDER=ollama → ChatOllama against the local daemon.
    """
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot build the SQL agent LLM")

    headers = {}
    if "openrouter" in (settings.OPENAI_BASE_URL or ""):
        if settings.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        if settings.OPENROUTER_APP_NAME:
            headers["X-Title"] = settings.OPENROUTER_APP_NAME

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        temperature=0,
        timeout=settings.OPENAI_TIMEOUT_S,
        max_retries=settings.OPENAI_MAX_RETRIES,
        default_headers=headers or None,
    )


def _build_agent():
    global _AGENT, _BUILD_FAILED
    if _AGENT is not None or _BUILD_FAILED:
        return _AGENT
    try:
        from langchain_community.utilities import SQLDatabase
        from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

        uri = _readonly_uri()
        if not uri:
            logger.warning("SQL agent: no Postgres DSN configured")
            _BUILD_FAILED = True
            return None

        include = _queryable_tables()
        db = SQLDatabase.from_uri(
            uri,
            include_tables=include or None,   # None → all public tables
            view_support=True,
            sample_rows_in_table_info=settings.SQL_AGENT_SAMPLE_ROWS,
        )
        llm = _build_llm()
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Preferred agent type, then robust fallbacks (LangChain API varies by
        # version / model tool-calling support).
        tried = []
        for atype in (settings.SQL_AGENT_TYPE, "openai-tools", "zero-shot-react-description"):
            if atype in tried:
                continue
            tried.append(atype)
            try:
                _AGENT = create_sql_agent(
                    llm=llm, toolkit=toolkit, agent_type=atype, verbose=False,
                    max_iterations=settings.SQL_AGENT_MAX_ITERATIONS,
                    handle_parsing_errors=True,
                )
                logger.info("LangChain SQL agent ready (model=%s, agent_type=%s, tables=%s)",
                            _agent_model_name(), atype, "all" if include is None else len(include))
                return _AGENT
            except Exception as exc:
                logger.info("SQL agent agent_type=%s unusable (%s)", atype, exc)

        logger.warning("SQL agent could not be built with any agent_type")
        _BUILD_FAILED = True
        return None
    except Exception as exc:
        logger.warning("SQL agent build failed (%s) — falling back to RAG", exc)
        _BUILD_FAILED = True
        return None


def answer(user_query: str) -> Optional[dict]:
    """
    Answer a structured question via the SQL agent. Returns a result dict
    (same shape as other paths) or None on any failure → caller falls back.
    """
    agent = _build_agent()
    if agent is None:
        return None
    try:
        out = agent.invoke({"input": user_query})
        text = out.get("output") if isinstance(out, dict) else str(out)
    except Exception as exc:
        logger.warning("SQL agent run failed (%s)", exc)
        return None
    if not text or not str(text).strip():
        return None
    low = str(text).lower()
    # The agent says this when it can't answer from the DB → let RAG try.
    if "i don't know" in low or "does not contain" in low or "no relevant" in low:
        return None
    return {
        "answer": str(text).strip(),
        "documents": [],
        "intent": {},
        "filter_used": None,
        "query_path": "sql_agent",
    }
