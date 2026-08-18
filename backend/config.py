"""
Central configuration — all values come from environment variables / .env file.
"""
import logging
from pydantic_settings import BaseSettings
from typing import Literal, Optional

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # ── ChromaDB (vector store — unstructured knowledge only) ─────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "medical_docs"
    # Client mode: "http" connects to a separate ChromaDB service (local Docker);
    # "persistent" runs Chroma embedded in-process from a local directory — used
    # for the single-container Streamlit demo (no separate Chroma service).
    CHROMA_MODE: str = "http"
    # Directory for the embedded (persistent) Chroma index. In the Streamlit demo
    # this points at the pre-built index committed in the repo.
    CHROMA_PERSIST_DIR: str = "chroma_index"

    # ── Embedding model — served by Ollama ────────────────────────────────────
    # nomic-embed-text: 8192-token context window.
    # Pull once: ollama pull nomic-embed-text
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Embedding backend: "ollama" (nomic-embed-text via Ollama, local dev) or
    # "sentence-transformers" (in-process CPU model, used on Hugging Face /
    # Streamlit where Ollama is unavailable). The index MUST be built with the
    # same backend + model that queries use, or similarity search breaks.
    EMBEDDING_BACKEND: str = "ollama"
    # sentence-transformers model used when EMBEDDING_BACKEND="sentence-transformers".
    # bge-small-en-v1.5: 384-dim, ~130 MB, strong quality, CPU/memory-friendly —
    # a good fit for the ~1 GB Streamlit Community Cloud memory ceiling.
    ST_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # "ollama" → fully local generation (requires Ollama on host)
    # "openai" → any OpenAI-compatible HTTP API (OpenRouter, NVIDIA NIM,
    #            OpenAI itself). Requires OPENAI_API_KEY + OPENAI_BASE_URL.
    # NOTE: embeddings ALWAYS go through Ollama (EMBEDDING_MODEL below),
    #       regardless of this setting — the vector store dimensions depend
    #       on nomic-embed-text, so switching them would require a re-ingest.
    LLM_PROVIDER: Literal["ollama", "openai"] = "openai"

    # Ollama (always used for embeddings; used for generation only when
    # LLM_PROVIDER="ollama")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Local generation fallback. devstral:24b — Mistral's agentic/tool-use
    # model, able to drive the LangChain SQL agent's ReAct tool loop.
    # Pull once: `ollama pull devstral:24b`. Only used when LLM_PROVIDER="ollama".
    OLLAMA_MODEL: str = "devstral:24b"
    # Keep the model resident between the many sequential calls that AIR×DER
    # issues per query — avoids reload latency on every call.
    OLLAMA_KEEP_ALIVE: str = "30m"
    # Per-call request timeout (seconds) for the Ollama HTTP client.
    OLLAMA_TIMEOUT_S: float = 120.0
    # In-process LLM response cache (prompt-hash keyed). Blunts the cost of the
    # 8–15 sequential calls a full ETHER→AIR→DER query can trigger on a local
    # model. Set to 0 to disable.
    LLM_CACHE_MAXSIZE: int = 2048

    # ── OpenAI-compatible API (default: OpenRouter → NVIDIA Nemotron 3 Ultra) ──
    # OPENAI_BASE_URL selects the host:
    #   OpenRouter   → https://openrouter.ai/api/v1        (key: sk-or-v1-…)
    #   NVIDIA NIM   → https://integrate.api.nvidia.com/v1 (key: nvapi-…)
    #   OpenAI       → https://api.openai.com/v1           (key: sk-…)
    # Set the key in .env as OPENAI_API_KEY — never hard-code it here.
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENAI_MODEL: str = "nvidia/nemotron-3-ultra"
    # Per-call request timeout (seconds). Nemotron Ultra is a large reasoning
    # model — the AIR/DER loops issue long prompts, so keep this generous.
    OPENAI_TIMEOUT_S: float = 180.0
    # Retries on transient 429/5xx from the API (handled by the OpenAI SDK).
    OPENAI_MAX_RETRIES: int = 3
    # Optional OpenRouter attribution headers (ignored by other hosts).
    OPENROUTER_SITE_URL: str = ""
    OPENROUTER_APP_NAME: str = "medical-rag"

    # ── Groq (PRIMARY generation endpoint) ────────────────────────────────────
    # Groq is OpenAI-compatible, so the SAME `openai` SDK is used — only the
    # base URL + key differ (no extra dependency). Free tier is generous
    # (~30 req/min, ~1000 req/day, no credit card) which suits a hosted demo far
    # better than OpenRouter's 50/day free cap.
    # Get a key at https://console.groq.com/keys  (starts with gsk_).
    # On Hugging Face Spaces, set this as a Space *secret*, not in a committed file.
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    # NOTE: Groq retired the Llama 3.x chat models on 2026-08-16. The current
    # production text models are OpenAI's open-weight GPT-OSS. gpt-oss-120b is
    # the flagship and the documented replacement for llama-3.3-70b-versatile —
    # well-suited to factual extraction (e.g. pulling "Z89" from a code list).
    # Swap to openai/gpt-oss-20b for lower latency / higher throughput.
    # Verify what your key can access:
    #   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TIMEOUT_S: float = 60.0
    GROQ_MAX_RETRIES: int = 2

    # ── Generation fallback chain ─────────────────────────────────────────────
    # When True (and LLM_PROVIDER != "ollama"), generation tries endpoints in
    # priority order: Groq first (if GROQ_API_KEY is set), then the OpenRouter /
    # OpenAI endpoint (OPENAI_* above) as a backup. A transient failure or
    # rate-limit (429 / 5xx / timeout) on one endpoint automatically cascades to
    # the next, instead of surfacing the raw-chunk "_fallback_answer". Set False
    # to use only the highest-priority configured endpoint.
    LLM_FALLBACK_ENABLED: bool = True

    # ── PostgreSQL structured facts database ──────────────────────────────────
    # POSTGRES_DSN:          read-write connection — used by ingestion and writes.
    # POSTGRES_READONLY_DSN: SELECT-only role — used by all query-path SELECTs
    #                        and generated SQL. If not set, falls back to
    #                        POSTGRES_DSN with a loud startup warning.
    #
    # When POSTGRES_READONLY_DSN is eventually configured, point it at a Postgres
    # role that has ONLY SELECT granted on the tables below — this ensures that
    # even if sqlglot's AST check has a bug and a DML query slips through, the
    # database itself rejects it.
    #
    # Example:
    #   POSTGRES_DSN=postgresql://rag_admin:pass@localhost:5432/medical_rag
    #   POSTGRES_READONLY_DSN=postgresql://rag_readonly:pass@localhost:5432/medical_rag
    POSTGRES_DSN: str = ""
    POSTGRES_READONLY_DSN: Optional[str] = None

    # Max connections per pool (Phase 6). Pooling replaces per-row connect/close
    # so large multi-dataset ingests no longer exhaust connections or time out.
    POSTGRES_POOL_MAX: int = 10

    # Name of the PostgreSQL role that POSTGRES_READONLY_DSN connects as.
    # After every CREATE VIEW, _build_and_create_view() issues:
    #   GRANT SELECT ON <view> TO <role>
    # so the read-only role can query exposed streams but NOT raw stg_ tables.
    # Leave empty to skip the GRANT (a warning is logged).
    POSTGRES_READONLY_ROLE: str = "rag_readonly"

    # Auto-created at init_db: a SELECT-only role the LangChain SQL agent connects
    # as, so its generated SQL can never write/drop. Password for that role.
    POSTGRES_READONLY_PASSWORD: str = "rag_readonly_pw"
    # When True, init_db creates the read-only role + grants and (if
    # POSTGRES_READONLY_DSN is unset) derives it from POSTGRES_DSN.
    AUTO_CREATE_READONLY_ROLE: bool = True

    # Path to the stream-config YAML (human_label, load_mode, column_mapping …).
    # Mounted as a volume so edits take effect without rebuilding the image.
    SOURCES_YAML_FILE: str = "/app/data/sources.yaml"

    # Path to the canonical event adapter config (Phase 1). Drives the
    # dataset-agnostic column→concept mapping used by ETHER/AIR/DER.
    CONCEPT_MAP_FILE: str = "/app/data/concept_map.yaml"

    # Path to the clinical prediction-task registry (Phase 4/5).
    PREDICTION_TASKS_FILE: str = "/app/data/prediction_tasks.yaml"

    # Snapshot reconcile guards — prevent mass soft-delete from a short file.
    # RECONCILE_MIN_FRACTION: current file must contain at least this fraction
    #   of the live rows already in the table.  0.90 = 90 % must still be there.
    # RECONCILE_ABSOLUTE_FLOOR: skip reconcile if the file has fewer than this
    #   many rows (protects against empty/truncated file incidents).
    STAGING_RECONCILE_MIN_FRACTION: float = 0.90
    STAGING_RECONCILE_ABSOLUTE_FLOOR: int = 10

    # ── Storage bucket directory ──────────────────────────────────────────────
    BUCKET_DIR: str = "/app/bucket"

    # Structured format-folders (comma-separated, relative to BUCKET_DIR). Each is
    # a folder of same-format documents processed by the folder-per-format
    # pipeline (1 LLM reference + deterministic rest) instead of the per-file
    # path — and EXCLUDED from the per-file auto-ingest so nothing is double-
    # processed. Example: "uploads/invoice1,uploads/statements".
    STRUCTURED_DOC_FOLDERS: str = "uploads/invoice1"

    # ── Startup behaviour ─────────────────────────────────────────────────────
    AUTO_INGEST: bool = True

    # ── RAG-only mode (Hugging Face / no-Postgres demo) ───────────────────────
    # When True, the app runs as a pure unstructured-PDF RAG service:
    #   • the query router always routes to the 'rag' path (no SQL / analytical
    #     lookups), so PostgreSQL is never touched at query time;
    #   • startup skips init_db, entity indexing, the event layer, and the
    #     structured-folder pipeline — all of which require PostgreSQL.
    # This lets the SAME codebase run locally WITH Postgres (RAG_ONLY_MODE=false)
    # and on Hugging Face Spaces WITHOUT it (RAG_ONLY_MODE=true).
    RAG_ONLY_MODE: bool = False

    # Run the second (hallucination / grounding) LLM call after answer synthesis.
    # Each RAG query costs TWO LLM calls when on. Turn OFF on rate-limited hosts
    # (e.g. the Groq free tier) to halve generation usage per question.
    ENABLE_HALLUCINATION_CHECK: bool = True

    # ── Document chunking ─────────────────────────────────────────────────────
    # Used for plain-text documents.  PDF guidebooks use the table-aware chunker
    # in ingestion.py which respects these settings but may produce smaller
    # logical units to avoid splitting tables.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Hybrid search balance: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced.
    HYBRID_SEARCH_ALPHA: float = 0.5

    # CrossEncoder reranker: retrieve RERANKER_INITIAL_K candidates, rerank,
    # keep the top RERANKER_TOP_K to pass to the LLM.
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Keep fewer, higher-quality chunks: 3 tightly-relevant chunks give the LLM a
    # single clear answer, where 5 invited it to spread across weakly-related ones.
    RERANKER_TOP_K: int = 3
    RERANKER_INITIAL_K: int = 20

    # Minimum CrossEncoder score a chunk must achieve after reranking to be
    # included in the answer context.  Chunks below this threshold are dropped
    # without any LLM call (replaces the per-chunk LLM relevance grader).
    # Range: CrossEncoder ms-marco scores are unbounded but typically –10 … +10;
    # a value of 0.0 DISABLES the filter (everything passes). We enable a real
    # relevance floor at 1.0 so weakly-related chunks are dropped before the LLM
    # sees them — this is a primary fix for vague / multi-answer responses. If it
    # ever filters out chunks you know are relevant, lower it toward 0.5; raise it
    # toward 2.0 if off-topic chunks still slip through.
    RERANKER_SCORE_THRESHOLD: float = 1.0

    # ── Text-to-SQL ───────────────────────────────────────────────────────────
    # Path to the schema metadata + few-shot examples YAML file used to ground
    # LLM-generated SQL for analytical queries.
    TEXT_TO_SQL_SCHEMA_FILE: str = "/app/data/sql_schema_metadata.yaml"

    # Maximum rows an LLM-generated SELECT may return.
    # Injected automatically if the generated SQL has no LIMIT clause.
    TEXT_TO_SQL_MAX_ROWS: int = 500

    # Statement timeout (milliseconds) applied to every generated SQL execution.
    # Prevents runaway analytical queries from blocking the connection pool.
    TEXT_TO_SQL_TIMEOUT_MS: int = 10000

    # ── LangChain SQL agent (structured query engine) ─────────────────────────
    # When ON, structured/data questions are answered by the LangChain SQL
    # Database Agent (schema introspection + tool-using agent over Postgres),
    # replacing the deterministic query shapes and the custom text-to-SQL
    # fallback. RAG remains the semantic fallback for reference questions.
    USE_SQL_AGENT: bool = True
    # Rows of sample data included per table in the schema the agent sees
    # (helps it learn value formats, e.g. DD/MM/YYYY dates).
    SQL_AGENT_SAMPLE_ROWS: int = 3
    # Max reasoning/tool steps per query (guards runaway agent loops).
    SQL_AGENT_MAX_ITERATIONS: int = 15
    # Agent type: "tool-calling" (needs a tool-calling model) or
    # "zero-shot-react-description" (text ReAct, works with any chat model).
    SQL_AGENT_TYPE: str = "tool-calling"
    # Expose ALL public tables (incl. raw stg_* staging) to the agent, or only
    # the curated tables + v_* views. Per current decision: everything.
    SQL_AGENT_EXPOSE_ALL_TABLES: bool = True

    # ── EHR-RAG pillars (ETHER / AIR / DER) ───────────────────────────────────
    # Master switch. When ON, retriever.query() routes RAG-path queries through
    # the event-aware ETHER retrieval and (for prediction tasks) the AIR+DER
    # reasoning loop instead of the single-pass hybrid pipeline.
    ENABLE_EHR_RAG: bool = False

    # ETHER — Event- & Time-Aware Hybrid Retrieval (§3.2)
    # alpha trades semantic similarity (s_sem) against U-shaped temporal
    # relevance (s_time):  s(c) = alpha*s_sem + (1-alpha)*s_time
    ETHER_ALPHA: float = 0.7
    # Decay constants (in days) for the U-shaped weight: emphasise events close
    # to the prediction time (tau_recent) and near trajectory onset (tau_early).
    ETHER_TAU_RECENT_DAYS: float = 90.0
    ETHER_TAU_EARLY_DAYS: float = 365.0
    # Final number of textual chunks kept after temporal re-scoring.
    ETHER_K_FINAL: int = 8
    # Numeric path: candidate indicators after coarse retrieval, indicators kept
    # after fine (LLM/cross-encoder) reranking, and most-recent values per indicator.
    ETHER_N_COARSE: int = 20
    ETHER_N_FINE: int = 8
    ETHER_N_RECENT: int = 5

    # AIR — Adaptive Iterative Retrieval (§3.3)
    AIR_MAX_ITERATIONS: int = 3
    # Hard cap on total textual chunks accumulated across AIR iterations
    # (context-budget guard for the local model).
    AIR_MAX_EVIDENCE_CHUNKS: int = 16

    # DER — Dual-Path Evidence Retrieval & Reasoning (§3.4)
    # When ON, prediction queries retrieve along factual (q+) and counterfactual
    # (q-) paths and fuse them in a comparative decision step.
    ENABLE_DER: bool = True

    # ── Compound query orchestration (feature flag) ───────────────────────────
    # When ON, queries that require SQL results to formulate the right RAG
    # retrieval (e.g. "is patient A's 99214 coded correctly?") are handled by
    # the compound loop in orchestrator.py instead of the single-pass hybrid path.
    #
    # Default OFF — the loop is stubbed and this flag exists as the clean seam.
    # Enable only after the loop is implemented and tested.
    ENABLE_COMPOUND_LOOP: bool = False

    # ── Audit logging ─────────────────────────────────────────────────────────
    # Rotating JSON-lines file: one record per query.
    # Required for HIPAA audit trail (query text, route, generated SQL, source IDs).
    AUDIT_LOG_PATH: str = "/app/logs/audit.jsonl"
    AUDIT_LOG_MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB per file
    AUDIT_LOG_BACKUP_COUNT: int = 10               # keep 10 rotated files

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# ── Post-load validation / warnings ──────────────────────────────────────────

if not settings.POSTGRES_DSN:
    logger.warning(
        "POSTGRES_DSN is not set — the structured database will be unavailable. "
        "Set POSTGRES_DSN in your .env file before starting the server."
    )

if settings.POSTGRES_READONLY_DSN is None:
    logger.warning(
        "POSTGRES_READONLY_DSN is not set. "
        "All SQL queries (including LLM-generated text-to-SQL) will run under "
        "the full-privilege POSTGRES_DSN connection. "
        "For production / HIPAA compliance, create a SELECT-only Postgres role "
        "and set POSTGRES_READONLY_DSN to use it as the query connection."
    )

if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
    logger.warning(
        "LLM_PROVIDER=openai but OPENAI_API_KEY is empty — every generation call "
        "(SQL agent, intent parsing, AIR/DER, answer synthesis) will fail. "
        "Add OPENAI_API_KEY to your .env file (base URL: %s, model: %s).",
        settings.OPENAI_BASE_URL, settings.OPENAI_MODEL,
    )
