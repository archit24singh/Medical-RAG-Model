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

    # ── Embedding model — served by Ollama ────────────────────────────────────
    # nomic-embed-text: 8192-token context window.
    # Pull once: ollama pull nomic-embed-text
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # "ollama" → fully local (requires Ollama on host)
    # "openai" → cloud, requires OPENAI_API_KEY
    LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"

    # Ollama (used for both embeddings and generation)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

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

    # Name of the PostgreSQL role that POSTGRES_READONLY_DSN connects as.
    # After every CREATE VIEW, _build_and_create_view() issues:
    #   GRANT SELECT ON <view> TO <role>
    # so the read-only role can query exposed streams but NOT raw stg_ tables.
    # Leave empty to skip the GRANT (a warning is logged).
    POSTGRES_READONLY_ROLE: str = ""

    # Path to the stream-config YAML (human_label, load_mode, column_mapping …).
    # Mounted as a volume so edits take effect without rebuilding the image.
    SOURCES_YAML_FILE: str = "/app/data/sources.yaml"

    # Snapshot reconcile guards — prevent mass soft-delete from a short file.
    # RECONCILE_MIN_FRACTION: current file must contain at least this fraction
    #   of the live rows already in the table.  0.90 = 90 % must still be there.
    # RECONCILE_ABSOLUTE_FLOOR: skip reconcile if the file has fewer than this
    #   many rows (protects against empty/truncated file incidents).
    STAGING_RECONCILE_MIN_FRACTION: float = 0.90
    STAGING_RECONCILE_ABSOLUTE_FLOOR: int = 10

    # ── Storage bucket directory ──────────────────────────────────────────────
    BUCKET_DIR: str = "/app/bucket"

    # ── Startup behaviour ─────────────────────────────────────────────────────
    AUTO_INGEST: bool = True

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
    RERANKER_TOP_K: int = 5
    RERANKER_INITIAL_K: int = 20

    # Minimum CrossEncoder score a chunk must achieve after reranking to be
    # included in the answer context.  Chunks below this threshold are dropped
    # without any LLM call (replaces the per-chunk LLM relevance grader).
    # Range: CrossEncoder ms-marco scores are unbounded but typically –10 … +10;
    # 0.0 is a reasonable starting point — tune upward (e.g. 1.0, 2.0) if the
    # LLM receives too many off-topic chunks.
    RERANKER_SCORE_THRESHOLD: float = 0.0

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
