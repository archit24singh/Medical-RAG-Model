"""
Central configuration - all values come from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ChromaDB (vector database)
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "medical_docs"

    # Embedding model — served by Ollama.
    # nomic-embed-text has an 8192-token context window; the previous
    # all-MiniLM-L6-v2 was capped at 256 tokens and silently truncated every
    # chunk beyond that limit, making half of each chunk invisible to search.
    # Pull once before starting the stack: ollama pull nomic-embed-text
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # LLM Provider
    # "ollama" -> free, fully local (requires Ollama installed on host)
    # "openai" -> cloud, requires OPENAI_API_KEY
    LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"

    # Ollama settings (used for both embeddings and text generation)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"

    # OpenAI settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # SQLite structured facts database
    # Stored in a dedicated volume so it persists across container restarts.
    DB_PATH: str = "/app/data/medical_rag.db"

    # Storage bucket directory
    # Currently a local folder mounted into the container via docker-compose.yml.
    # To upgrade to S3: replace this path with an S3 URI (e.g. s3://your-bucket)
    # and update ingestion.py loaders to use boto3/s3fs instead of open().
    BUCKET_DIR: str = "/app/bucket"

    # Startup behaviour
    AUTO_INGEST: bool = True

    # ── Document chunking ──────────────────────────────────────────────────────
    # Character-based sliding-window splitter — mirrors the reference RAG baseline
    # (RecursiveCharacterTextSplitter, chunk_size=1000, chunk_overlap=200).
    # 1000 chars ≈ 200-250 tokens, well inside nomic-embed-text's 8192-token limit.
    # No LLM calls during ingestion.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ──────────────────────────────────────────────────────────────
    # Hybrid search balance — 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced.
    HYBRID_SEARCH_ALPHA: float = 0.5

    # CrossEncoder reranker — retrieve RERANKER_INITIAL_K candidates from the
    # hybrid search, rerank with the cross-encoder, then pass only the top
    # RERANKER_TOP_K to the LLM for answer generation.
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K: int = 5
    RERANKER_INITIAL_K: int = 20

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
