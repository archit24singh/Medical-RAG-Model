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

    # Embedding model (runs locally inside the backend container)
    # all-MiniLM-L6-v2 is ~90MB, downloaded once and cached
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # LLM Provider
    # "ollama" -> free, fully local (requires Ollama installed on host)
    # "openai" -> cloud, requires OPENAI_API_KEY
    LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"

    # Ollama settings
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

    # ── Document processing pipeline ──────────────────────────────────────────
    # Prepend 2-3 sentences of context to every chunk before storing.
    # Improves retrieval accuracy at the cost of 1 extra LLM call per chunk.
    ENABLE_CONTEXTUAL_ENRICHMENT: bool = True

    # Use SmolVLM (256 M-param vision model) to describe embedded images.
    # Disabled by default — enable once the model is cached in model_cache volume.
    ENABLE_IMAGE_DESCRIPTION: bool = False

    # Target maximum tokens per chunk (≈ words × 1.3).
    # Chunks at or below this size are not split further by the LLM.
    CHUNK_MAX_TOKENS: int = 512

    # Hybrid search balance — 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced.
    HYBRID_SEARCH_ALPHA: float = 0.5

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
