"""
FastAPI backend for the Medical RAG System.

Endpoints:
  GET  /health              - service health + document count
  POST /ingest              - ingest all files from bucket/ folder
  POST /ingest/upload       - upload and ingest a file directly
  POST /query               - natural-language document retrieval
  GET  /documents           - list all indexed documents
  DELETE /documents/{id}    - remove a document from the index

Interactive API docs: http://localhost:8000/docs
"""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from db.schema import init_db
from rag.ingestion import ingest_directory, ingest_file
from rag.retriever import query as rag_query
from rag.vectorstore import delete_document, get_document_count, list_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Medical RAG System starting up...")

    # Initialise SQLite structured facts database
    try:
        init_db()
    except Exception as e:
        logger.warning("SQLite init failed (non-fatal): %s", e)

    if settings.AUTO_INGEST:
        # Run ingestion in a background thread so FastAPI starts serving
        # requests immediately — ingestion can take a long time with LLM extraction.
        def _bg_ingest():
            logger.info("Auto-ingesting files from bucket: %s ...", settings.BUCKET_DIR)
            try:
                res = ingest_directory(settings.BUCKET_DIR)
                logger.info(
                    "Auto-ingest done - %d indexed, %d errors",
                    len(res["success"]), len(res["errors"])
                )
            except Exception as e:
                logger.warning("Auto-ingest failed (will retry on demand): %s", e)

        threading.Thread(target=_bg_ingest, daemon=True).start()

    yield
    logger.info("Medical RAG System shut down.")


app = FastAPI(
    title="Medical RAG System",
    description="Retrieve patient and provider medical documents using natural language queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5


@app.get("/health")
async def health():
    """Health check - reports document count and active LLM."""
    try:
        count = get_document_count()
        return {
            "status": "healthy",
            "documents_indexed": count,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": (
                settings.OLLAMA_MODEL
                if settings.LLM_PROVIDER == "ollama"
                else settings.OPENAI_MODEL
            ),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/ingest")
async def ingest_all():
    """
    Walk the bucket/ directory and ingest every supported file.
    Drop files into bucket/patients/ or bucket/providers/ then call this endpoint.

    Runs ingest_directory in a thread-pool executor so the FastAPI event loop
    stays free during ingestion — /health and /query remain responsive throughout.
    """
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, ingest_directory, settings.BUCKET_DIR)
        return {
            "message":       "Ingestion complete",
            "success_count": len(res["success"]),
            "error_count":   len(res["errors"]),
            "skipped_count": len(res["skipped"]),
            "details":       res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def ingest_uploaded_file(file: UploadFile = File(...)):
    """Upload a file and ingest it immediately. Saved to bucket/uploads/ then indexed."""
    upload_dir = Path(settings.BUCKET_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    try:
        content = await file.read()
        dest.write_bytes(content)
        result = ingest_file(str(dest))

        if result["status"] == "success":
            return result
        raise HTTPException(status_code=422, detail=result.get("message", "Ingestion failed"))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the vector store with a natural language prompt.

    Examples:
      "Get patient Alice Johnson bill for 27-10-2025"
      "What is the NPI number for Dr. Robert Chen?"
      "Show me all records for patient P001"
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return rag_query(request.query)
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List every document currently indexed in the vector store."""
    try:
        docs = list_all_documents()
        return {"total": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Remove a document from the vector store by its ID."""
    if delete_document(doc_id):
        return {"message": "Document " + doc_id + " removed."}
    raise HTTPException(
        status_code=404,
        detail="Document " + doc_id + " not found or could not be deleted."
    )
