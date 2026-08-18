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
from rag.ingestion import (
    ingest_directory, ingest_file, refresh_event_layer, ingest_structured_folders,
)
from rag.orchestrator import query as rag_query, predict as rag_predict
from rag.vectorstore import delete_document, get_document_count, list_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Medical RAG System starting up...")

    # RAG-only mode (Hugging Face / no-Postgres demo): skip every startup step
    # that depends on PostgreSQL. Only the ChromaDB vector path is initialised.
    if settings.RAG_ONLY_MODE:
        logger.info("RAG_ONLY_MODE enabled — skipping all PostgreSQL startup steps.")
    else:
        # Initialise PostgreSQL structured facts database
        try:
            init_db()
        except Exception as e:
            logger.warning("PostgreSQL init failed (non-fatal): %s", e)

        # Ensure the all-columns entity-profile path is index-backed (scales to 1M+).
        # Best-effort: no-op if there are no staging tables yet.
        try:
            from rag.profiles import ensure_entity_indexes
            ensure_entity_indexes()
        except Exception as e:
            logger.warning("Entity index init failed (non-fatal): %s", e)

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
                # The event layer and structured-folder pipeline both require
                # PostgreSQL — skip them entirely in RAG-only mode.
                if not settings.RAG_ONLY_MODE:
                    # Auto-build the event layer so predict/rollup/event-Q&A are
                    # ready without a manual /events/backfill.
                    refresh_event_layer()
                    # Structured format-folders (e.g. uploads/invoice1) →
                    # folder-per-format pipeline (excluded from the per-file walk).
                    sf = ingest_structured_folders(settings.BUCKET_DIR)
                    if sf:
                        logger.info("Structured folders ingested: %s", list(sf.keys()))
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


class PredictRequest(BaseModel):
    task: str                              # task key in prediction_tasks.yaml or free text
    patient_key: Optional[str] = None
    prediction_time: Optional[str] = None  # ISO date anchor (τ*)


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
            "llm_endpoint": (
                settings.OLLAMA_BASE_URL
                if settings.LLM_PROVIDER == "ollama"
                else settings.OPENAI_BASE_URL
            ),
            # Never echo the key itself — just whether one is configured.
            "llm_api_key_configured": (
                True if settings.LLM_PROVIDER == "ollama" else bool(settings.OPENAI_API_KEY)
            ),
            "embedding_model": settings.EMBEDDING_MODEL,
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
        # Auto-refresh the event layer (no manual /events/backfill needed).
        event_summary = await loop.run_in_executor(None, refresh_event_layer)
        # Structured format-folders → folder-per-format pipeline.
        structured = await loop.run_in_executor(
            None, ingest_structured_folders, settings.BUCKET_DIR
        )
        return {
            "message":       "Ingestion complete",
            "success_count": len(res["success"]),
            "error_count":   len(res["errors"]),
            "skipped_count": len(res["skipped"]),
            "event_layer":   event_summary,
            "structured":    structured,
            "details":       res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/async")
async def ingest_async():
    """
    Start ingestion as a background job (Phase 6) and return a job_id immediately.
    Use for large multi-dataset loads that would otherwise time out the request.
    Builds canonical events + event index after ingest. Poll /ingest/status/{id}.
    """
    try:
        from rag.jobs import submit_directory_job
        job_id = submit_directory_job(settings.BUCKET_DIR, build_events=True)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    """Poll the status/result of a background ingestion job."""
    from rag.jobs import get_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


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
            # Auto-refresh event layer so the new file is immediately usable by
            # predict/rollup/event-Q&A without a manual /events/backfill.
            try:
                result["event_layer"] = refresh_event_layer()
            except Exception as exc:
                logger.warning("Event-layer refresh after upload failed: %s", exc)
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


class FolderIngestRequest(BaseModel):
    folder: str                            # folder path (e.g. "bucket/invoice1")
    coverage_threshold: Optional[float] = 90.0


@app.post("/ingest/folder")
async def ingest_folder_endpoint(request: FolderIngestRequest):
    """
    Folder-per-format ingest for structured documents (invoices/statements):
    one LLM reference call per folder + deterministic passes for the rest,
    stored as JSONB + flattened into the queryable staging layer.
    """
    try:
        from rag.doc_store import ingest_folder
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, ingest_folder, request.folder, request.coverage_threshold
        )
    except Exception as e:
        logger.error("Folder ingest failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/events/backfill")
async def events_backfill():
    """
    Build the canonical events table from already-ingested staging data and index
    timestamped event chunks into the vector store (Phase 1). Idempotent — safe to
    re-run. This is what makes ETHER/AIR/DER usable on existing data without
    re-ingesting the source files.
    """
    try:
        from rag.events import backfill_events
        from rag.event_chunker import index_all_events

        loop = asyncio.get_event_loop()
        inserted = await loop.run_in_executor(None, backfill_events, None)
        chunks = await loop.run_in_executor(None, index_all_events, None)
        return {"events_inserted": inserted, "event_chunks_indexed": chunks}
    except Exception as e:
        logger.error("Events backfill failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_outcome(request: PredictRequest):
    """
    Run a clinical prediction (ETHER → AIR → DER) for a task defined in
    prediction_tasks.yaml. This is the predictive path (not Q/A): it returns a
    class label, confidence, rationale, and the dual-path evidence used.

    NOTE: predictions can be *produced* on any event data, but are only
    *benchmarkable* (Macro-F1) once labeled longitudinal data is ingested.
    """
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="task cannot be empty.")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, rag_predict, request.task, request.patient_key, request.prediction_time
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patient/profile")
async def patient_profile_endpoint(name: Optional[str] = None, key: Optional[str] = None):
    """
    Full per-patient profile (all columns, schema-on-read) straight from the
    staging tables — no events backfill needed. Provide ?name= or ?key=.
    Returns identity fields, summed charge/payment totals, and per-claim line items.
    """
    if not name and not key:
        raise HTTPException(status_code=400, detail="Provide name or key.")
    try:
        from rag.profiles import patient_profile
        prof = patient_profile(name=name, key=key)
        if not prof:
            raise HTTPException(status_code=404, detail="No matching patient found.")
        return prof
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile failed: %s", e, exc_info=True)
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
