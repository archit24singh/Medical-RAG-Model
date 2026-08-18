# 🏥 Medical RAG System

A privacy-preserving **Retrieval-Augmented Generation (RAG)** system for clinical
records. It answers plain-English questions about patient bills, medical records,
and provider information by combining structured metadata filtering / exact SQL
lookup with semantic + keyword (hybrid) search over a vector database — without
ever sending patient data to a third-party API by default.

---

## What it does

Ask questions like:

```
Get patient A's bill for 27-10-2025
What is the NPI number for Dr. R?
Show me John Doe's medical record from September 2025
What is the total amount A owes?
```

...and get an answer grounded directly in your ingested documents, along with the
exact source record(s) it came from. The system is designed around an
**anti-hallucination guarantee**: structured (SQL) results are quoted verbatim,
and the LLM is used only to format or summarize retrieved facts — never to invent
them.

---

## Architecture

```
Your Files (PDFs, JSONs, CSVs)
        │
        ▼
  [ Ingestion Pipeline ]
  • Extract text from file
  • Extract metadata: patient name, date, doc type, NPI, etc.
  • Convert text → vector embedding (sentence-transformers)
        │
        ▼
  [ ChromaDB — Vector Database ]
  • Stores embeddings + metadata for every document
  • Supports exact metadata filtering + semantic similarity search
        │
        ▼  (query time)
  [ Intent Parser (LLM) ]
  • "Get Alice Johnson's bill for 27-10-2025"
    → { patient_name: "Alice Johnson", date: "2025-10-27", doc_type: "bill" }
        │
        ▼
  [ ChromaDB Search ]
  • Filter by metadata (exact) + rank by semantic similarity
        │
        ▼
  [ Answer Generator (LLM) ]
  • Reads retrieved documents and writes a clear answer
        │
        ▼
  [ Streamlit UI / FastAPI ]
  • Displays answer + source documents with relevance scores
```

In addition to the vector search path above, the system maintains a parallel
**SQLite "facts" database** populated directly from tabular and unstructured
ingestion. Precise factual queries (e.g. "What is the claim number for...?")
are routed to SQL first for an exact, deterministic, zero-hallucination answer;
open-ended questions fall back to the hybrid vector + BM25 search shown above.

---

## Inference model — cloud by default, local on request

The default generation model is **NVIDIA Nemotron 3 Ultra, served over the
OpenRouter API** (`LLM_PROVIDER=openai`). Embeddings still run locally through
Ollama (`nomic-embed-text`), so the vector store never leaves the machine.

⚠️ **PHI leaves the device on the default configuration.** Every prompt sent to
the LLM — clinical notes, patient names, diagnoses, billing detail, and the
retrieved chunks used as context — is transmitted to the API host. Before using
this on real patient data you need the corresponding data agreements (a HIPAA
BAA with the API provider, data-residency review, and confirmation that prompts
are excluded from provider logging/training). Neither OpenRouter nor NVIDIA NIM
is covered by such an agreement out of the box.

**The fully local path is still supported and is one line of config:**

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=devstral:24b     # ollama pull devstral:24b (~14 GB Q4)
```

With `LLM_PROVIDER=ollama`, nothing is transmitted to an external API — no
third-party data-handling agreement, no cross-border transfer question, and no
risk of patient data appearing in a provider's logs. That mode also gives
fixed, versioned behavior for reproducible benchmarking, at the cost of needing
24 GB of unified memory and accepting higher latency.

The two paths share one code path (`backend/rag/llm_client.py`), so cloud vs.
local can be benchmarked head-to-head by flipping `LLM_PROVIDER`.

---

## Supported file formats

| Format | Ingestion path |
|--------|---------------|
| `.csv`, `.xlsx`, `.xls` | Tabular pipeline — grouped by patient, plus per-row direct SQLite ingestion |
| `.json` | Structured field mapping (with LLM fallback) |
| `.pdf`, `.docx`, `.doc`, `.pptx`, `.html`, `.htm`, `.xml` | Unstructured pipeline — Docling parse → chunk → enrich → structured extraction |
| `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` | Unstructured pipeline with OCR (RapidOCR) |
| `.txt`, `.md` | Unstructured pipeline |

---

## Setup / Installation

For installation, configuration (Ollama vs. OpenAI), Docker usage, and
troubleshooting, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**.

---

## EHR-RAG pillars (predictive layer)

On top of the retrieval-Q/A system above, the project now implements the three
pillars from the EHR-RAG paper, turning it from lookup-Q/A into a *predictive*
system. The local model is **Llama 3.1 8B** (via Ollama).

- **Canonical event adapter** (`rag/events.py`, `data/concept_map.yaml`) maps any
  source schema into canonical events `(concept, value, timestamp, type)`. Adding
  a dataset (MIMIC/EHRSHOT) is a config edit, not a code change. Events are built
  from the existing staging tables — no re-ingestion needed.
- **ETHER** (`rag/ether.py`) — Event- & Time-Aware Hybrid Retrieval: U-shaped
  temporal scoring on textual events + per-indicator numeric trajectories with
  coarse-to-fine selection.
- **AIR** (`rag/air.py`) — Adaptive Iterative Retrieval: the LLM judges evidence
  sufficiency and issues focused refinement queries (capped iterations + budget).
- **DER** (`rag/der.py`) — Dual-Path Evidence Retrieval & Reasoning: factual vs.
  counterfactual retrieval, dual hypotheses, evidence fusion, comparative decision
  → a predicted label + rationale.
- **PDF structured extraction + OCR** (`rag/pdf_tables.py`, `rag/ocr.py`) — PDF
  tables become canonical events (SQL-/event-queryable); scanned PDFs/images are
  OCR'd (RapidOCR).
- **Evaluation harness** (`eval/evaluate.py`) — Accuracy / Macro-F1 / per-class F1
  against a labels file. Meaningful only once labeled longitudinal data exists.

### New endpoints

```
POST /events/backfill          # build canonical events + index event chunks
POST /predict                  # {task, patient_key, prediction_time} → prediction
POST /ingest/async             # background ingest (returns job_id)
GET  /ingest/status/{job_id}   # poll a background ingest job
```

Typical predictive workflow: ingest → `POST /events/backfill` → `POST /predict`
with a task from `data/prediction_tasks.yaml`. To benchmark, add a labels file
and run `python -m eval.evaluate --task <name> --labels <file>`.

## Research Context

This system is being developed as the technical foundation for a research study
on **privacy-preserving Retrieval-Augmented Generation for clinical data**. The
current implementation is validated against synthetic/demo data; the next phase
of work migrates the pipeline to **[MIMIC-III](https://physionet.org/content/mimiciii/)**,
a de-identified, publicly available clinical dataset from PhysioNet that is a
standard benchmark in healthcare NLP research. MIMIC-III provides realistic
clinical notes, discharge summaries, and structured billing/coding data
(`NOTEEVENTS.csv`, `DIAGNOSES_ICD.csv`, etc.) at a scale and complexity that
mirrors real-world EHR systems, making it a suitable benchmark for evaluating
this system's ingestion, retrieval, and privacy properties under realistic
conditions.

---

## Roadmap

- **MIMIC-III integration** — migrate ingestion and retrieval to operate over
  the MIMIC-III benchmark dataset (clinical notes, discharge summaries,
  structured billing/coding data) in place of synthetic demo data.
- **Membership inference attack evaluation** — assess whether the system's
  retrieval and LLM outputs leak signal about whether a specific patient's
  record was present in the underlying dataset.
- **Differential privacy evaluation** — evaluate techniques for adding formal
  privacy guarantees to retrieval and/or generation, and measure the resulting
  accuracy/privacy trade-off.
- **Local vs. cloud inference benchmarking** — systematically compare retrieval
  accuracy and latency between local (Ollama/Mistral) and cloud (OpenAI)
  inference paths.
