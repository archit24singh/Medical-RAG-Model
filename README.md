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

## Why local inference matters for clinical data

This system runs its LLM (Mistral) **locally via Ollama** by default, with
OpenAI available only as an explicit opt-in alternative.

For clinical data, this design choice is not cosmetic:

- **No PHI leaves the machine.** Every prompt sent to the LLM — including raw
  clinical notes, patient names, diagnoses, and billing details — is processed
  entirely on local hardware when using Ollama. Nothing is transmitted to an
  external API, so there is no third-party data-handling agreement, no
  cross-border data transfer question, and no risk of patient data appearing in
  a cloud provider's logs or training data.
- **Reduced regulatory surface.** Local inference sidesteps a large class of
  HIPAA Business Associate Agreement (BAA) and data residency concerns that
  arise the moment PHI is sent to a cloud LLM API.
- **Reproducibility for research.** A fixed local model (Mistral via Ollama)
  gives deterministic, versioned behavior for benchmarking — important when
  evaluating retrieval accuracy or running repeated experiments against the
  same data.
- **OpenAI remains available as a cloud baseline.** `LLM_PROVIDER=openai` is
  supported for users who have appropriate data agreements in place, and is
  useful as a comparison point when benchmarking local vs. cloud inference
  (see Roadmap below).

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
