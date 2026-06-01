# 🏥 Medical RAG System — Setup Guide

Retrieve patient bills, medical records, and provider data using plain English queries.

---

## Tools Required

| Tool | Purpose | Install Link |
|------|---------|--------------|
| **Docker Desktop** | Runs all 3 services (ChromaDB, backend, frontend) | https://docker.com/products/docker-desktop |
| **Ollama** *(if using local LLM)* | Runs the AI model locally — free, no API key | https://ollama.com |
| **OpenAI API key** *(alternative to Ollama)* | Cloud LLM — costs per token | https://platform.openai.com |

> The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically the first time.

---

## How It Works (Architecture)

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

---

## Quick Start (Docker — Recommended)

### Step 1 — Install Docker Desktop
Download and install from https://docker.com/products/docker-desktop  
Make sure Docker is running (icon in system tray).

### Step 2 — Set up Ollama (local LLM, free)
```bash
# Install from https://ollama.com, then:
ollama pull mistral
```
Keep Ollama running in the background.

### Step 3 — Configure environment
```bash
# In the RAG MODEL folder:
copy .env.example .env
# The default .env uses Ollama — no changes needed if using local LLM.
# To use OpenAI instead, edit .env and set LLM_PROVIDER=openai + OPENAI_API_KEY=sk-...
```

### Step 4 — Start all services
```bash
# In the RAG MODEL folder (where docker-compose.yml is):
docker-compose up --build
```

First run takes 3–5 minutes (downloads images + embedding model).  
Subsequent starts take ~20 seconds.

### Step 5 — Open the app
| Service | URL |
|---------|-----|
| 🖥️ Chat UI | http://localhost:8501 |
| 🔧 API Docs | http://localhost:8000/docs |
| 🗄️ ChromaDB | http://localhost:8001 |

---

## Adding Your Own Medical Files

### Option A — Folder drop (recommended)
1. Place files in `data/patients/` or `data/providers/`
2. Click **⚡ Ingest All Files** in the sidebar, OR restart Docker
3. Files are indexed automatically

### Option B — Upload through the UI
Click the upload area in the sidebar → select your file → click **📤 Upload & Ingest**

### Supported formats
| Format | Best for |
|--------|---------|
| `.json` | Structured records (bills, provider info) — metadata extracted directly |
| `.csv` | Tabular data (billing exports, provider lists) |
| `.xlsx` | Excel spreadsheets |
| `.pdf` | Scanned or digital reports, discharge summaries |
| `.txt` | Free-text notes, policy documents |

### JSON field names the system recognises automatically
The ingestion pipeline maps these field names to searchable metadata:

| Metadata field | Recognised JSON keys |
|---------------|---------------------|
| `patient_name` | `patient_name`, `patient`, `name`, `full_name` |
| `patient_id` | `patient_id`, `mrn`, `member_id` |
| `date` | `date`, `bill_date`, `service_date`, `date_of_service` |
| `doc_type` | `doc_type`, `type`, `record_type` |
| `provider_name` | `provider_name`, `provider`, `physician`, `doctor` |
| `provider_npi` | `provider_npi`, `npi`, `npi_number` |
| `total_amount` | `total_amount`, `total`, `amount`, `bill_amount` |

---

## Example Queries

```
Get patient Alice Johnson's bill for 27-10-2025
What is the NPI number for Dr. Robert Chen?
Show me John Smith's medical record from September 2025
What is the total amount Alice owes?
Get provider information for NPI 9876543210
What is the date of birth for Dr. Sarah Williams?
Show all bills for patient P001
```

---

## Stopping the System
```bash
docker-compose down          # Stop containers (data preserved)
docker-compose down -v       # Stop and DELETE all indexed data
```

---

## Switching to OpenAI (instead of Ollama)

Edit `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```
Then restart: `docker-compose down && docker-compose up`

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| "Cannot reach backend" | Make sure Docker Desktop is running |
| Ollama errors | Run `ollama pull mistral` and keep Ollama open |
| Empty results | Click **Ingest All Files** in the sidebar |
| Slow first start | Normal — embedding model (~90MB) downloads once |
| Port conflict | Change ports in `docker-compose.yml` (left side of `X:Y`) |

---

## Project Structure

```
RAG MODEL/
├── docker-compose.yml       # Starts all 3 services together
├── .env.example             # Config template — copy to .env
├── .env                     # Your config (create from .env.example)
├── data/
│   ├── patients/            # Drop patient files here
│   └── providers/           # Drop provider files here
├── backend/
│   ├── main.py              # FastAPI API endpoints
│   ├── config.py            # All settings from .env
│   └── rag/
│       ├── llm_client.py    # Calls Ollama or OpenAI
│       ├── vectorstore.py   # ChromaDB operations (add, search, list)
│       ├── ingestion.py     # File loading + metadata extraction
│       ├── intent_parser.py # NL query → structured search criteria
│       └── retriever.py     # Full RAG pipeline
└── frontend/
    └── app.py               # Streamlit chat UI
```
