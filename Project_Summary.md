# Project Handoff Summary

_A working record of the system, the decisions behind it, what actually works,
and what is still unproven. Written to be read by the next person (or future
self) picking this up. Honest over flattering._

---

## 1. What this project is

A **local-first, privacy-preserving RAG application** for structured documents.
Target users (see `goal.md`): individuals / small businesses / clinics who want
RAG over **medical structured data, personal business documents (invoices), and
general knowledge**, running **on their own hardware** (incl. mobile/iOS later),
with **all storage on-device and no cloud dependency by default**.

It began as a medical billing/records Q&A RAG and, over this work, grew two
distinct capabilities:
1. **EHR-RAG prediction pillars** (from the paper `2601.md`) — event/time-aware
   retrieval + iterative + dual-path reasoning for clinical prediction.
2. **Structured-document extraction & querying** (invoices) — the bulk of the
   later work, which is where the real product focus landed.

---

## 2. Tech stack (current)

- **Backend**: FastAPI (`backend/main.py`)
- **Vector store**: ChromaDB (semantic/RAG)
- **Structured store**: PostgreSQL 16 (facts, staging, documents)
- **Embeddings**: `nomic-embed-text` (via Ollama)
- **Reranker**: CrossEncoder `ms-marco-MiniLM-L-6-v2`
- **LLM**: **`nvidia/nemotron-3-ultra`** (cloud API, via OpenRouter — `LLM_PROVIDER=openai`).
  Local `devstral:24b` via Ollama remains a one-line fallback. See model history below.
- **SQL query engine**: **LangChain SQL agent** (primary), text-to-SQL + deterministic shapes (legacy fallbacks)
- **PDF/structure**: pdfplumber (tables), PyMuPDF (text), RapidOCR (scans)
- **Validation**: sqlglot
- **Frontend**: Streamlit (demo UI)
- **Deploy**: docker-compose (postgres + chromadb + backend + frontend; Ollama on host for embeddings only)

**No LangChain was used anywhere except the new SQL agent.** No LlamaIndex.

### Model history (important context)
`mistral` → `llama3.1:8b` → (`kimi-k2.7-code:cloud`, briefly) → `devstral:24b`
→ **`nvidia/nemotron-3-ultra` (OpenRouter API, 2026-08-18)**.
Each step up was driven by the previous model being **too unreliable for the
reasoning + SQL generation**. devstral drove the LangChain SQL agent's ReAct
loop but was tight on 24 GB unified memory (Q4 ~14 GB) and slow. The move to a
hosted Nemotron 3 Ultra trades the on-device/no-egress property for
tool-calling reliability and speed.

**Consequence to keep in view: the no-PHI-egress property is gone by default.**
Prompts (clinical notes, patient names, billing detail, retrieved chunks) go to
OpenRouter. Real patient data needs a BAA + logging/training exclusions first.
`LLM_PROVIDER=ollama` restores the fully local path unchanged.

### Provider wiring (2026-08-18)
- `backend/rag/llm_client.py` — `_call_openai` now builds a cached `OpenAI`
  client with `base_url`, timeout, retries, and OpenRouter attribution headers;
  falls back to plain-text + lenient JSON parsing when a model rejects
  `response_format=json_object`. Cache key includes the endpoint host.
- `backend/rag/sql_agent.py` — `_build_llm()` returns `ChatOpenAI` (base_url'd)
  when `LLM_PROVIDER=openai`, `ChatOllama` when local. Needs `langchain-openai`.
- `backend/config.py` — added `OPENAI_BASE_URL`, `OPENAI_TIMEOUT_S`,
  `OPENAI_MAX_RETRIES`, `OPENROUTER_*`; `LLM_PROVIDER` default flipped to
  `openai`; startup warning when the key is missing.
- `/health` now reports `llm_endpoint`, `llm_api_key_configured`, `embedding_model`.
- Embeddings unchanged: `nomic-embed-text` on Ollama — moving them would
  invalidate every ChromaDB vector and force a full re-ingest.

---

## 3. What was built (chronological)

### A. EHR-RAG pillars (from paper `2601.md`)
- **Phase 0** — LLM upgrade, `call_llm_json()` (JSON mode), response cache, keep-alive.
- **Phase 1** — Canonical **event adapter** (`rag/events.py`, `data/concept_map.yaml`): maps any source schema → `(concept, value, timestamp, type)` events, config-driven (no hardcoded columns). Timestamp-preserving chunker (`rag/event_chunker.py`). New `events` table.
- **Phase 1b/1c** — PDF table extraction to events (`rag/pdf_tables.py`); OCR for scans (`rag/ocr.py`).
- **Phase 2 — ETHER** (`rag/ether.py`): U-shaped time-aware retrieval + numeric indicator trajectories.
- **Phase 3 — AIR** (`rag/air.py`): iterative retrieval with LLM sufficiency check + query refinement (capped).
- **Phase 4 — DER** (`rag/der.py`): factual/counterfactual dual-path → prediction + rationale. Task registry `data/prediction_tasks.yaml`.
- **Phase 5 — Eval harness** (`eval/evaluate.py`): Accuracy / Macro-F1 / per-class F1 (needs labeled data to be meaningful).
- **Phase 6 — Scale**: Postgres connection pool, background ingest jobs (`rag/jobs.py`).

### B. Structured document (invoice) pipeline — the main later effort
- **Patient/entity profile** (`rag/profiles.py`): schema-on-read "all columns for an entity" from staging + dynamic indexes.
- **pdfplumber → staging** (`rag/pdf_tables.py`): invoice tables into the tabular staging pipeline.
- **Bridge fix** (`db/schema.py` `auto_expose_streams`): auto-create `v_*` views over staging so the SQL engine can see ingested data.
- **Markdown extraction pipeline** (`rag/markdown_extractor.py`): PDF → clean Markdown (column-split header + tables) → LLM exhaustive schema-less JSON → coverage check. Prototype: `scripts/prototype_markdown.py`.
- **Folder-per-format extraction** (`rag/folder_extractor.py`): 1 LLM reference call per folder + **deterministic passes** for the rest (100% coverage on the test invoices, ~21 ms/file). Prototype: `scripts/prototype_folder.py`.
- **Storage wiring** (`rag/doc_store.py`): full record → `documents` JSONB table; flattened header + line items → staging → auto-exposed views. `/ingest/folder` endpoint + `scripts/ingest_folder.py`. Folder-aware auto-ingest (`STRUCTURED_DOC_FOLDERS`).
- **Deterministic query shapes** (`rag/query_shapes.py`): generic distinct/count/aggregate templates resolving table+column dynamically — schema-agnostic across invoices/providers/credentialing.
- **Robust text-to-SQL** (`rag/text_to_sql.py`): Planner → Writer → sqlglot validate → self-repair.
- **LangChain SQL agent** (`rag/sql_agent.py`): **now the primary query path** — introspects the catalog itself, ReAct tool loop over Postgres, runs as a **read-only role** (`rag_readonly`, auto-created in `init_db`). RAG is the semantic fallback.

---

## 4. Key design decisions & rationale

- **Dual store (Chroma + Postgres).** Pure vector retrieval can't do completeness, aggregation, or exact numeric/date/ID filters (dense embeddings are number-insensitive), and it fabricates when the LLM does math over top-k. Structured queries go to SQL; semantic queries to vectors.
- **Schema-on-read, config-driven everywhere.** No hardcoded columns. `concept_map.yaml` holds synonyms/measures/dimensions; staging auto-creates `stg_*` tables by column signature; new document types need config, not code.
- **Extraction: deterministic-first, LLM-as-fallback.** The folder-per-format approach runs the LLM once per format to "learn," then parses the rest deterministically → scales to millions of docs locally (vs. ~70 days if every doc hit the LLM).
- **Querying went through three philosophies** (each because the prior failed): (1) deterministic query shapes, (2) robust text-to-SQL, (3) **LangChain SQL agent** as primary — the user's chosen direction, accepting "much better but still not guaranteed on a local model."
- **Read-only role for the agent.** The agent writes arbitrary SQL; it connects as a SELECT-only Postgres role so it can't write/drop.

---

## 5. Current state — what works vs. what's unproven

### Works (validated in-session)
- **Extraction on clean, digital, single-format invoices**: deterministic parse, seller/client with addresses, line items, totals — 100% token-coverage on ~100 docs, 0 LLM calls after the reference.
- **Folder-per-format scaling**: 100 invoices extracted deterministically, ~21 ms/file; only 1–2 LLM calls total.
- **Deterministic query shapes**: correct SQL generated across 3 unrelated schemas (verified with mocked catalogs).
- **Storage**: `documents` JSONB + flattened staging + auto-exposed views compile and the flatten logic is verified.

### Unproven / not validated
- **The LangChain SQL agent path was NOT run** in-session (no LangChain/Ollama/DB in the dev sandbox) — only compile-checked. LangChain's API varies by version; the agent may need a tweak on first real run (it falls back across `agent_type`s and then to RAG rather than crashing).
- **Retrieval accuracy at scale is unmeasured** — there is no labeled query→gold test set. Any "100%" figure refers to *extraction coverage on a best-case subset*, not retrieval.
- **Prediction (ETHER/AIR/DER) is unvalidated** — no labeled longitudinal/clinical data was ever ingested, so Macro-F1 was never computed. Runs, but unscored.
- **Model quality is qualitative only** — the eval harness exists but was never run on labeled data.

---

## 6. Known issues / honest limitations

- **LLM-generated SQL hallucinates on small/mid models.** The old fallback invented a `$1,234.00` total. devstral + the agent's schema-introspection + sample rows should reduce this, but it is **not eliminated** — join queries and exact date-format casts (`DD/MM/YYYY` text) are the weak spots.
- **Join + date queries** ("invoice items for date X") need `items ⋈ headers` — the recommended fix (not yet done) is to **denormalize** header fields onto item rows so it's a single-table filter. Approved in principle, not implemented.
- **Deterministic shapes are greedy** — "all invoice items for date X" wrongly triggered the `distinct` shape. A `filter` shape + a guard were proposed, not built.
- **Scanned/OCR and unseen layouts** degrade extraction sharply; the parser is format-specific until it sees a new format.
- **`invoice_no` shows as float** (`51109302.0`) — pandas cast; cosmetic, unfixed.
- **Exposing all tables (incl. raw staging) to the agent** makes the schema large — may slow/confuse the agent; `SQL_AGENT_EXPOSE_ALL_TABLES=false` scopes it to clean views if needed.
- **24B model on 24 GB** is tight and multi-call (ReAct loop) → high per-query latency; in tension with the minimum-hardware/iOS goal.
- **Grounding checker is itself an LLM** — an unreliable guard on an unreliable generator. Not safe for real clinical use; this is a research prototype.

---

## 7. Data & config layout

- `data/concept_map.yaml` — column→concept synonyms, measures, dimensions, field aliases, document keys.
- `data/prediction_tasks.yaml` — clinical prediction task registry.
- `data/sources.yaml`, `data/sql_schema_metadata.yaml` — stream config + text-to-SQL schema.
- `data/eval/` — labels for the eval harness (example only).
- `bucket/uploads/` — loose files (guidelines PDF, Campbell claims xlsx) → per-file path.
- `bucket/uploads/invoice1/` — structured format folder → folder-per-format pipeline (configured via `STRUCTURED_DOC_FOLDERS`).
- `goal.md` — product vision/constraints (git-ignored).

### Key Postgres tables
- `patients / providers / records / admissions` — curated medical.
- `events` — canonical events (ETHER/AIR/DER).
- `stg_*` — schema-on-read staging (per column signature); `v_auto_*` — auto-exposed views.
- `documents` — extracted structured records (JSONB), folder-per-format pipeline.
- `source_catalog` — stream registry.

---

## 8. How to run

```bash
# one-time: pull the embedding model on the host (Ollama runs on host, not in Docker)
ollama pull nomic-embed-text
# ollama pull devstral:24b          # only needed for the LLM_PROVIDER=ollama fallback

# put the OpenRouter key in .env (generation runs on the API now):
#   OPENAI_API_KEY=sk-or-v1-…

# full reset + rebuild (clears stale data; creates read-only role on startup)
docker compose down -v && docker compose up -d --build

# folder-per-format ingest runs automatically for STRUCTURED_DOC_FOLDERS on startup;
# to run manually:
docker exec medical_rag_backend python3 -m scripts.ingest_folder bucket/uploads/invoice1

# watch it
docker logs -f medical_rag_backend | grep -iE "sql agent|ingest|documents"
```

UI: `http://localhost:8501` · API: `http://localhost:8080/docs`.

Prototypes: `scripts/prototype_markdown.py`, `scripts/prototype_folder.py`.

---

## 9. Next steps (in priority order)

1. **Validate the LangChain SQL agent on-device** — first real run; fix the LangChain API call if the build errors (check which `agent_type` succeeds in logs).
2. **Denormalize item rows** (header fields onto each line item) so "items for date/client" become single-table filters — removes the join dependency that keeps failing.
3. **Build a labeled query→gold test set** and measure the agent's real hit rate by query type. Stop quoting extraction coverage as retrieval accuracy.
4. **Add the missing `filter`/date-range shapes** + guard the greedy `distinct` shape (if keeping deterministic routes as a fast tier).
5. **Decide the model story for the product vs. demo** — the demo now runs on hosted Nemotron 3 Ultra (fast, reliable tool-calling, but PHI egress); the on-device product still needs a smaller local model + more deterministic coverage. Benchmark cloud vs. `LLM_PROVIDER=ollama` head-to-head before committing.
6. **The research question worth pursuing** (if this becomes a contribution): _how much structured-document querying can be made deterministic (zero-LLM), and how does that frontier move as the model shrinks, under an on-device no-egress constraint?_ — measurable as deterministic-coverage curves vs. model size.

---

## 10. One-paragraph status

The extraction side is solid for clean, same-format digital documents and scales
locally. The querying side has been rebuilt three times and now rests on a
LangChain SQL agent, now pointed at hosted **Nemotron 3 Ultra** (OpenRouter),
still **implemented but not yet validated end-to-end**. Nothing about retrieval accuracy or clinical
prediction is measured — those claims are qualitative. Treat this as a capable
**prototype** with a clear, honest path to a real evaluation, not a finished or
safe-for-clinical-use system.
