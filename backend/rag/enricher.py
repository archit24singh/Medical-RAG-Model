# DEPRECATED — no longer imported anywhere.
#
# Previously: contextual enricher that called the LLM once per chunk to prepend
# a situating sentence.  Added 1 Ollama call per chunk (50-100 extra calls per
# PDF), delaying ingestion by minutes and causing enrichment failures that
# silently dropped context.
#
# Replaced by: a lightweight contextual chunk header prepended inline in
# _ingest_text_document() (ingestion.py) — format:
#     "<Document Title> [chunk N/M]:\n<chunk text>"
# Zero LLM calls, zero latency, deterministic output.
#
# File kept as an empty stub so any stale .pyc cache does not raise
# ModuleNotFoundError.  Safe to delete once __pycache__ is cleared.
