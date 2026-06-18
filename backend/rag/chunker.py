# DEPRECATED — no longer imported anywhere.
#
# Previously: LLM-based semantic chunker that made 1+ Ollama call(s) per chunk
# during ingestion (200+ calls for a 120-page PDF, causing timeouts and missing
# chunks).
#
# Replaced by: _split_text_into_chunks() in ingestion.py — a character-based
# sliding-window splitter with zero LLM calls (1000-char chunks / 200 overlap).
#
# File kept as an empty stub so any stale .pyc cache does not raise
# ModuleNotFoundError.  Safe to delete once __pycache__ is cleared.
