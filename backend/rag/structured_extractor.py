# DEPRECATED — no longer imported anywhere.
#
# Previously: per-chunk LLM extraction that called Ollama once per chunk to
# pull patient/provider/billing data into SQLite.  For a 120-page PDF this
# produced 100+ Ollama calls, overwhelming the local model and causing the
# ingestion pipeline to stall or drop chunks.
#
# Replaced by: direct columnar SQL ingestion in _write_rows_to_sql() (ingestion.py)
# which maps spreadsheet columns deterministically to SQLite fields — zero LLM
# calls, scales to tens of thousands of rows, and captures every field.
#
# File kept as an empty stub so any stale .pyc cache does not raise
# ModuleNotFoundError.  Safe to delete once __pycache__ is cleared.
