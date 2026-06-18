# DEPRECATED — no longer imported anywhere.
#
# Previously: Docling-based document parser for PDF/DOCX/XML that required
# docling>=2.5.0 and rapidocr-onnxruntime (heavy dependencies, slow startup).
#
# Replaced by: _extract_text_from_file() in ingestion.py — uses PyMuPDF (fitz)
# for PDF text extraction with zero deep-learning dependencies.
#
# File kept as an empty stub so any stale .pyc cache does not raise
# ModuleNotFoundError.  Safe to delete once __pycache__ is cleared.
