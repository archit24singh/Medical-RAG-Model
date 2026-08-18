"""
Build the baked ChromaDB index for the Streamlit demo.

Extracts text from one or more PDFs (PyMuPDF, page by page so code lists keep
their line structure), splits with the sentence-aware chunker, embeds with the
configured sentence-transformers model, and writes an embedded (persistent)
Chroma index to CHROMA_PERSIST_DIR.

The index this produces is committed to the repo, so the deployed Streamlit app
loads it directly and never runs the heavy ingestion pipeline at runtime.

Usage (from repo root):
    python scripts/build_index.py "bucket/uploads/ICD-10-CM FY25 Guidelines October 1, 2024-3.pdf"

Set BUILD_FRESH=1 to wipe the existing index directory before building.
"""
import hashlib
import os
import shutil
import sys

# ── Resolve paths and force the demo embedding + storage config BEFORE importing
#    config/vectorstore (pydantic Settings reads env at import time). ───────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
BACKEND   = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, BACKEND)

os.environ.setdefault("EMBEDDING_BACKEND", "sentence-transformers")
os.environ.setdefault("CHROMA_MODE", "persistent")
os.environ.setdefault("RAG_ONLY_MODE", "true")
os.environ.setdefault("CHROMA_COLLECTION", "medical_docs")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(REPO_ROOT, "chroma_index"))

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Sentence-aware sliding-window splitter (mirrors ingestion._split_text_into_chunks)."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    min_advance = chunk_size // 2

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end == len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        cut = end
        search_from = start + min_advance
        _SEPARATOR_TIERS = (
            ("\n\n",),
            (". ", ".\n", "? ", "?\n", "! ", "!\n"),
            ("\n",),
            (" ",),
        )
        for tier in _SEPARATOR_TIERS:
            best = -1
            best_sep = ""
            for sep in tier:
                pos = text.rfind(sep, search_from, end)
                if pos > best:
                    best, best_sep = pos, sep
            if best != -1:
                cut = best + len(best_sep)
                break

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        next_start = cut - chunk_overlap
        start = next_start if next_start > start else cut

    return chunks


def build(pdf_paths):
    import fitz  # PyMuPDF
    from rag.vectorstore import add_documents_batch, get_collection

    persist_dir = os.environ["CHROMA_PERSIST_DIR"]
    if os.environ.get("BUILD_FRESH") == "1" and os.path.isdir(persist_dir):
        print(f"BUILD_FRESH=1 → removing existing index at {persist_dir}")
        shutil.rmtree(persist_dir)

    ids, texts, metas = [], [], []
    for pdf_path in pdf_paths:
        fname = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        for pno in range(n_pages):
            page_text = doc[pno].get_text("text")
            if not page_text.strip():
                continue
            for ci, chunk in enumerate(split_text_into_chunks(page_text)):
                cid = hashlib.md5(f"{fname}:{pno + 1}:{ci}".encode()).hexdigest()
                ids.append(cid)
                texts.append(chunk)
                metas.append({
                    "file_name":    fname,
                    "file_type":    "pdf",
                    "page_number":  pno + 1,
                    "chunk_number": ci,
                })
        doc.close()
        print(f"  {fname}: {n_pages} page(s) → running chunk total {len(ids)}")

    if not ids:
        print("No text extracted — nothing to index.")
        return

    stored = add_documents_batch(ids, texts, metas)
    print(f"\nIndexed {stored} chunk(s) from {len(pdf_paths)} file(s).")
    print(f"Collection '{os.environ['CHROMA_COLLECTION']}' now has "
          f"{get_collection().count()} document(s).")
    print(f"Index written to: {persist_dir}")


if __name__ == "__main__":
    pdfs = sys.argv[1:]
    if not pdfs:
        print("usage: python scripts/build_index.py <pdf> [pdf ...]")
        raise SystemExit(1)
    build(pdfs)
