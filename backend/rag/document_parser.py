"""
Document parser — converts unstructured files into page-level Markdown.

Supported inputs
----------------
  PDF    — Docling + RapidOCR (handles scanned pages, tables, multi-column layouts)
  DOCX   — Docling (preserves headings, tables, lists)
  XML    — Docling
  PPTX   — Docling
  HTML   — Docling
  Images — Docling (PNG / JPG / TIFF / BMP)
  TXT    — direct read (no Docling overhead needed)

Fallback
--------
  If Docling is unavailable, PDFs fall back to pypdf (text-only, no OCR).
  All other formats raise an ImportError with an install hint.

Output
------
  List of page dicts:
    {
      "page_number": int,   # 1-based
      "text":        str,   # Markdown text for this page
      "word_count":  int,
    }

  Pages are separated by Docling's standard <!-- PageBreak --> marker.
  Single-page documents (DOCX, TXT, HTML) return a list with one entry.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions handled by the Docling pipeline
DOCLING_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".xml"}
IMAGE_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

# Full set of unstructured extensions (excluding .csv/.xlsx/.json — tabular pipeline)
UNSTRUCTURED_EXTENSIONS = DOCLING_EXTENSIONS | IMAGE_EXTENSIONS | {".txt"}

_PAGE_BREAK_RE = re.compile(r"<!--\s*PageBreak\s*-->", re.IGNORECASE)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_document(file_path: str, describe_images: bool = False) -> list[dict]:
    """
    Parse any supported document into a list of page-level dicts.

    Args:
        file_path:       Absolute path to the document.
        describe_images: Whether to run SmolVLM to describe embedded images.
                         Requires ENABLE_IMAGE_DESCRIPTION=true in config.

    Returns:
        List of {"page_number": int, "text": str, "word_count": int}
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    logger.info("Parsing document: %s (type=%s)", path.name, ext)

    if ext == ".txt":
        return _parse_txt(file_path)

    if ext in DOCLING_EXTENSIONS | IMAGE_EXTENSIONS:
        try:
            return _parse_with_docling(file_path, ext, describe_images)
        except ImportError:
            logger.error(
                "Docling is not installed. "
                "Add `docling>=2.5.0` to requirements.txt and rebuild."
            )
            # Graceful PDF fallback so the system keeps running without Docling
            if ext == ".pdf":
                logger.warning("Falling back to pypdf for %s (no OCR, no table detection)", path.name)
                return _parse_pdf_fallback(file_path)
            raise
        except Exception as exc:
            logger.warning("Docling failed for %s: %s — trying fallback", path.name, exc)
            if ext == ".pdf":
                return _parse_pdf_fallback(file_path)
            raise RuntimeError(f"Could not parse {path.name}: {exc}") from exc

    raise ValueError(f"Unsupported extension for document parser: {ext}")


# ── Docling pipeline ──────────────────────────────────────────────────────────

def _parse_with_docling(file_path: str, ext: str, describe_images: bool) -> list[dict]:
    """Run Docling on the file and return page-level markdown."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.datamodel.base_models import InputFormat

    # PDF gets a full pipeline with OCR + layout detection
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr           = True
    pipeline_options.ocr_options      = RapidOcrOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale     = 1.0

    # Optional SmolVLM image descriptions
    if describe_images:
        try:
            from docling.datamodel.pipeline_options import PictureDescriptionApiOptions  # noqa
            pipeline_options.do_picture_description = True
            logger.info("Image description enabled (SmolVLM)")
        except ImportError:
            logger.warning("SmolVLM not available — ENABLE_IMAGE_DESCRIPTION ignored")

    format_options = {}
    if ext == ".pdf":
        format_options[InputFormat.PDF] = PdfFormatOption(
            pipeline_options=pipeline_options
        )

    converter = DocumentConverter(format_options=format_options)
    result    = converter.convert(file_path)

    # Export the whole document to markdown
    full_markdown = result.document.export_to_markdown()

    pages = _split_by_page_breaks(full_markdown)
    logger.info("Docling parsed %d page(s) from %s", len(pages), Path(file_path).name)
    return pages


def _split_by_page_breaks(full_markdown: str) -> list[dict]:
    """
    Split Docling's full-document markdown on <!-- PageBreak --> markers.
    Returns a list of page dicts (1-based page numbers).
    If no markers found (e.g. DOCX), the whole document is page 1.
    """
    raw_pages = _PAGE_BREAK_RE.split(full_markdown)

    pages = []
    for i, text in enumerate(raw_pages, start=1):
        text = text.strip()
        if not text:
            continue
        pages.append({
            "page_number": i,
            "text":        text,
            "word_count":  len(text.split()),
        })

    if not pages:
        # Entire markdown was blank — return one empty page so caller doesn't crash
        pages = [{"page_number": 1, "text": full_markdown.strip(), "word_count": 0}]

    return pages


# ── Plain-text parser ─────────────────────────────────────────────────────────

def _parse_txt(file_path: str) -> list[dict]:
    """Read a plain-text file directly — no Docling needed."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read().strip()
    return [{"page_number": 1, "text": text, "word_count": len(text.split())}]


# ── pypdf fallback (PDF only, no OCR) ────────────────────────────────────────

def _parse_pdf_fallback(file_path: str) -> list[dict]:
    """
    Extract text from a PDF using pypdf.
    No OCR — scanned pages will appear blank.
    Use only when Docling is unavailable.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Neither Docling nor pypdf are available.") from exc

    reader = PdfReader(file_path)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "page_number": i,
                "text":        text,
                "word_count":  len(text.split()),
            })

    if not pages:
        logger.warning("pypdf extracted no text from %s (possibly scanned — needs OCR)", file_path)
        pages = [{"page_number": 1, "text": "", "word_count": 0}]

    return pages
