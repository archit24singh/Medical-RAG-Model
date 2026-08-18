"""
OCR for scanned PDFs and image files (Phase 1c).

The README historically advertised OCR (RapidOCR) but the capability had been
removed from the code, so scanned/image-only PDFs extracted nothing. This module
restores it using ``rapidocr-onnxruntime`` (ONNX-based, no torch dependency).

Usage
-----
  ocr_available()            → bool (engine importable)
  ocr_image(path)            → str  (OCR text of an image file)
  ocr_pdf_page(page)         → str  (render a fitz page to image, OCR it)
  page_needs_ocr(page)       → bool (True when the text layer is empty/sparse)

The engine is loaded lazily and cached for the process lifetime.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ENGINE = None
_ENGINE_TRIED = False

# Minimum characters of native text on a page before we consider OCR unnecessary.
_MIN_TEXT_LAYER_CHARS = 20


def _get_engine():
    global _ENGINE, _ENGINE_TRIED
    if _ENGINE is not None or _ENGINE_TRIED:
        return _ENGINE
    _ENGINE_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
        logger.info("RapidOCR engine loaded")
    except Exception as exc:
        logger.warning("RapidOCR unavailable (%s) — OCR disabled", exc)
        _ENGINE = None
    return _ENGINE


def ocr_available() -> bool:
    return _get_engine() is not None


def _ocr_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return concatenated text."""
    engine = _get_engine()
    if engine is None:
        return ""
    try:
        import numpy as np
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result, _ = engine(np.array(img))
        if not result:
            return ""
        # result: list of [box, text, score]
        return "\n".join(line[1] for line in result if len(line) >= 2)
    except Exception as exc:
        logger.warning("OCR failed (%s)", exc)
        return ""


def ocr_image(file_path: str) -> str:
    """OCR a standalone image file (.png/.jpg/.tiff/…)."""
    try:
        with open(file_path, "rb") as f:
            return _ocr_bytes(f.read())
    except Exception as exc:
        logger.warning("Could not read image %s (%s)", file_path, exc)
        return ""


def page_needs_ocr(page) -> bool:
    """True when a PDF page has little/no extractable text layer."""
    try:
        return len((page.get_text() or "").strip()) < _MIN_TEXT_LAYER_CHARS
    except Exception:
        return True


def ocr_pdf_page(page, zoom: float = 2.0) -> str:
    """Render a PyMuPDF page to a PNG image at `zoom` scale and OCR it."""
    if _get_engine() is None:
        return ""
    try:
        import fitz  # noqa: F401  (page is already a fitz.Page)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return _ocr_bytes(pix.tobytes("png"))
    except Exception as exc:
        logger.warning("PDF page OCR failed (%s)", exc)
        return ""


def ocr_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
    """OCR an entire (scanned) PDF, page by page. Returns concatenated text."""
    try:
        import fitz
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.warning("Could not open PDF for OCR %s (%s)", file_path, exc)
        return ""

    parts = []
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        parts.append(ocr_pdf_page(page))
    return "\n\n".join(p for p in parts if p)
