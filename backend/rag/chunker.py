"""
Chunker — splits page-level Markdown into semantically coherent chunks.

Two-stage approach (mirrors the video pipeline):

  Stage 1 — Structural split
    Split on Markdown header lines (# / ## / ###).
    This respects the document's own section boundaries.

  Stage 2 — LLM semantic split (Mistral via Ollama)
    Pass the rough header-chunks to the LLM.
    The LLM decides where REAL section boundaries are and responds with
    only a JSON array of chunk-index split points — no text rewriting,
    minimal token usage.

    Hard rules enforced in the prompt:
      - NEVER split inside a table
      - NEVER split inside a list
      - NEVER split a billing line-item block

Token budget
------------
  Word count × 1.3 ≈ token estimate (good enough for planning, not billing).
  If the total document is ≤ CHUNK_MAX_TOKENS × 1.5 tokens it is stored as
  a single chunk — no LLM call needed.

Page tracking
-------------
  Pages are joined with embedded <!-- PAGE:N --> markers so that each chunk
  knows which page it came from. Markers are removed from the final text.

Output
------
  List of chunk dicts:
    {
      "text":           str,   # clean chunk text (no page markers)
      "page_number":    int,   # page the chunk starts on
      "chunk_index":    int,   # 0-based position within this document
      "total_chunks":   int,
      "word_count":     int,
      "token_estimate": int,
    }
"""

import json
import logging
import re

from rag.llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

# Embedded page marker used internally — stripped before storage
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE:(\d+)\s*-->", re.IGNORECASE)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict], filename: str = "") -> list[dict]:
    """
    Convert page-level dicts (from document_parser) into final chunks.

    Args:
        pages:    Output of document_parser.parse_document().
        filename: Used only for log messages.

    Returns:
        List of chunk dicts ready for the enricher and then ChromaDB.
    """
    if not pages:
        return []

    # ── Merge pages with embedded page markers ────────────────────────────────
    merged = _merge_pages(pages)
    total_words  = sum(p["word_count"] for p in pages)
    total_tokens = int(total_words * 1.3)

    logger.info(
        "Chunking '%s': %d page(s), ~%d words, ~%d tokens",
        filename, len(pages), total_words, total_tokens,
    )

    # ── Small document — store as single chunk ────────────────────────────────
    if total_tokens <= settings.CHUNK_MAX_TOKENS * 1.5 and len(pages) == 1:
        page = pages[0]
        logger.info("Document fits in one chunk — skipping LLM split")
        return [_make_chunk(page["text"], page["page_number"], 0, 1)]

    # ── Stage 1: structural header split ─────────────────────────────────────
    rough = _split_on_headers(merged)
    logger.info("Header split → %d rough chunk(s)", len(rough))

    # ── Stage 2: LLM semantic split ───────────────────────────────────────────
    if len(rough) > 1:
        try:
            final_texts = _llm_semantic_split(rough, filename)
        except Exception as exc:
            logger.warning("LLM semantic split failed (%s) — using header split", exc)
            final_texts = [c["text"] for c in rough]
    else:
        final_texts = [c["text"] for c in rough]

    # ── Attach page numbers, strip markers, return ───────────────────────────
    total = len(final_texts)
    chunks = []
    for i, text in enumerate(final_texts):
        clean_text, page_no = _strip_page_markers(text)
        if not clean_text.strip():
            continue
        chunks.append(_make_chunk(clean_text.strip(), page_no, i, total))

    # Re-number total_chunks now that we know the real count
    real_total = len(chunks)
    for i, c in enumerate(chunks):
        c["chunk_index"]  = i
        c["total_chunks"] = real_total

    logger.info("Final: %d chunk(s) for '%s'", real_total, filename)
    return chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _merge_pages(pages: list[dict]) -> str:
    """Join pages into one string, embedding page-number markers."""
    parts = []
    for p in pages:
        marker = f"<!-- PAGE:{p['page_number']} -->"
        parts.append(f"{marker}\n{p['text']}")
    return "\n\n".join(parts)


def _split_on_headers(text: str) -> list[dict]:
    """
    Split on Markdown header lines.
    Uses a look-ahead so the header stays attached to its section content.
    """
    # Split just before any line that starts with one or more '#'
    raw_splits = re.split(r"(?=\n#+\s)", "\n" + text)

    chunks = []
    for i, chunk in enumerate(raw_splits):
        chunk = chunk.strip()
        if not chunk:
            continue
        page_no = _peek_page(chunk, default=1)
        chunks.append({"index": i, "text": chunk, "page_number": page_no})

    return chunks or [{"index": 0, "text": text, "page_number": 1}]


def _llm_semantic_split(rough: list[dict], filename: str) -> list[str]:
    """
    Ask Mistral to identify where real section boundaries lie.

    The LLM sees abbreviated chunk previews (first 600 chars) and responds
    with a JSON array of chunk indices where a NEW section begins.
    This is minimal-token: no text is regenerated, only indices are returned.
    """
    if len(rough) <= 2:
        # Not worth an LLM call for 1-2 chunks
        return [c["text"] for c in rough]

    # Build the formatted chunk list for the prompt
    previews = "\n\n".join(
        f"[START_CHUNK_{c['index']}]\n{c['text'][:600]}\n[END_CHUNK_{c['index']}]"
        for c in rough
    )

    prompt = f"""You are an expert at splitting medical documents into coherent retrieval units.

The document has been pre-split into {len(rough)} rough chunks, each labelled START_CHUNK_N / END_CHUNK_N.

TASK: Decide where real section boundaries are.

STRICT RULES (never violate these):
- NEVER split inside a Markdown table (rows between | ... | belong together)
- NEVER split inside a bullet list or numbered list
- NEVER split a billing summary or line-item block
- Keep diagnosis + treatment plan in the same chunk
- Keep lab result headers + their values in the same chunk
- Each final chunk must make sense on its own when read in isolation

OUTPUT FORMAT: Respond with ONLY a valid JSON array of chunk indices where a NEW section begins.
- Always include 0 (the first chunk always starts a new section)
- Aim for sections of roughly {settings.CHUNK_MAX_TOKENS} tokens each
- Example: [0, 4, 9] → section 1 = chunks 0-3, section 2 = chunks 4-8, section 3 = chunks 9+

Document chunks:
{previews}

JSON split-point array:"""

    response = call_llm(prompt)

    # Parse the JSON array from the response
    match = re.search(r"\[[\d,\s]+\]", response)
    if not match:
        logger.warning("LLM returned no valid split array for '%s' — using header split", filename)
        return [c["text"] for c in rough]

    try:
        split_points = sorted(set(json.loads(match.group())))
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM split array for '%s'", filename)
        return [c["text"] for c in rough]

    # Ensure 0 is always a split point
    if not split_points or split_points[0] != 0:
        split_points = [0] + split_points

    # Group rough chunks by split points
    final_texts = []
    for i, start in enumerate(split_points):
        end = split_points[i + 1] if i + 1 < len(split_points) else len(rough)
        combined = "\n\n".join(c["text"] for c in rough[start:end])
        final_texts.append(combined)

    return final_texts


def _peek_page(text: str, default: int = 1) -> int:
    """Return the last PAGE marker found in `text` without removing it."""
    matches = _PAGE_MARKER_RE.findall(text)
    return int(matches[-1]) if matches else default


def _strip_page_markers(text: str) -> tuple[str, int]:
    """
    Remove all <!-- PAGE:N --> markers from text.
    Returns (clean_text, page_number_of_last_marker).
    """
    matches = _PAGE_MARKER_RE.findall(text)
    page_no = int(matches[-1]) if matches else 1
    clean   = _PAGE_MARKER_RE.sub("", text).strip()
    return clean, page_no


def _make_chunk(text: str, page_number: int, chunk_index: int, total_chunks: int) -> dict:
    word_count = len(text.split())
    return {
        "text":           text,
        "page_number":    page_number,
        "chunk_index":    chunk_index,
        "total_chunks":   total_chunks,
        "word_count":     word_count,
        "token_estimate": int(word_count * 1.3),
    }
