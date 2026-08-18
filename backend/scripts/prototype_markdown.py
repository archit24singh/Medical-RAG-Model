"""
Prototype runner for the Markdown extraction pipeline (PDF → Markdown → JSON).

Runs the FULL chain including the local LLM (Ollama) step, which the sandbox
can't do — so run this inside the backend container where Ollama is reachable:

    docker exec medical_rag_backend \
        python scripts/prototype_markdown.py bucket/uploads/invoice_51109301.pdf

Prints: the markdown handed to the LLM, the extracted JSON record, and the
coverage report (100% = nothing dropped; lists any omitted source tokens).
"""
import json
import os
import sys

# Ensure the app root (parent of scripts/) is importable when run as a file,
# e.g. `python scripts/prototype_markdown.py ...` — otherwise `import rag` fails.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.markdown_extractor import pdf_to_markdown, extract_record, coverage_check


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python scripts/prototype_markdown.py <pdf_path> [--show-markdown]")
        return 1
    path = argv[0]
    show_md = "--show-markdown" in argv

    md, raw = pdf_to_markdown(path)
    if show_md:
        print("===== MARKDOWN (LLM input) =====")
        print(md)
        print()

    print("===== EXTRACTED JSON RECORD =====")
    record = extract_record(path)          # ← calls the local LLM (Ollama)
    print(json.dumps(record, indent=2, ensure_ascii=False))

    print("\n===== COVERAGE CHECK (vs source text) =====")
    print(json.dumps(coverage_check(record, raw), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
