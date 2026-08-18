"""
CLI: folder-per-format ingest for structured documents.

Extracts a folder of same-format PDFs (1 LLM reference + deterministic rest),
stores each as JSONB in `documents`, and flattens header + line items into the
queryable staging layer. Run inside the backend container:

    docker exec medical_rag_backend python3 -m scripts.ingest_folder bucket/uploads
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.doc_store import ingest_folder


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python3 -m scripts.ingest_folder <folder> [coverage_threshold]")
        return 1
    folder = argv[0]
    threshold = float(argv[1]) if len(argv) > 1 else 90.0
    summary = ingest_folder(folder, coverage_threshold=threshold)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
