"""
Prototype runner for folder-per-format extraction.

All same-format documents live in one folder; the FIRST file uses the LLM
(reference), the rest are deterministic (no LLM) with a coverage-check LLM
fallback. Run inside the backend container (needs Ollama for the reference call):

    docker exec medical_rag_backend \
        python3 -m scripts.prototype_folder bucket/uploads

Prints per-file method + coverage, the call-count stats, and elapsed time.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.folder_extractor import extract_folder


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python3 -m scripts.prototype_folder <folder> [coverage_threshold]")
        return 1
    folder = argv[0]
    threshold = float(argv[1]) if len(argv) > 1 else 90.0

    t0 = time.time()
    result = extract_folder(folder, coverage_threshold=threshold)
    dt = time.time() - t0

    print("===== PER-FILE METHOD =====")
    for r in result["records"]:
        print(f"  {r.get('_source_file'):40}  {r.get('_method')}")

    print("\n===== STATS =====")
    print(json.dumps(result["stats"], indent=2))
    print(f"elapsed: {dt:.1f}s   ({dt / max(len(result['records']),1):.2f}s/file avg)")

    if result["records"]:
        print("\n===== SAMPLE RECORD (2nd file, deterministic if it qualified) =====")
        sample = result["records"][1] if len(result["records"]) > 1 else result["records"][0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
