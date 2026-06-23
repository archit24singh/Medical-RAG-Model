"""
One-time migration script: purge CSV/XLSX/XLS row embeddings from ChromaDB.

Background
----------
The old ingestion pipeline wrote one ChromaDB document per spreadsheet row in
addition to storing the same rows in the relational database.  This duplication
caused structured billing data to pollute unstructured RAG searches — a bill
for "Alice Johnson" could surface in a general query about ICD codes.

After this script runs, ChromaDB contains ONLY unstructured knowledge PDFs.
All structured billing/clinical row data lives exclusively in PostgreSQL.

Usage
-----
    python scripts/purge_structured_chroma.py [options]

Options
-------
  --rebuild    Delete and recreate the entire collection (nuclear option).
               WARNING: also removes all PDF chunks — re-ingest everything after.
  --dry-run    Count and list matching IDs without deleting anything.
  --host       ChromaDB host (default: $CHROMA_HOST or 'localhost')
  --port       ChromaDB port (default: $CHROMA_PORT or 8001)
  --collection Collection name (default: $CHROMA_COLLECTION or 'medical_docs')

After running
-------------
Re-ingest your PDF/knowledge documents so ChromaDB is repopulated:
    curl -X POST http://localhost:8080/ingest
"""

import argparse
import os
import sys

BATCH_SIZE = 500
TABULAR_FILE_TYPES = ("csv", "xlsx", "xls")


def purge_tabular_docs(
    host:            str,
    port:            int,
    collection_name: str,
    dry_run:         bool = False,
    rebuild:         bool = False,
) -> int:
    """
    Delete all ChromaDB documents where metadata file_type is csv/xlsx/xls.

    Parameters
    ----------
    host, port, collection_name : ChromaDB connection details.
    dry_run  : When True, count matching docs but do not delete.
    rebuild  : When True, delete and recreate the entire collection.

    Returns
    -------
    Number of documents deleted (0 for dry-run or rebuild).
    """
    import chromadb

    client = chromadb.HttpClient(host=host, port=port)

    if rebuild:
        print(f"--rebuild: deleting and recreating collection '{collection_name}'")
        if not dry_run:
            try:
                client.delete_collection(collection_name)
                print(f"  Collection '{collection_name}' deleted.")
            except Exception as exc:
                print(f"  Warning: could not delete collection — {exc}")

            client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"  Collection '{collection_name}' recreated (empty).")
            print(
                "\nRebuild complete.\n"
                "Re-ingest PDF/knowledge documents to repopulate ChromaDB:\n"
                "    curl -X POST http://localhost:8080/ingest"
            )
        else:
            print("[dry-run] Would delete and recreate collection.")
        return 0

    collection = client.get_collection(collection_name)
    total_deleted = 0

    for file_type in TABULAR_FILE_TYPES:
        print(f"\nSearching for file_type='{file_type}' documents...", flush=True)
        try:
            result = collection.get(
                where={"file_type": {"$eq": file_type}},
                include=["metadatas"],
            )
        except Exception as exc:
            print(f"  Query failed: {exc}")
            continue

        ids = result.get("ids", [])
        if not ids:
            print(f"  No documents found with file_type='{file_type}'.")
            continue

        print(f"  Found {len(ids)} document(s) with file_type='{file_type}'.")

        if dry_run:
            print(f"  [dry-run] Would delete {len(ids)} document(s).")
            total_deleted += len(ids)
            continue

        # Delete in batches to stay within ChromaDB request-size limits.
        for batch_start in range(0, len(ids), BATCH_SIZE):
            batch = ids[batch_start : batch_start + BATCH_SIZE]
            collection.delete(ids=batch)
            total_deleted += len(batch)
            pct = 100.0 * (batch_start + len(batch)) / len(ids)
            print(f"  [{pct:5.1f}%] Deleted {len(batch)} doc(s) (batch starting at {batch_start}).")

    return total_deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge CSV/XLSX row embeddings from ChromaDB — keep only PDF chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Nuclear option: delete and recreate the entire collection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count and list matching documents without deleting anything.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CHROMA_HOST", "localhost"),
        help="ChromaDB host (default: $CHROMA_HOST or 'localhost')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CHROMA_PORT", "8001")),
        help="ChromaDB port (default: $CHROMA_PORT or 8001)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "medical_docs"),
        help="ChromaDB collection name (default: $CHROMA_COLLECTION or 'medical_docs')",
    )
    args = parser.parse_args()

    print(f"ChromaDB:   {args.host}:{args.port}")
    print(f"Collection: {args.collection}")
    if args.dry_run:
        print("Mode:       DRY RUN — no changes will be made.\n")
    elif args.rebuild:
        print("Mode:       REBUILD — entire collection will be recreated.\n")
    else:
        print(f"Mode:       PURGE file_types {TABULAR_FILE_TYPES}\n")

    try:
        deleted = purge_tabular_docs(
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            dry_run=args.dry_run,
            rebuild=args.rebuild,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal {'(would be) ' if args.dry_run else ''}deleted: {deleted} document(s).")
    if not args.dry_run and not args.rebuild and deleted > 0:
        print(
            "\nPurge complete.  ChromaDB now contains only unstructured documents.\n"
            "PDF chunks are preserved — no re-ingestion needed unless you also ran --rebuild."
        )


if __name__ == "__main__":
    main()
