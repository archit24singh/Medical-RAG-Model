#!/usr/bin/env python3
"""
catalog_admin.py — CLI for managing structure-defined stream catalog.

Usage (always run inside the backend container):

    docker exec medical_rag_backend python scripts/catalog_admin.py list
    docker exec medical_rag_backend python scripts/catalog_admin.py label   <sig8> <human_label>
    docker exec medical_rag_backend python scripts/catalog_admin.py alias   <new_sig8> <target_sig8>
    docker exec medical_rag_backend python scripts/catalog_admin.py expose  <sig8>
    docker exec medical_rag_backend python scripts/catalog_admin.py unexpose <sig8>

Every command first re-reads data/sources.yaml and applies any config overlay
to source_catalog before performing its action (Fix C).  This means you can
edit sources.yaml and run any catalog_admin command to apply changes without
restarting the backend.

Security
--------
This is a CLI tool only — it runs via docker exec and requires shell access to
the container.  It is intentionally NOT exposed as a FastAPI endpoint.
PHI is present in the database; no unauthenticated HTTP surface here.
"""

import argparse
import json
import sys

# Ensure /app packages are importable when running from inside the container
sys.path.insert(0, "/app")

import yaml

from config import settings
from db.schema import (
    get_db,
    fetchone_dict,
    fetchall_dicts,
    _quote_identifier,
    _build_and_create_view,
)


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _read_sources_yaml() -> dict:
    """Load sources.yaml; return {} if not found."""
    try:
        with open(settings.SOURCES_YAML_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"  WARN: could not read sources.yaml: {exc}")
        return {}


def _apply_config_to_catalog(conn, sig: str, cfg: dict) -> None:
    """Write YAML config fields onto the source_catalog row for sig."""
    set_clauses: list[str] = []
    values: list = []

    def _add(col: str, val, as_jsonb: bool = False) -> None:
        if val is None:
            return
        if as_jsonb:
            set_clauses.append(f"{col} = %s::jsonb")
            values.append(json.dumps(val))
        else:
            set_clauses.append(f"{col} = %s")
            values.append(val)

    if "human_label"    in cfg: _add("human_label",    cfg["human_label"])
    if "load_mode"      in cfg: _add("load_mode",      cfg["load_mode"])
    if "natural_key"    in cfg: _add("natural_key",    cfg.get("natural_key"), as_jsonb=True)
    if "query_exposed"  in cfg: _add("query_exposed",  cfg["query_exposed"])
    if "column_mapping" in cfg: _add("column_mapping", cfg.get("column_mapping"), as_jsonb=True)

    if not set_clauses:
        return

    values.append(sig)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE source_catalog SET {', '.join(set_clauses)} "
            f"WHERE column_signature = %s",
            values,
        )
    conn.commit()


def _sync_yaml(conn) -> int:
    """
    Re-read sources.yaml and apply config overlay to source_catalog.
    Fix C: called at the start of every catalog_admin command.
    Returns the count of streams updated.
    """
    data    = _read_sources_yaml()
    streams = data.get("streams", [])
    updated = 0

    for cfg in streams:
        sig = (cfg.get("column_signature") or "").strip()
        if len(sig) != 32:          # reject 8-char prefixes — must be full md5
            if sig:
                print(f"  WARN: sources.yaml entry has invalid signature "
                      f"{sig!r} (must be 32-char md5) — skipped")
            continue

        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM source_catalog WHERE column_signature = %s",
                [sig],
            )
            exists = cur.fetchone()

        if not exists:
            print(f"  WARN: signature {sig[:8]} not yet in catalog "
                  f"(first ingest pending) — skipped")
            continue

        _apply_config_to_catalog(conn, sig, cfg)
        updated += 1

    if updated:
        print(f"  YAML sync: {updated} stream(s) updated from sources.yaml")
    return updated


# ── Lookup helpers ────────────────────────────────────────────────────────────

def _get_by_sig8(conn, sig8: str) -> dict | None:
    """
    Return the source_catalog row whose signature starts with sig8.
    Errors if zero or multiple matches.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM source_catalog "
            "WHERE column_signature LIKE %s",
            [sig8 + "%"],
        )
        rows = fetchall_dicts(cur)

    if not rows:
        print(f"  ERROR: no stream found with signature prefix {sig8!r}")
        return None
    if len(rows) > 1:
        print(
            f"  ERROR: {len(rows)} streams match prefix {sig8!r} — "
            "use more characters to disambiguate"
        )
        for r in rows:
            print(f"    {r['column_signature'][:8]}  {r.get('human_label') or '(unlabelled)'}")
        return None
    return rows[0]


def _parse_json_field(val) -> list | dict:
    if val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(conn) -> None:
    """Print all streams in source_catalog."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_signature, staging_table, aliased_to,
                   human_label, load_mode, query_exposed, view_name,
                   candidate_drift_of, rows_total, last_ingested_at
            FROM   source_catalog
            ORDER  BY first_ingested_at
            """
        )
        rows = fetchall_dicts(cur)

    if not rows:
        print("source_catalog is empty — ingest a CSV/Excel file first.")
        return

    print(f"\n{'SIG8':<10} {'LABEL':<24} {'TABLE':<20} {'MODE':<10} "
          f"{'EXPOSED':<8} {'ROWS':>8}  {'ALIAS / DRIFT / VIEW'}")
    print("-" * 110)
    for r in rows:
        sig8    = (r["column_signature"] or "")[:8]
        label   = (r.get("human_label") or "(unlabelled)")[:23]
        table   = (r.get("staging_table") or "(alias)")[:19]
        mode    = (r.get("load_mode") or "")[:9]
        exposed = "YES" if r.get("query_exposed") else "no"
        rows_n  = r.get("rows_total") or 0

        notes: list[str] = []
        if r.get("aliased_to"):
            notes.append(f"→ alias:{r['aliased_to'][:8]}")
        if r.get("view_name"):
            notes.append(f"view:{r['view_name']}")
        if r.get("candidate_drift_of"):
            notes.append(f"drift?{r['candidate_drift_of'][:8]}")

        print(
            f"{sig8:<10} {label:<24} {table:<20} {mode:<10} "
            f"{exposed:<8} {rows_n:>8}  {', '.join(notes)}"
        )
    print()


def cmd_label(conn, sig8: str, human_label: str) -> None:
    """Set the human_label for a stream."""
    stream = _get_by_sig8(conn, sig8)
    if not stream:
        return

    safe_label = human_label.lower().strip().replace(" ", "_")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_catalog SET human_label = %s "
            "WHERE column_signature = %s",
            [safe_label, stream["column_signature"]],
        )
    conn.commit()
    print(f"OK: {sig8} labelled as '{safe_label}'")


def cmd_alias(conn, new_sig8: str, target_sig8: str) -> None:
    """
    Route new_sig8's ingestion into target_sig8's staging table.

    Sets new_sig.aliased_to = target_sig, new_sig.staging_table = NULL.
    Runs ADD COLUMN IF NOT EXISTS for any columns in new_sig not in target.
    The target must own its own table (no chained aliases).
    """
    from db.schema import _sync_staging_columns

    new_row    = _get_by_sig8(conn, new_sig8)
    target_row = _get_by_sig8(conn, target_sig8)
    if not new_row or not target_row:
        return

    if new_row["column_signature"] == target_row["column_signature"]:
        print("  ERROR: cannot alias a stream to itself")
        return
    if target_row.get("aliased_to"):
        print(
            f"  ERROR: target {target_sig8} is itself an alias — "
            "chained aliases are not allowed"
        )
        return
    if new_row.get("aliased_to"):
        print(
            f"  WARN: {new_sig8} already has alias → "
            f"{(new_row['aliased_to'] or '')[:8]} — overwriting"
        )

    # Sync stored columns from new sig into target table
    new_stored    = set(_parse_json_field(new_row.get("stored_columns")))
    target_stored = set(_parse_json_field(target_row.get("stored_columns")))
    extra         = [c for c in new_stored if c not in target_stored]
    if extra:
        _sync_staging_columns(conn, target_row["staging_table"], extra)
        new_target_stored = sorted(target_stored | set(extra))
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE source_catalog SET stored_columns = %s::jsonb "
                "WHERE column_signature = %s",
                [json.dumps(new_target_stored), target_row["column_signature"]],
            )
        conn.commit()
        print(f"  Added {len(extra)} column(s) to {target_row['staging_table']}: {extra}")

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE source_catalog "
            "SET aliased_to = %s, staging_table = NULL "
            "WHERE column_signature = %s",
            [target_row["column_signature"], new_row["column_signature"]],
        )
    conn.commit()
    print(
        f"OK: {new_sig8} now routes to {target_row['staging_table']} "
        f"(owner: {target_sig8})"
    )


def cmd_expose(conn, sig8: str) -> None:
    """
    Create (or replace) the VIEW for a stream and mark it query_exposed = TRUE.

    Requires:
      • human_label set (catalog_admin label)
      • column_mapping set in sources.yaml

    The view uses safe_numeric / safe_date / safe_timestamptz helpers so dirty
    cells return NULL rather than erroring the whole query.

    After CREATE, GRANTs SELECT to POSTGRES_READONLY_ROLE if configured.
    """
    stream = _get_by_sig8(conn, sig8)
    if not stream:
        return

    if stream.get("aliased_to"):
        print(
            f"  ERROR: {sig8} is an alias row — expose its target "
            f"{(stream['aliased_to'] or '')[:8]} instead"
        )
        return

    try:
        view_name = _build_and_create_view(conn, stream)
        print(f"OK: view '{view_name}' created for stream {sig8}")
        if not (settings.POSTGRES_READONLY_ROLE or "").strip():
            print(
                "  NOTE: POSTGRES_READONLY_ROLE not set in .env — "
                "GRANT SELECT was skipped. The read-only role cannot yet "
                "query this view."
            )
    except ValueError as exc:
        print(f"  ERROR: {exc}")
    except Exception as exc:
        print(f"  ERROR: unexpected failure: {exc}")
        raise


def cmd_unexpose(conn, sig8: str) -> None:
    """
    Drop the view for a stream and mark it query_exposed = FALSE.
    Schema-linking will no longer surface this stream in text-to-SQL prompts.
    """
    stream = _get_by_sig8(conn, sig8)
    if not stream:
        return

    view_name = stream.get("view_name")
    if not view_name:
        print(f"  Stream {sig8} has no view — already unexposed")
        return

    qview = _quote_identifier(view_name)
    with conn.cursor() as cur:
        cur.execute(f"DROP VIEW IF EXISTS {qview}")
        cur.execute(
            "UPDATE source_catalog "
            "SET query_exposed = FALSE, view_name = NULL "
            "WHERE column_signature = %s",
            [stream["column_signature"]],
        )
    conn.commit()
    print(f"OK: view '{view_name}' dropped, stream {sig8} marked unexposed")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the structure-defined stream catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all streams in source_catalog")

    p_label = sub.add_parser("label", help="Set human_label for a stream")
    p_label.add_argument("sig8",        help="First 8 chars of column_signature")
    p_label.add_argument("human_label", help="Label (lowercase, underscores ok)")

    p_alias = sub.add_parser(
        "alias", help="Route new_sig8 ingestion into target_sig8's table"
    )
    p_alias.add_argument("new_sig8",    help="Signature prefix of the stream to alias")
    p_alias.add_argument("target_sig8", help="Signature prefix of the owning stream")

    p_expose = sub.add_parser("expose", help="Create VIEW and mark stream query-exposed")
    p_expose.add_argument("sig8", help="First 8 chars of column_signature")

    p_unexpose = sub.add_parser("unexpose", help="Drop VIEW and mark stream unexposed")
    p_unexpose.add_argument("sig8", help="First 8 chars of column_signature")

    args = parser.parse_args()

    with get_db() as conn:
        # Fix C: re-sync sources.yaml before every command
        _sync_yaml(conn)

        if args.command == "list":
            cmd_list(conn)
        elif args.command == "label":
            cmd_label(conn, args.sig8, args.human_label)
        elif args.command == "alias":
            cmd_alias(conn, args.new_sig8, args.target_sig8)
        elif args.command == "expose":
            cmd_expose(conn, args.sig8)
        elif args.command == "unexpose":
            cmd_unexpose(conn, args.sig8)


if __name__ == "__main__":
    main()
