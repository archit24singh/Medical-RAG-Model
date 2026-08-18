"""
Prediction evaluation harness (Phase 5).

Runs a DER prediction for each labeled example and reports Accuracy, Macro-F1,
and per-class F1 — the same metrics as the EHR-RAG paper (§4.1).

Labels file
-----------
CSV or JSON mapping a patient key to a ground-truth label, e.g.

  patient_key,label
  620853,long
  712044,short

or JSON: [{"patient_key": "620853", "label": "long"}, …]

Usage
-----
  python -m eval.evaluate --task long_length_of_stay --labels data/eval/los_labels.csv
  python -m eval.evaluate --task readmission_30d --labels labels.json --limit 50

IMPORTANT
---------
This harness is only meaningful with real labels. The Campbell demo data has no
outcome labels, so until labeled longitudinal data (EHRSHOT/MIMIC) is ingested,
metrics cannot be computed — the harness will say so rather than fabricate a score.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def load_labels(path: str) -> list[dict]:
    """Load labels from CSV (patient_key,label) or JSON list of {patient_key,label}."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return [{"patient_key": str(r["patient_key"]), "label": str(r["label"])} for r in data]
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pk = r.get("patient_key") or r.get("patient") or r.get("id")
            lab = r.get("label") or r.get("y") or r.get("outcome")
            if pk is None or lab is None:
                continue
            rows.append({"patient_key": str(pk).strip(), "label": str(lab).strip()})
    return rows


def evaluate(task: str, labels_path: str, limit: Optional[int] = None,
             prediction_time: Optional[str] = None) -> dict:
    from rag.orchestrator import predict as rag_predict

    labels = load_labels(labels_path)
    if not labels:
        return {"error": f"No labels loaded from {labels_path}."}
    if limit:
        labels = labels[:limit]

    y_true, y_pred, details = [], [], []
    for i, row in enumerate(labels, 1):
        pk = row["patient_key"]
        gold = row["label"]
        try:
            res = rag_predict(task, patient_key=pk, prediction_time=prediction_time)
            pred = res.get("prediction", "")
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", pk, exc)
            pred = ""
        y_true.append(gold)
        y_pred.append(pred)
        details.append({"patient_key": pk, "gold": gold, "pred": pred})
        logger.info("[%d/%d] patient=%s gold=%s pred=%s", i, len(labels), pk, gold, pred)

    return _score(y_true, y_pred, details, task)


def _score(y_true: list[str], y_pred: list[str], details: list[dict], task: str) -> dict:
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
    except ImportError:
        return {"error": "scikit-learn not installed — pip install scikit-learn", "details": details}

    labels_sorted = sorted(set(y_true) | set(y_pred))
    report = {
        "task": task,
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "per_class_f1": {
            lab: round(f1_score(y_true, y_pred, labels=[lab], average="macro", zero_division=0) * 100, 2)
            for lab in labels_sorted
        },
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "details": details,
    }
    return report


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="EHR-RAG prediction evaluation harness")
    ap.add_argument("--task", required=True, help="task key in prediction_tasks.yaml")
    ap.add_argument("--labels", required=True, help="path to labels CSV/JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prediction-time", default=None, help="ISO date anchor τ*")
    ap.add_argument("--out", default=None, help="optional path to write JSON report")
    args = ap.parse_args(argv)

    report = evaluate(args.task, args.labels, args.limit, args.prediction_time)

    if "error" in report:
        print("ERROR:", report["error"], file=sys.stderr)
        return 1

    print("\n=== EHR-RAG Evaluation ===")
    print(f"Task:     {report['task']}")
    print(f"N:        {report['n']}")
    print(f"Accuracy: {report['accuracy']}%")
    print(f"Macro-F1: {report['macro_f1']}%")
    print("Per-class F1:")
    for lab, f1 in report["per_class_f1"].items():
        print(f"  {lab}: {f1}%")
    print("\n" + report["classification_report"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
