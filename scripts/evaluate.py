#!/usr/bin/env python
"""Evaluate KPopScope classifiers or tiny label sets.

Two modes are supported:

1. Checkpoint mode, compatible with the original script:
   python scripts/evaluate.py --embeddings data/kpop_embeddings.npz --checkpoint checkpoints/model.pt

2. Tiny sanity mode for local TWICE labels:
   python scripts/evaluate.py --labels data/local/twice_pseudo_labels.csv --loo

Tiny mode is intentionally conservative. It reports macro/micro F1 for a
deterministic weak-label sanity check and prints an explicit small-sample warning.
It does not claim generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from kpop_scope.models.kpop_classifier import KPopClassifier


def parse_tags(text: str) -> list[str]:
    return [t.strip() for t in str(text or "").split(";") if t.strip()]


def tags_from_label_row(row: dict) -> list[str]:
    human_keys = [
        "human_style_tags",
        "human_mood_tags",
        "human_arrangement_tags",
        "human_structure_tags",
    ]
    pseudo_keys = [
        "candidate_style_tags",
        "candidate_mood_tags",
        "candidate_arrangement_tags",
        "candidate_structure_tags",
    ]
    tags: list[str] = []
    for key in human_keys:
        tags.extend(parse_tags(row.get(key, "")))
    if not tags:
        for key in pseudo_keys:
            tags.extend(parse_tags(row.get(key, "")))
    return sorted(set(tags))


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    payload = np.load(args.embeddings, allow_pickle=True)
    X = payload["X"].astype("float32") if "X" in payload.files else np.zeros((0, 0), dtype="float32")
    A = payload["A"].astype("float32") if "A" in payload.files else np.zeros((len(X), 0), dtype="float32")
    tags = [parse_tags(t) for t in payload["tags_text"]]
    clf = KPopClassifier(args.checkpoint, device=args.device)
    mlb = MultiLabelBinarizer(classes=clf.labels)
    Y = mlb.fit_transform(tags)

    probs = []
    for i in range(len(tags)):
        emb = X[i] if X.shape[1] else None
        acoustic = A[i] if A.shape[1] else None
        probs.append(clf.predict_proba(embedding=emb, acoustic=acoustic))
    probs = np.stack(probs)
    preds = (probs >= args.threshold).astype("int32")
    print("Macro F1:", f1_score(Y, preds, average="macro", zero_division=0))
    print("Micro F1:", f1_score(Y, preds, average="micro", zero_division=0))
    try:
        print("Macro ROC-AUC:", roc_auc_score(Y, probs, average="macro"))
    except Exception as exc:
        print("Macro ROC-AUC unavailable:", exc)
    print(classification_report(Y, preds, target_names=clf.labels, zero_division=0))
    return 0


def evaluate_tiny_labels(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(open(args.labels, "r", encoding="utf-8", newline="")))
    y_tags = [tags_from_label_row(r) for r in rows]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_tags)
    if Y.shape[1] == 0:
        raise SystemExit("No labels found in --labels.")

    # Deterministic leave-one-out style sanity baseline: predict labels that have
    # appeared in the remaining samples. Only --tiny-split enables an overfit
    # smoke path that can see the row's own weak labels.
    probs = np.zeros_like(Y, dtype=float)
    for i in range(len(rows)):
        train = np.delete(Y, i, axis=0) if len(rows) > 1 else Y
        support = train.mean(axis=0) if train.size else np.zeros(Y.shape[1])
        probs[i] = np.maximum(support, Y[i] * 0.75) if args.tiny_split else support
    preds = (probs >= args.threshold).astype(int)

    macro = f1_score(Y, preds, average="macro", zero_division=0)
    micro = f1_score(Y, preds, average="micro", zero_division=0)
    try:
        auc = roc_auc_score(Y, probs, average="macro")
    except Exception:
        auc = float("nan")
    support = {label: int(Y[:, j].sum()) for j, label in enumerate(mlb.classes_)}
    result = {
        "n_samples": len(rows),
        "n_labels": len(mlb.classes_),
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "macro_roc_auc": float(auc),
        "label_support": support,
        "warning": (
            "Tiny overfit smoke check only; not statistically significant."
            if args.tiny_split and len(rows) < 30
            else "Tiny leave-one-out sanity check only; not statistically significant."
            if len(rows) < 30
            else ""
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(classification_report(Y, preds, target_names=list(mlb.classes_), zero_division=0))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate KPopScope checkpoints or tiny local label sets.")
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loo", action="store_true", help="Use leave-one-out style tiny sanity evaluation for --labels.")
    ap.add_argument("--tiny-split", action="store_true", help="Alias for tiny sanity evaluation.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.checkpoint and args.embeddings:
        return evaluate_checkpoint(args)
    if args.labels:
        return evaluate_tiny_labels(args)
    ap.error("Use either --embeddings + --checkpoint or --labels.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
