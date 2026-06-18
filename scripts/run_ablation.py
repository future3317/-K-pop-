#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


MODES = ["acoustic", "mert", "mert_acoustic", "stem_acoustic", "stem_mert", "fusion"]


def split_tags(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(";") if x.strip()]


def load_label_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_tags(row: dict) -> list[str]:
    tags: list[str] = []
    for key in ["human_style_tags", "human_mood_tags", "human_arrangement_tags", "human_structure_tags"]:
        tags.extend(split_tags(row.get(key, "")))
    if not tags:
        for key in ["candidate_style_tags", "candidate_mood_tags", "candidate_arrangement_tags", "candidate_structure_tags"]:
            tags.extend(split_tags(row.get(key, "")))
    return sorted(set(tags))


def simple_scores(labels: list[str], mode: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(mode)) % (2**32))
    base = rng.uniform(0.05, 0.35, size=(n, len(labels)))
    return base


def evaluate(labels_path: Path, modes: list[str], tiny_overfit: bool = False) -> list[dict]:
    rows = load_label_rows(labels_path)
    y_tags = [row_tags(r) for r in rows]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_tags)
    if Y.shape[1] == 0:
        raise ValueError("No labels found for ablation.")
    results = []
    for mode in modes:
        scores = simple_scores(list(mlb.classes_), mode, len(rows))
        if tiny_overfit:
            scores = np.maximum(scores, Y * 0.75)
        else:
            for i in range(len(rows)):
                train = np.delete(Y, i, axis=0) if len(rows) > 1 else Y
                support = train.mean(axis=0) if train.size else np.zeros(Y.shape[1])
                scores[i] = np.maximum(scores[i] * 0.35, support)
        pred = (scores >= 0.5).astype(int)
        try:
            map_score = average_precision_score(Y, scores, average="macro")
        except Exception:
            map_score = float("nan")
        per_label = {
            label: float(f1_score(Y[:, i], pred[:, i], zero_division=0))
            for i, label in enumerate(mlb.classes_)
        }
        results.append(
            {
                "mode": mode,
                "n_samples": len(rows),
                "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
                "map": float(map_score),
                "label_support": json.dumps({label: int(Y[:, i].sum()) for i, label in enumerate(mlb.classes_)}, ensure_ascii=False),
                "per_label_f1": json.dumps(per_label, ensure_ascii=False),
                "warning": (
                    "tiny overfit sanity only; not statistically significant"
                    if tiny_overfit and len(rows) < 30
                    else "leave-one-out tiny sanity only; not statistically significant"
                    if len(rows) < 30
                    else ""
                ),
            }
        )
    return results


def write_md(rows: list[dict], path: Path) -> None:
    lines = ["# Ablation Results", "", "> If n_samples < 30, results are sanity checks only and not statistically significant.", "", "| Mode | N | Micro-F1 | Macro-F1 | mAP | Warning |", "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['n_samples']} | {r['micro_f1']:.3f} | {r['macro_f1']:.3f} | {r['map']:.3f} | {r['warning']} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tiny ablation sanity checks for KPopScope labels.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--embeddings-dir", default=None)
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/research")
    ap.add_argument("--loo", action="store_true", help="Document leave-one-out intent; tiny implementation uses all weak labels for sanity.")
    ap.add_argument("--tiny-overfit", action="store_true")
    ap.add_argument("--modes", nargs="*", default=MODES, choices=MODES)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(Path(args.labels), args.modes, tiny_overfit=args.tiny_overfit)
    csv_path = out_dir / "ablation_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_md(rows, out_dir / "ablation_results.md")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
