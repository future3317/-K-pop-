#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DIMS = ["accuracy", "evidence_grounding", "informativeness", "kpop_specificity", "fluency"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize blinded human evaluation scores.")
    ap.add_argument("--input", default="outputs/research/human_eval_sheet.csv")
    ap.add_argument("--output", default="outputs/research/human_eval_summary.md")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.input, "r", encoding="utf-8", newline="")))
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    for r in rows:
        for side in ["A", "B"]:
            sys = r.get(f"system_{side}", "")
            for dim in DIMS:
                try:
                    scores[sys][dim].append(float(r.get(f"{dim}_{side}", "")))
                except Exception:
                    pass
        pref = str(r.get("overall_preference", "")).strip().upper()
        if pref in {"A", "B"}:
            wins[r.get(f"system_{pref}", "")] += 1
        elif pref:
            wins["tie"] += 1
    lines = ["# Human Evaluation Summary", "", "| System | Dimension | Mean | Std | N |", "|---|---|---:|---:|---:|"]
    for sys, dims in scores.items():
        for dim in DIMS:
            vals = np.asarray(dims.get(dim, []), dtype=float)
            if vals.size:
                lines.append(f"| {sys} | {dim} | {vals.mean():.3f} | {vals.std():.3f} | {vals.size} |")
    lines += ["", "## Preference Wins", ""]
    for sys, n in wins.items():
        lines.append(f"- {sys}: {n}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
