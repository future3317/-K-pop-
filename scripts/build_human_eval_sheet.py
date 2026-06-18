#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build blinded A/B human evaluation sheet for report modes.")
    ap.add_argument("--analysis-dir", default="outputs/twice")
    ap.add_argument("--output", default="outputs/research/human_eval_sheet.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for track in sorted(Path(args.analysis_dir).iterdir()) if Path(args.analysis_dir).exists() else []:
        if not track.is_dir():
            continue
        tag = track / "report_tag_only.md"
        ev = track / "report_evidence_grounded.md"
        if not tag.exists() or not ev.exists():
            continue
        pair = [("tag_only", tag), ("evidence_grounded", ev)]
        rng.shuffle(pair)
        rows.append({
            "track_id": track.name,
            "report_A": pair[0][1].as_posix(),
            "report_B": pair[1][1].as_posix(),
            "system_A": pair[0][0],
            "system_B": pair[1][0],
            "accuracy_A": "", "accuracy_B": "",
            "evidence_grounding_A": "", "evidence_grounding_B": "",
            "informativeness_A": "", "informativeness_B": "",
            "kpop_specificity_A": "", "kpop_specificity_B": "",
            "fluency_A": "", "fluency_B": "",
            "overall_preference": "",
            "notes": "",
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["track_id", "report_A", "report_B", "system_A", "system_B", "accuracy_A", "accuracy_B", "evidence_grounding_A", "evidence_grounding_B", "informativeness_A", "informativeness_B", "kpop_specificity_A", "kpop_specificity_B", "fluency_A", "fluency_B", "overall_preference", "notes"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
