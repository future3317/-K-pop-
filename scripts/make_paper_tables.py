#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, "r", encoding="utf-8", newline="")))


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect research outputs into Markdown tables for the paper.")
    ap.add_argument("--research-dir", default="outputs/research")
    ap.add_argument("--manifest", default="data/local/twice_manifest.csv")
    ap.add_argument("--labels", default="data/local/twice_pseudo_labels.csv")
    ap.add_argument("--output", default="outputs/research/paper_tables.md")
    args = ap.parse_args()
    research = Path(args.research_dir)
    manifest = read_rows(Path(args.manifest))
    labels = read_rows(Path(args.labels))
    ablation = read_rows(research / "ablation_results.csv")
    lines = ["# Paper Tables", "", "## Dataset Statistics", "", "| Split | Tracks |", "|---|---:|", f"| demo | {len(manifest)} |", "", "## Label Distribution", "", "| Label | Count |", "|---|---:|"]
    counts: dict[str, int] = {}
    for r in labels:
        for key in ["candidate_style_tags", "candidate_mood_tags", "candidate_arrangement_tags", "candidate_structure_tags", "human_style_tags", "human_mood_tags", "human_arrangement_tags", "human_structure_tags"]:
            for tag in str(r.get(key, "")).split(";"):
                tag = tag.strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + 1
    for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {tag} | {n} |")
    lines += ["", "## Ablation", "", "| Mode | Micro-F1 | Macro-F1 | mAP | Warning |", "|---|---:|---:|---:|---|"]
    for r in ablation:
        lines.append(f"| {r.get('mode')} | {r.get('micro_f1')} | {r.get('macro_f1')} | {r.get('map')} | {r.get('warning', '')} |")
    human_summary = research / "human_eval_summary.md"
    if human_summary.exists():
        lines += ["", "## Human Eval Summary", "", human_summary.read_text(encoding="utf-8")]
    lines += ["", "## Case Study Summary", "", "Run `scripts/make_case_study.py` for a selected track and paste the generated section here."]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
