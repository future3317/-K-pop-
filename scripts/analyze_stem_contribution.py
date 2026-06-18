#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kpop_scope.explain.stem_contribution import build_stem_contribution


def to_markdown(data: dict) -> str:
    lines = ["# Stem Contribution", "", f"Mode: `{data.get('mode', 'unknown')}`", "", "| Tag | Full | Vocals | Drums | Bass | Other | Rhythm | Structure | Evidence |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in data.get("tag_contributions", []):
        c = item.get("contribution", {})
        lines.append(
            f"| {item.get('tag')} | {c.get('full_mix', 0):.3f} | {c.get('vocals', 0):.3f} | "
            f"{c.get('drums', 0):.3f} | {c.get('bass', 0):.3f} | {c.get('other', 0):.3f} | "
            f"{c.get('rhythm', 0):.3f} | {c.get('structure', 0):.3f} | {'; '.join(item.get('evidence', []))} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create stem contribution JSON/Markdown from analysis.json.")
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    data = build_stem_contribution(
        analysis.get("tag_result", {}),
        stem_features=analysis.get("stem_features"),
        segment_features=analysis.get("features", {}).get("segments", {}),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md = Path(args.markdown) if args.markdown else out.with_suffix(".md")
    md.write_text(to_markdown(data), encoding="utf-8")
    print(f"wrote {out} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
