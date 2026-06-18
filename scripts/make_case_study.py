#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Markdown case study from one track analysis.")
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    analysis_path = Path(args.analysis)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    features = analysis.get("features", {})
    tags = analysis.get("tag_result", {}).get("tags", [])
    segs = features.get("segments", {}).get("segments", [])
    title = analysis_path.parent.name
    lines = [f"# Case Study: {title}", "", "## 基础特征", ""]
    lines.append(f"- BPM: {float(features.get('tempo', {}).get('bpm', 0.0) or 0.0):.1f}")
    lines.append(f"- Key: {features.get('key', {}).get('key', 'unknown')}")
    lines.append(f"- Duration: {float(features.get('duration_seconds', 0.0) or 0.0):.1f}s")
    lines += ["", "## Top Tags", ""]
    for item in tags[:10]:
        lines.append(f"- {item.get('tag')}: {float(item.get('score', 0.0) or 0.0):.3f}")
    lines += ["", "## 段落时间线", "", "| Time | Label | Confidence | Evidence |", "|---|---|---:|---|"]
    for s in segs:
        lines.append(f"| {s.get('start', 0):.1f}-{s.get('end', 0):.1f}s | {s.get('label', s.get('label_guess'))} | {float(s.get('label_confidence', 0) or 0):.2f} | {'; '.join(s.get('evidence', s.get('label_evidence', []))[:2])} |")
    lines += ["", "## 为什么 chorus/drop 被识别", ""]
    chorus = [s for s in segs if "chorus" in str(s.get("label", s.get("label_guess", "")))]
    lines.append("检测到 chorus/drop 候选。" if chorus else "未检测到高置信 chorus/drop，需人工复核。")
    for s in chorus[:3]:
        lines.append(f"- {s.get('start', 0):.1f}-{s.get('end', 0):.1f}s: energy={float(s.get('energy_mean', 0) or 0):.2f}, onset={float(s.get('onset_density', 0) or 0):.2f}")
    lines += ["", "## 为什么 dance break 被识别或未识别", ""]
    dance = [s for s in segs if "dance break" in str(s.get("label", s.get("label_guess", "")))]
    lines.append("检测到 dance break 候选。" if dance else "未检测到明确 dance break；可能是人声持续主导，或没有 stems 时证据不足。")
    lines += ["", "## Stem Contribution", ""]
    contrib = analysis.get("stem_contribution", {})
    if contrib:
        for item in contrib.get("tag_contributions", [])[:6]:
            lines.append(f"- {item.get('tag')}: {'; '.join(item.get('evidence', [])[:2])}")
    else:
        lines.append("未生成 stem contribution。")
    lines += ["", "## 不确定性说明", "", "该 case study 基于自动分段、规则证据和可选模型输出。小样本 TWICE demo 只适合作为系统链路和定性案例，不证明泛化能力。"]
    out = Path(args.output) if args.output else Path("outputs/research") / f"case_study_{title}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
