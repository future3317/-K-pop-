#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kpop_scope.taxonomy import labels_by_group


def safe_name_from_rel(rel_path: str) -> str:
    stem = Path(rel_path).stem.lower()
    text = re.sub(r"[^a-z0-9]+", "_", stem)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "track"


def level(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def split_tags(analysis: dict) -> tuple[list[str], list[str], list[str]]:
    groups = labels_by_group()
    style = set(groups.get("Style", []))
    mood = set(groups.get("Mood", []))
    arr = set(groups.get("Arrangement", []))
    # Backward-compatible aliases from acoustic prior.
    aliases = {
        "retro synth": "retro pop",
        "low-frequency drive": "sub-bass drive",
        "pre-chorus build-up": "pre-chorus buildup",
        "dance break likely": "dance break",
        "rap section likely": "rap section",
    }
    out = {"style": [], "mood": [], "arr": []}
    for item in analysis.get("tag_result", {}).get("tags", []):
        tag = str(item.get("tag"))
        tag = aliases.get(tag, tag)
        if tag in style:
            out["style"].append(tag)
        if tag in mood:
            out["mood"].append(tag)
        if tag in arr:
            out["arr"].append(tag)
    return out["style"], out["mood"], out["arr"]


def pseudo_row(manifest_row: dict, analysis: dict) -> dict:
    features = analysis.get("features", {})
    seg = features.get("segments", {})
    segs = seg.get("segments", []) or []
    ss = seg.get("structure_summary", {})
    style, mood, arr = split_tags(analysis)
    bpm = float(features.get("tempo", {}).get("bpm", 0.0) or 0.0)
    onset = float(features.get("onset", {}).get("onset_density_per_sec", 0.0) or 0.0)
    rms = float(features.get("loudness", {}).get("rms_stats", {}).get("p90", 0.0) or 0.0)
    centroid = float(features.get("spectral", {}).get("centroid_hz", {}).get("mean", 0.0) or 0.0)
    chorus_lift = max([float(s.get("energy_delta", 0.0) or 0.0) for s in segs if "chorus" in str(s.get("label", ""))] or [0.0])
    has_pre = bool(ss.get("has_prechorus", False))
    has_dance = bool(ss.get("has_dance_break", False))
    has_vocal_layering = "vocal layering" in arr
    has_rap = "rap section" in arr
    conf = min(0.95, 0.30 + 0.08 * len(style + mood + arr) + 0.20 * float(ss.get("structure_confidence", 0.0) or 0.0))
    needs_review = conf < 0.68 or analysis.get("tag_result", {}).get("source", "").startswith("acoustic_prior")
    structure_tags = []
    if ss.get("has_intro"):
        structure_tags.append("clear intro")
    if ss.get("has_prechorus") and ss.get("has_chorus_drop"):
        structure_tags.append("verse-prechorus-chorus structure")
    if ss.get("has_dance_break"):
        structure_tags.append("dance break section")
    if any("outro" in str(s.get("label", "")) for s in segs):
        structure_tags.append("outro/fade")
    evidence = [
        f"BPM={bpm:.1f}",
        f"onset_density={onset:.2f}/s",
        f"chorus_energy_delta={chorus_lift:.2f}",
        f"structure_confidence={float(ss.get('structure_confidence', 0.0) or 0.0):.2f}",
    ]
    return {
        "track_id": manifest_row.get("track_id", ""),
        "rel_path": manifest_row.get("rel_path", ""),
        "source": "pseudo",
        "candidate_style_tags": ";".join(dict.fromkeys(style)),
        "candidate_mood_tags": ";".join(dict.fromkeys(mood)),
        "candidate_arrangement_tags": ";".join(dict.fromkeys(arr)),
        "candidate_structure_tags": ";".join(dict.fromkeys(structure_tags)),
        "bpm": f"{bpm:.2f}",
        "key": features.get("key", {}).get("key", "unknown"),
        "energy_level": level(rms, 0.04, 0.10),
        "brightness_level": level(centroid, 2200, 3500),
        "danceability_level": level(onset, 1.8, 3.0),
        "chorus_energy_lift": f"{chorus_lift:.3f}",
        "has_dance_break": str(has_dance).lower(),
        "has_rap_like_section": str(has_rap).lower(),
        "has_prechorus_buildup": str(has_pre).lower(),
        "has_vocal_layering": str(has_vocal_layering).lower(),
        "pseudo_label_confidence": f"{conf:.3f}",
        "needs_human_review": str(needs_review).lower(),
        "evidence": " | ".join(evidence),
        "human_style_tags": "",
        "human_mood_tags": "",
        "human_arrangement_tags": "",
        "human_structure_tags": "",
        "human_notes": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap weak pseudo labels from KPopScope analysis JSON files.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    rows = []
    for row in manifest:
        track_dir = Path(args.analysis_dir) / safe_name_from_rel(row.get("rel_path", row.get("filename", "")))
        analysis_path = track_dir / "analysis.json"
        if not analysis_path.exists():
            rows.append({"track_id": row.get("track_id", ""), "rel_path": row.get("rel_path", ""), "source": "pseudo", "needs_human_review": "true", "evidence": "missing analysis.json"})
            continue
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        rows.append(pseudo_row(row, analysis))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "track_id", "rel_path", "source", "candidate_style_tags", "candidate_mood_tags",
        "candidate_arrangement_tags", "candidate_structure_tags", "bpm", "key",
        "energy_level", "brightness_level", "danceability_level", "chorus_energy_lift",
        "has_dance_break", "has_rap_like_section", "has_prechorus_buildup",
        "has_vocal_layering", "pseudo_label_confidence", "needs_human_review",
        "evidence", "human_style_tags", "human_mood_tags", "human_arrangement_tags",
        "human_structure_tags", "human_notes",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"wrote {out} with {len(rows)} pseudo-label rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
