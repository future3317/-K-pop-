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

from kpop_scope.pipeline import analyze


AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}
STYLE = {"k-pop dance", "dance-pop", "electropop", "synth-pop", "hip-hop pop", "r&b pop", "bubblegum pop", "ballad", "retro pop", "retro synth", "tropical/summer pop", "trap-pop", "jersey club influence"}
MOOD = {"bright", "cute", "energetic", "confident", "dreamy", "sentimental", "dark", "playful", "dramatic", "euphoric"}
ARR = {"heavy drums", "four-on-the-floor", "syncopated groove", "synth bass", "sub-bass drive", "low-frequency drive", "vocal layering", "chant hook", "rap section likely", "rap section", "pre-chorus build-up", "pre-chorus buildup", "chorus energy lift", "drop chorus", "instrumental post-chorus", "dance break", "dance break likely", "bridge contrast", "acoustic/piano intro"}


def safe_name(path: Path) -> str:
    text = path.stem.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "track"


def audio_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _top_by_group(tags: list[dict], group: set[str], limit: int = 4) -> str:
    found = [str(t.get("tag")) for t in tags if str(t.get("tag")) in group]
    return ";".join(found[:limit])


def summarize_result(src: Path, out_dir: Path, result: dict) -> dict:
    features = result.get("features", {})
    tags = result.get("tag_result", {}).get("tags", [])
    seg_summary = features.get("segments", {}).get("structure_summary", {})
    rel_report = (out_dir / "report.md").relative_to(out_dir.parent).as_posix()
    return {
        "file": src.name,
        "track_dir": out_dir.name,
        "duration": features.get("duration_seconds", 0.0),
        "bpm": features.get("tempo", {}).get("bpm", 0.0),
        "key": features.get("key", {}).get("key", "unknown"),
        "top_style_tags": _top_by_group(tags, STYLE),
        "top_mood_tags": _top_by_group(tags, MOOD),
        "top_arrangement_tags": _top_by_group(tags, ARR),
        "chorus_drop_count": len(seg_summary.get("chorus_candidates", []) or []),
        "has_dance_break": bool(seg_summary.get("has_dance_break", False)),
        "report": rel_report,
        "tag_source": result.get("tag_result", {}).get("source", "unknown"),
    }


def write_index(rows: list[dict], output_dir: Path) -> None:
    lines = ["# KPopScope Batch Index", "", "| File | Duration | BPM | Key | Style | Mood | Arrangement | Chorus/drop | Dance break | Report |", "|---|---:|---:|---|---|---|---|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['file']} | {float(r['duration']):.1f}s | {float(r['bpm']):.1f} | {r['key']} | "
            f"{r['top_style_tags']} | {r['top_mood_tags']} | {r['top_arrangement_tags']} | "
            f"{r['chorus_drop_count']} | {r['has_dance_break']} | [{Path(r['report']).name}]({r['track_dir']}/report.md) |"
        )
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def run_batch(args: argparse.Namespace) -> list[dict]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = audio_files(input_dir)
    if args.limit:
        files = files[: args.limit]

    rows: list[dict] = []
    errors_path = output_dir / "errors.jsonl"
    used: dict[str, int] = {}
    for src in files:
        name = safe_name(src)
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            name = f"{name}_{used[name]}"
        track_dir = output_dir / name
        if args.skip_existing and (track_dir / "analysis.json").exists():
            try:
                result = json.loads((track_dir / "analysis.json").read_text(encoding="utf-8"))
                rows.append(summarize_result(src, track_dir, result))
            except Exception:
                pass
            continue
        try:
            result = analyze(
                src,
                output_dir=track_dir,
                use_stems=args.stems,
                make_plots=args.plots,
                use_mert=args.mert,
                classifier_path=args.classifier,
                report_mode="evidence_grounded",
            )
            if args.both_report_modes:
                tag_only = analyze(
                    src,
                    output_dir=track_dir,
                    use_stems=args.stems,
                    make_plots=args.plots,
                    use_mert=args.mert,
                    classifier_path=args.classifier,
                    report_mode="tag_only",
                )
                (track_dir / "report_tag_only.md").write_text(tag_only["report_markdown"], encoding="utf-8")
                (track_dir / "report_evidence_grounded.md").write_text(result["report_markdown"], encoding="utf-8")
            rows.append(summarize_result(src, track_dir, result))
            print(f"ok: {src.name} -> {track_dir}")
        except Exception as exc:
            with open(errors_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": src.name, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
            print(f"failed: {src.name}: {exc}")

    if rows:
        with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(rows, output_dir)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch analyze local audio files without copying audio.")
    ap.add_argument("--input-dir", default="twice")
    ap.add_argument("--output-dir", default="outputs/twice")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--report", choices=["md"], default="md")
    ap.add_argument("--mert", action="store_true")
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="Reserved for MERT/Demucs configs; default cuda.")
    ap.add_argument("--num-workers", type=int, default=1, help="Currently processed sequentially for GPU safety.")
    ap.add_argument("--both-report-modes", action="store_true")
    args = ap.parse_args()
    rows = run_batch(args)
    print(f"finished {len(rows)} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
