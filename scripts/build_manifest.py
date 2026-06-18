#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf


AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


def track_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    st = path.stat()
    raw = f"{rel}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def guess_artist_title(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    stem = re.sub(r"\s+", " ", stem).strip()
    for sep in [" - ", "-", "_"]:
        if sep in stem:
            left, right = stem.split(sep, 1)
            return left.strip() or "unknown", right.strip() or stem
    return "unknown", stem


def audio_info(path: Path) -> tuple[float, int, int, str, str]:
    try:
        info = sf.info(str(path))
        duration = float(info.frames / max(info.samplerate, 1))
        return duration, int(info.samplerate), int(info.channels), str(info.format), ""
    except Exception as exc:
        return 0.0, 0, 0, path.suffix.lower().lstrip("."), f"metadata_error: {type(exc).__name__}: {exc}"


def build_manifest(input_dir: str | Path, output: str | Path, split: str = "demo", recursive: bool = True) -> list[dict[str, str]]:
    root = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    rows: list[dict[str, str]] = []
    for path in files:
        artist, title = guess_artist_title(path.name)
        duration, sr, channels, fmt, note = audio_info(path)
        rows.append(
            {
                "track_id": track_id_for(path, root),
                "rel_path": path.relative_to(root).as_posix(),
                "filename": path.name,
                "artist_guess": artist,
                "title_guess": title,
                "duration": f"{duration:.3f}",
                "sample_rate": str(sr),
                "channels": str(channels),
                "format": fmt,
                "split": split,
                "notes": note,
            }
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "track_id", "rel_path", "filename", "artist_guess", "title_guess", "duration",
            "sample_rate", "channels", "format", "split", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a local audio manifest without hashing audio content.")
    ap.add_argument("--input-dir", default="twice")
    ap.add_argument("--output", default="data/local/twice_manifest.csv")
    ap.add_argument("--split", default="demo")
    ap.add_argument("--no-recursive", action="store_true")
    args = ap.parse_args()
    rows = build_manifest(args.input_dir, args.output, split=args.split, recursive=not args.no_recursive)
    print(f"wrote {args.output} with {len(rows)} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
