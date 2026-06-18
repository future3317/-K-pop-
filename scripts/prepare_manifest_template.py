#!/usr/bin/env python
"""Create a small CSV manifest template for your own legal/local audio files."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--output", default="data/manifest.csv")
    args = ap.parse_args()
    audio_dir = Path(args.audio_dir)
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    rows = []
    for p in sorted(audio_dir.rglob("*")):
        if p.suffix.lower() in exts:
            rows.append({"path": str(p), "tags": "dance-pop;energetic"})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}; edit the tags column before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
