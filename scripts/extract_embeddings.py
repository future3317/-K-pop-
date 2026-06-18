#!/usr/bin/env python
"""Extract MERT embeddings and optional acoustic features for a CSV manifest.

Manifest format:
    path,tags
    /abs/or/relative/song1.mp3,"dance-pop;bright;heavy drums"

Output NPZ:
    X: MERT embeddings [N, D]
    A: acoustic feature vectors [N, F]
    acoustic_feature_names: feature names [F]
    paths: source paths
    tags_text: semicolon-separated label text
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from kpop_scope.audio.features import extract_features
from kpop_scope.audio.loader import load_audio
from kpop_scope.config import load_config
from kpop_scope.models.feature_vector import acoustic_feature_vector
from kpop_scope.models.mert_embedder import MERTEmbedder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default=None, help="Legacy NPZ output path")
    ap.add_argument("--output-dir", default=None, help="Directory for twice_mert.npy/twice_acoustic.npy/twice_ids.json")
    ap.add_argument("--model", default="m-a-p/MERT-v1-95M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-duration", type=float, default=45.0)
    ap.add_argument("--chunk-seconds", type=float, default=20.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-mert", action="store_true", help="Only extract acoustic features; useful for smoke tests")
    ap.add_argument("--mert", action="store_true", help="Extract MERT embeddings; default when --skip-mert is not set")
    ap.add_argument("--resume", action="store_true", help="Reuse existing per-track cache files in output-dir/cache")
    args = ap.parse_args()
    if not args.output and not args.output_dir:
        ap.error("Provide --output legacy NPZ or --output-dir.")

    df = pd.read_csv(args.manifest)
    cfg = load_config(args.config)
    audio_cfg = cfg.get("audio", {})
    feat_cfg = cfg.get("features", {})

    embedder = None if args.skip_mert else MERTEmbedder(args.model, device=args.device)
    X = []
    A = []
    paths = []
    tags_text = []
    feature_names = None
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if "path" in row:
            path = Path(row["path"])
        else:
            path = Path(args.manifest).parent.parent.parent / str(row.get("rel_path", ""))
            fallback = Path("twice") / str(row.get("rel_path", ""))
            if fallback.exists():
                path = fallback
        track_id = str(row.get("track_id", path.stem))
        cache_dir = Path(args.output_dir) / "cache" if args.output_dir else None
        cache_npz = cache_dir / f"{track_id}.npz" if cache_dir else None
        if args.resume and cache_npz and cache_npz.exists():
            cached = np.load(cache_npz, allow_pickle=True)
            A.append(cached["A"].astype(np.float32))
            X.append(cached["X"].astype(np.float32))
            paths.append(str(path))
            tags_text.append(str(row.get("tags", "")))
            feature_names = cached["acoustic_feature_names"].tolist()
            continue
        y, sr = load_audio(
            path,
            sample_rate=int(audio_cfg.get("sample_rate", 44100)),
            mono=True,
            duration=args.max_duration,
        )
        features = extract_features(
            y,
            sr,
            hop_length=int(feat_cfg.get("hop_length", 512)),
            n_fft=int(feat_cfg.get("n_fft", 2048)),
            segment_min_seconds=float(feat_cfg.get("segment_min_seconds", 8.0)),
            segment_max_seconds=float(feat_cfg.get("segment_max_seconds", 32.0)),
            novelty_smooth_seconds=float(feat_cfg.get("novelty_smooth_seconds", 1.0)),
            max_segments=int(feat_cfg.get("max_segments", 12)),
        )
        acoustic, names = acoustic_feature_vector(features)
        feature_names = names
        A.append(acoustic)
        if embedder is None:
            X.append(np.zeros(0, dtype=np.float32))
        else:
            emb = embedder.embed_audio(y, sr=sr, chunk_seconds=args.chunk_seconds)
            X.append(emb.astype(np.float32))
        paths.append(str(path))
        tags_text.append(str(row.get("tags", "")))
        if cache_npz:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_npz, X=X[-1], A=A[-1], acoustic_feature_names=np.array(feature_names or [], dtype=object))

    X_arr = np.stack(X) if X and X[0].size else np.zeros((len(A), 0), dtype=np.float32)
    A_arr = np.stack(A).astype(np.float32)
    if args.output:
        np.savez_compressed(
            args.output,
            X=X_arr,
            A=A_arr,
            acoustic_feature_names=np.array(feature_names or [], dtype=object),
            paths=np.array(paths, dtype=object),
            tags_text=np.array(tags_text, dtype=object),
        )
        print(f"saved {args.output}; X={X_arr.shape}, A={A_arr.shape}")
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "twice_mert.npy", X_arr)
        np.save(out_dir / "twice_acoustic.npy", A_arr)
        (out_dir / "twice_ids.json").write_text(
            json.dumps(
                {
                    "track_ids": [str(row.get("track_id", "")) for _, row in df.iterrows()],
                    "paths": paths,
                    "tags_text": tags_text,
                    "acoustic_feature_names": feature_names or [],
                    "model": args.model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {out_dir}; MERT={X_arr.shape}, acoustic={A_arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
