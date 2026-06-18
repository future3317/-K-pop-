from __future__ import annotations

from pathlib import Path

import numpy as np

from kpop_scope.audio.loader import load_audio
from kpop_scope.audio.features import extract_features


def _compact_stem_summary(features: dict) -> dict:
    loud = features.get("loudness", {})
    spectral = features.get("spectral", {})
    onset = features.get("onset", {})
    tempo = features.get("tempo", {})
    return {
        "duration_seconds": features.get("duration_seconds"),
        "rms_mean": loud.get("rms_stats", {}).get("mean", 0.0),
        "rms_p90": loud.get("rms_stats", {}).get("p90", 0.0),
        "onset_density_per_sec": onset.get("onset_density_per_sec", 0.0),
        "tempo_bpm": tempo.get("bpm", 0.0),
        "spectral_centroid_mean_hz": spectral.get("centroid_hz", {}).get("mean", 0.0),
        "low_energy_ratio_mean": spectral.get("band_energy_ratio", {}).get("low_20_250", {}).get("mean", 0.0),
        "high_energy_ratio_mean": spectral.get("band_energy_ratio", {}).get("high_6000_12000", {}).get("mean", 0.0),
        "loudness_curve": loud.get("rms_curve", []),
        "frame_times": loud.get("frame_times", []),
    }


def analyze_stems(
    stems: dict[str, str],
    sample_rate: int = 44100,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> dict:
    """Extract compact features for each separated stem."""
    out: dict[str, dict] = {}
    for stem_name, path in stems.items():
        if not Path(path).exists():
            out[stem_name] = {"error": f"file not found: {path}"}
            continue
        y, sr = load_audio(path, sample_rate=sample_rate, mono=True)
        feat = extract_features(y, sr, hop_length=hop_length, n_fft=n_fft, max_segments=8)
        out[stem_name] = _compact_stem_summary(feat)

    # Relative stem contribution by mean RMS.
    means = {k: v.get("rms_mean", 0.0) for k, v in out.items() if isinstance(v, dict)}
    total = float(np.sum(list(means.values())) + 1e-12)
    for stem, value in means.items():
        out[stem]["relative_rms_share"] = float(value / total)
    return out
