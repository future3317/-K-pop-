from __future__ import annotations

from typing import Any

import numpy as np


ACOUSTIC_FEATURE_NAMES: list[str] = [
    "tempo_bpm_norm",
    "beat_density_norm",
    "onset_density_norm",
    "rms_mean",
    "rms_std",
    "rms_p90",
    "rms_dynamic_range_db_norm",
    "centroid_mean_norm",
    "centroid_std_norm",
    "bandwidth_mean_norm",
    "rolloff_mean_norm",
    "zcr_mean_norm",
    "low_20_250_mean",
    "lowmid_250_500_mean",
    "mid_500_2000_mean",
    "highmid_2000_6000_mean",
    "high_6000_12000_mean",
    "segment_count_norm",
    "max_segment_energy",
    "chorus_lift_proxy",
    "energy_contrast",
    "prechorus_build_proxy",
    "dancebreak_proxy",
]


def _get(d: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = d
    try:
        for key in path:
            cur = cur[key]
        if cur is None:
            return default
        if hasattr(cur, "item"):
            cur = cur.item()
        return float(cur)
    except Exception:
        return default


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _norm_bpm(bpm: float) -> float:
    # Maps roughly 60-180 BPM to 0-1 while keeping half-time K-pop usable.
    if bpm <= 0:
        return 0.0
    while bpm < 70:
        bpm *= 2.0
    while bpm > 190:
        bpm /= 2.0
    return _clip01((bpm - 60.0) / 140.0)


def _segments(features: dict[str, Any]) -> list[dict[str, Any]]:
    return list(features.get("segments", {}).get("segments", []) or [])


def acoustic_feature_vector(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Create a compact numeric feature vector for K-pop tag classification.

    These features intentionally stay simple and explainable. They let a trained
    K-pop classifier fuse MERT embeddings with rhythm/energy cues, and they also
    provide a deterministic fallback when no learned checkpoint is configured.
    """
    segs = _segments(features)
    energies = np.array([float(s.get("mean_energy", 0.0) or 0.0) for s in segs], dtype=float)
    onsets = np.array([float(s.get("mean_onset", 0.0) or 0.0) for s in segs], dtype=float)

    bpm = _get(features, ["tempo", "bpm"])
    beat_density = _get(features, ["tempo", "beat_density_per_sec"])
    onset_density = _get(features, ["onset", "onset_density_per_sec"])
    rms_mean = _get(features, ["loudness", "rms_stats", "mean"])
    rms_std = _get(features, ["loudness", "rms_stats", "std"])
    rms_p90 = _get(features, ["loudness", "rms_stats", "p90"])
    db_p90 = _get(features, ["loudness", "rms_db_relative_stats", "p90"])
    db_p10 = _get(features, ["loudness", "rms_db_relative_stats", "p10"])
    centroid_mean = _get(features, ["spectral", "centroid_hz", "mean"])
    centroid_std = _get(features, ["spectral", "centroid_hz", "std"])
    bandwidth_mean = _get(features, ["spectral", "bandwidth_hz", "mean"])
    rolloff_mean = _get(features, ["spectral", "rolloff_hz", "mean"])
    zcr_mean = _get(features, ["spectral", "zero_crossing_rate", "mean"])

    band = features.get("spectral", {}).get("band_energy_ratio", {})
    low = float(band.get("low_20_250", {}).get("mean", 0.0) or 0.0)
    lowmid = float(band.get("lowmid_250_500", {}).get("mean", 0.0) or 0.0)
    mid = float(band.get("mid_500_2000", {}).get("mean", 0.0) or 0.0)
    highmid = float(band.get("highmid_2000_6000", {}).get("mean", 0.0) or 0.0)
    high = float(band.get("high_6000_12000", {}).get("mean", 0.0) or 0.0)

    if energies.size:
        global_mean = float(np.mean(energies))
        max_energy = float(np.max(energies))
        energy_contrast = float(np.max(energies) - np.min(energies))
        chorus_lift_proxy = _clip01((max_energy - global_mean + 0.15) / 0.65)
    else:
        max_energy = energy_contrast = chorus_lift_proxy = 0.0

    pre_build = 0.0
    dancebreak = 0.0
    for i, seg in enumerate(segs):
        label = str(seg.get("label_guess", ""))
        pre_build = max(pre_build, 1.0 if "pre-chorus" in label else 0.0)
        dancebreak = max(dancebreak, 1.0 if "dance break" in label else 0.0)
        if i + 1 < len(segs):
            cur_e = float(seg.get("mean_energy", 0.0) or 0.0)
            next_e = float(segs[i + 1].get("mean_energy", 0.0) or 0.0)
            cur_o = float(seg.get("mean_onset", 0.0) or 0.0)
            pre_build = max(pre_build, _clip01((next_e - cur_e + 0.1) / 0.5) * _clip01((cur_o + 0.1) / 0.8))
    if onsets.size and energies.size:
        dense = _clip01((float(np.max(onsets)) - float(np.mean(onsets)) + 0.10) / 0.55)
        loud = _clip01((float(np.max(energies)) - 0.45) / 0.45)
        dancebreak = max(dancebreak, dense * loud)

    values = [
        _norm_bpm(bpm),
        _clip01(beat_density / 3.5),
        _clip01(onset_density / 5.0),
        _clip01(rms_mean * 8.0),
        _clip01(rms_std * 12.0),
        _clip01(rms_p90 * 8.0),
        _clip01((db_p90 - db_p10) / 45.0),
        _clip01(centroid_mean / 6000.0),
        _clip01(centroid_std / 3500.0),
        _clip01(bandwidth_mean / 7000.0),
        _clip01(rolloff_mean / 12000.0),
        _clip01(zcr_mean / 0.35),
        _clip01(low),
        _clip01(lowmid),
        _clip01(mid),
        _clip01(highmid),
        _clip01(high),
        _clip01(len(segs) / 14.0),
        _clip01(max_energy),
        _clip01(chorus_lift_proxy),
        _clip01(energy_contrast),
        _clip01(pre_build),
        _clip01(dancebreak),
    ]
    return np.asarray(values, dtype=np.float32), list(ACOUSTIC_FEATURE_NAMES)
