from __future__ import annotations

import librosa
import numpy as np

from .beat import estimate_beats
from .key import estimate_key
from .segmentation import segment_track


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _band_energy(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict:
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = np.sum(S, axis=0) + 1e-10
    bands = {
        "low_20_250": (20, 250),
        "lowmid_250_500": (250, 500),
        "mid_500_2000": (500, 2000),
        "highmid_2000_6000": (2000, 6000),
        "high_6000_12000": (6000, 12000),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            ratio = np.zeros_like(total)
        else:
            ratio = np.sum(S[mask, :], axis=0) / total
        out[name] = _summary_stats(ratio)
    return out


def extract_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    segment_min_seconds: float = 8.0,
    segment_max_seconds: float = 32.0,
    novelty_smooth_seconds: float = 1.0,
    max_segments: int = 12,
) -> dict:
    """Extract explainable MIR features from a mono audio signal."""
    duration = float(len(y) / sr)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, units="time", backtrack=False
    )
    beat_info = estimate_beats(y, sr, hop_length=hop_length)
    key_info = estimate_key(chroma_mean)
    segment_info = segment_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        min_seconds=segment_min_seconds,
        max_seconds=segment_max_seconds,
        smooth_seconds=novelty_smooth_seconds,
        max_segments=max_segments,
    )
    band_energy = _band_energy(y, sr, n_fft=n_fft, hop_length=hop_length)

    frames_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return {
        "duration_seconds": duration,
        "sample_rate": int(sr),
        "hop_length": int(hop_length),
        "n_fft": int(n_fft),
        "tempo": {
            "bpm": beat_info["tempo_bpm"],
            "beat_count": beat_info["beat_count"],
            "beat_density_per_sec": beat_info["beat_density_per_sec"],
            "beat_times": beat_info["beat_times"],
        },
        "onset": {
            "onset_count": int(len(onset_times)),
            "onset_density_per_sec": float(len(onset_times) / max(duration, 1e-8)),
            "onset_times": [float(t) for t in onset_times],
            "onset_envelope_stats": _summary_stats(onset_env),
        },
        "key": key_info,
        "chroma_mean": [float(v) for v in chroma_mean],
        "loudness": {
            "rms_stats": _summary_stats(rms),
            "rms_db_relative_stats": _summary_stats(rms_db),
            "rms_curve": [float(v) for v in rms],
            "rms_db_relative_curve": [float(v) for v in rms_db],
            "frame_times": [float(v) for v in frames_time],
        },
        "spectral": {
            "centroid_hz": _summary_stats(centroid),
            "bandwidth_hz": _summary_stats(bandwidth),
            "rolloff_hz": _summary_stats(rolloff),
            "zero_crossing_rate": _summary_stats(zcr),
            "band_energy_ratio": band_energy,
        },
        "segments": segment_info,
    }
