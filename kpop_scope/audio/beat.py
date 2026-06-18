from __future__ import annotations

import librosa
import numpy as np


def estimate_beats(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """Estimate tempo and beat positions."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )
    if hasattr(tempo, "item"):
        tempo = tempo.item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    beat_density = len(beat_times) / max(duration, 1e-8)
    return {
        "tempo_bpm": float(tempo),
        "beat_times": beat_times.astype(float).tolist(),
        "beat_count": int(len(beat_times)),
        "beat_density_per_sec": float(beat_density),
        "onset_envelope": onset_env.astype(float).tolist(),
    }
