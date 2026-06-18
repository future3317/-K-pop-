from __future__ import annotations

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def estimate_key(chroma_mean: np.ndarray) -> dict:
    """Estimate global key from a 12-D chroma vector.

    This is intentionally simple and explainable. It is good enough for a report
    cue, not for musicological ground-truth claims.
    """
    chroma_mean = np.asarray(chroma_mean, dtype=float).reshape(-1)
    if chroma_mean.shape[0] != 12:
        raise ValueError("chroma_mean must have 12 elements")
    if np.sum(chroma_mean) <= 1e-8:
        return {"key": "unknown", "tonic": None, "mode": None, "confidence": 0.0}

    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-8)
    scores: list[tuple[str, str, int, float]] = []
    for shift in range(12):
        tonic = PITCH_CLASSES[shift]
        major_score = _corr(chroma_norm, np.roll(MAJOR_PROFILE, shift))
        minor_score = _corr(chroma_norm, np.roll(MINOR_PROFILE, shift))
        scores.append((f"{tonic} major", "major", shift, major_score))
        scores.append((f"{tonic} minor", "minor", shift, minor_score))
    scores.sort(key=lambda item: item[3], reverse=True)
    best = scores[0]
    second = scores[1]
    confidence = max(0.0, min(1.0, (best[3] - second[3] + 0.15) / 0.45))
    return {
        "key": best[0],
        "tonic": PITCH_CLASSES[best[2]],
        "mode": best[1],
        "confidence": float(confidence),
        "score": float(best[3]),
    }
