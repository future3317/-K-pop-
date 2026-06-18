from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    path: str | Path,
    sample_rate: int = 44100,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 numpy array.

    librosa uses audioread/ffmpeg backends for mp3/m4a when available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(
        path,
        sr=sample_rate,
        mono=mono,
        offset=offset,
        duration=duration,
    )
    if y.size == 0:
        raise ValueError(f"Loaded empty audio from {path}")
    return np.asarray(y, dtype=np.float32), int(sr)
