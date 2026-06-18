import wave
from pathlib import Path

import numpy as np

from scripts.build_manifest import build_manifest


def write_wav(path: Path) -> None:
    sr = 8000
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(y.tobytes())


def test_build_manifest(tmp_path):
    audio = tmp_path / "twice"
    audio.mkdir()
    write_wav(audio / "Artist - Title.wav")
    out = tmp_path / "manifest.csv"
    rows = build_manifest(audio, out)
    assert out.exists()
    assert rows[0]["rel_path"] == "Artist - Title.wav"
    assert rows[0]["track_id"]
