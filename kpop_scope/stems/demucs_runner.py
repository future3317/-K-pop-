from __future__ import annotations

import shutil
from pathlib import Path

from kpop_scope.utils import ensure_dir, run_command

STEM_NAMES = ["vocals", "drums", "bass", "other"]


def separate_with_demucs(
    audio_path: str | Path,
    output_dir: str | Path,
    model: str = "htdemucs",
    device: str = "auto",
) -> dict:
    """Run Demucs source separation if the demucs CLI is available.

    Returns a dict with `available`, `stems`, and `warnings`. It never raises on
    missing Demucs; the pipeline can continue with full-mix analysis.
    """
    audio_path = Path(audio_path)
    output_dir = ensure_dir(output_dir)
    warnings: list[str] = []

    if shutil.which("demucs") is None:
        return {
            "available": False,
            "stems": {},
            "warnings": ["demucs CLI not found. Install with `pip install demucs` to enable stem analysis."],
            "raw_output_dir": None,
        }

    cmd = ["demucs", "-n", model, "-o", str(output_dir), str(audio_path)]
    if device and device != "auto":
        cmd.extend(["-d", device])
    code, stdout, stderr = run_command(cmd)
    if code != 0:
        return {
            "available": False,
            "stems": {},
            "warnings": [f"demucs failed with exit code {code}", stderr[-2000:]],
            "raw_output_dir": str(output_dir),
        }

    # Demucs usually writes output_dir/model_name/track_stem/stem.wav
    track_name = audio_path.stem
    model_dir = output_dir / model / track_name
    if not model_dir.exists():
        # Fallback: find any directory containing vocals.wav.
        candidates = list(output_dir.rglob("vocals.wav"))
        if candidates:
            model_dir = candidates[0].parent
    stems = {}
    for stem in STEM_NAMES:
        path = model_dir / f"{stem}.wav"
        if path.exists():
            stems[stem] = str(path)
        else:
            warnings.append(f"Missing expected stem: {stem}")
    return {
        "available": bool(stems),
        "stems": stems,
        "warnings": warnings,
        "raw_output_dir": str(output_dir),
        "stdout_tail": stdout[-1000:],
    }
