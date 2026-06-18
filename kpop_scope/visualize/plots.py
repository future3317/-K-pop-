from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from kpop_scope.utils import ensure_dir


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def make_plots(y: np.ndarray, sr: int, features: dict, output_dir: str | Path, stem_features: dict | None = None) -> dict:
    """Create report plots and return paths relative to report directory."""
    output_dir = ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    paths: dict[str, str] = {}

    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(12, 3))
    plt.plot(t, y, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    p = fig_dir / "waveform.png"
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    paths["waveform"] = _rel(p, output_dir)

    loud = features.get("loudness", {})
    times = np.array(loud.get("frame_times", []), dtype=float)
    rms_db = np.array(loud.get("rms_db_relative_curve", []), dtype=float)
    if times.size and rms_db.size:
        plt.figure(figsize=(12, 3))
        plt.plot(times[: len(rms_db)], rms_db)
        plt.xlabel("Time (s)")
        plt.ylabel("Relative dB")
        plt.title("Loudness curve")
        p = fig_dir / "loudness_curve.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["loudness"] = _rel(p, output_dir)

    seg = features.get("segments", {})
    novelty = np.array(seg.get("novelty_curve", []), dtype=float)
    ntimes = np.array(seg.get("frame_times", []), dtype=float)
    if novelty.size and ntimes.size:
        plt.figure(figsize=(12, 3))
        plt.plot(ntimes[: len(novelty)], novelty)
        for b in seg.get("boundaries", []):
            plt.axvline(b, linestyle="--", linewidth=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Novelty")
        plt.title("Energy/onset novelty and segment boundaries")
        p = fig_dir / "novelty_curve.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["novelty"] = _rel(p, output_dir)

    segments = seg.get("segments", [])
    if segments:
        plt.figure(figsize=(12, 2.5))
        y0 = 0
        for s in segments:
            start = s.get("start", 0.0)
            dur = s.get("duration", 0.0)
            plt.broken_barh([(start, dur)], (y0, 0.8))
            plt.text(start + dur / 2, y0 + 0.4, s.get("label_guess", ""), ha="center", va="center", fontsize=8)
        plt.ylim(0, 1)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title("Coarse segment timeline")
        p = fig_dir / "segment_timeline.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["segments"] = _rel(p, output_dir)

    if stem_features:
        plt.figure(figsize=(12, 3))
        for stem, info in stem_features.items():
            times = np.array(info.get("frame_times", []), dtype=float)
            curve = np.array(info.get("loudness_curve", []), dtype=float)
            if times.size and curve.size:
                plt.plot(times[: len(curve)], curve / (np.max(curve) + 1e-8), label=stem)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized RMS")
        plt.title("Stem energy curves")
        plt.legend()
        p = fig_dir / "stem_energy.png"
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        paths["stem_energy"] = _rel(p, output_dir)

    return paths
