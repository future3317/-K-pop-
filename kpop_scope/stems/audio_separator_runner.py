from __future__ import annotations

import shutil
from pathlib import Path

from kpop_scope.utils import ensure_dir

STEM_NAMES = ["vocals", "drums", "bass", "other"]
DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _normalized_stem_name(path: str | Path) -> str | None:
    text = Path(path).stem.lower()
    for name in STEM_NAMES:
        if name in text:
            return name
    aliases = {
        "instrumental": "other",
        "no_vocals": "other",
        "accompaniment": "other",
    }
    for alias, name in aliases.items():
        if alias in text:
            return name
    return None


def _needs_ascii_proxy(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _ascii_proxy_audio(audio_path: Path, output_dir: Path) -> Path:
    if not _needs_ascii_proxy(audio_path):
        return audio_path
    proxy_dir = ensure_dir(output_dir / "_separator_input")
    proxy_path = proxy_dir / f"track_input{audio_path.suffix.lower()}"
    if not proxy_path.exists() or proxy_path.stat().st_size != audio_path.stat().st_size:
        shutil.copy2(audio_path, proxy_path)
    return proxy_path


def separate_with_audio_separator(
    audio_path: str | Path,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    model_dir: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """Separate stems with python-audio-separator.

    The runner intentionally returns the same high-level schema as the old stem
    runner: `available`, `stems`, `warnings`, and `raw_output_dir`.
    """
    del device
    audio_path = Path(audio_path)
    output_dir = ensure_dir(output_dir)
    warnings: list[str] = []

    try:
        from audio_separator.separator import Separator
    except Exception as exc:
        return {
            "available": False,
            "backend": "audio_separator",
            "model": model,
            "stems": {},
            "warnings": [f"audio-separator is not installed or failed to import: {type(exc).__name__}: {exc}"],
            "raw_output_dir": str(output_dir),
        }

    separator_audio_path = _ascii_proxy_audio(audio_path, output_dir)
    if separator_audio_path != audio_path:
        warnings.append(f"Using ASCII proxy for audio-separator input: {separator_audio_path.name}")

    try:
        separator = Separator(
            output_dir=str(output_dir),
            output_format="WAV",
            model_file_dir=str(model_dir) if model_dir else str(output_dir / "_models"),
            use_soundfile=True,
        )
        separator.load_model(model)
        output_files = separator.separate(str(separator_audio_path))
    except Exception as exc:
        return {
            "available": False,
            "backend": "audio_separator",
            "model": model,
            "stems": {},
            "warnings": [f"audio-separator failed: {type(exc).__name__}: {exc}", *warnings],
            "raw_output_dir": str(output_dir),
        }

    stems: dict[str, str] = {}
    for file in output_files or []:
        stem = _normalized_stem_name(file)
        if stem:
            stems[stem] = str(file)

    if "vocals" in stems and len(stems) == 2 and "other" not in stems:
        for file in output_files:
            if str(file) != stems["vocals"]:
                stems["other"] = str(file)
                break

    missing = [name for name in STEM_NAMES if name not in stems]
    if missing:
        warnings.append(
            "Model did not produce all four canonical stems: "
            + ", ".join(missing)
            + ". Stem contribution will use the available stems only."
        )

    return {
        "available": bool(stems),
        "backend": "audio_separator",
        "model": model,
        "stems": stems,
        "warnings": warnings,
        "raw_output_dir": str(output_dir),
        "output_files": [str(p) for p in output_files or []],
    }
