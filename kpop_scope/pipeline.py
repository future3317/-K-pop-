from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio.features import extract_features
from .audio.loader import load_audio
from .config import load_config
from .explain.report_generator import generate_report
from .explain.rules import build_explanations
from .explain.stem_contribution import build_stem_contribution
from .models.tagger import predict_kpop_tags
from .stems.audio_separator_runner import separate_with_audio_separator
from .stems.stem_features import analyze_stems
from .utils import ensure_dir, write_json, write_text
from .visualize.plots import make_plots as create_plots


def analyze(
    audio_path: str | Path,
    output_dir: str | Path = "outputs/kpop_scope_analysis",
    config_path: str | Path | None = None,
    use_stems: bool | None = None,
    make_plots: bool | None = None,
    max_duration: float | None = None,
    use_mert: bool | None = None,
    classifier_path: str | Path | None = None,
    fallback_to_acoustic_prior: bool | None = None,
    report_mode: str | None = None,
    report_detail: str | None = None,
    stem_model: str | None = None,
) -> dict[str, Any]:
    """Run the full KPopScope analysis pipeline.

    Parameters
    ----------
    audio_path:
        Local audio path. Copyrighted audio is never uploaded or bundled.
    output_dir:
        Where analysis outputs are written.
    config_path:
        Optional YAML config overriding configs/default.yaml.
    use_stems:
        Override config.stems.enabled.
    make_plots:
        Override config.output.plots.
    max_duration:
        Optional analysis duration cap in seconds. Useful for quick debugging.
    use_mert:
        Override config.models.use_mert.
    classifier_path:
        Optional MERT K-pop classifier checkpoint path.
    fallback_to_acoustic_prior:
        Whether to fall back to the transparent acoustic-prior classifier when no checkpoint is configured.
    """
    config = load_config(config_path)
    if use_stems is not None:
        config["stems"]["enabled"] = bool(use_stems)
    if make_plots is not None:
        config["output"]["plots"] = bool(make_plots)
    if max_duration is not None:
        config["audio"]["max_duration"] = float(max_duration)
    if use_mert is not None:
        config.setdefault("models", {})["use_mert"] = bool(use_mert)
    if classifier_path is not None:
        config.setdefault("models", {})["kpop_classifier_path"] = str(classifier_path)
    if fallback_to_acoustic_prior is not None:
        config.setdefault("models", {})["fallback_to_acoustic_prior"] = bool(fallback_to_acoustic_prior)
    if report_mode is not None:
        config.setdefault("report", {})["mode"] = str(report_mode)
    if report_detail is not None:
        config.setdefault("report", {})["detail"] = str(report_detail)
    if stem_model is not None:
        config.setdefault("stems", {})["model"] = str(stem_model)

    audio_path = Path(audio_path)
    output_dir = ensure_dir(output_dir)

    y, sr = load_audio(
        audio_path,
        sample_rate=int(config["audio"].get("sample_rate", 44100)),
        mono=bool(config["audio"].get("mono", True)),
        offset=float(config["audio"].get("offset", 0.0)),
        duration=config["audio"].get("max_duration"),
    )
    feat_cfg = config.get("features", {})
    features = extract_features(
        y,
        sr,
        hop_length=int(feat_cfg.get("hop_length", 512)),
        n_fft=int(feat_cfg.get("n_fft", 2048)),
        segment_min_seconds=float(feat_cfg.get("segment_min_seconds", 8.0)),
        segment_max_seconds=float(feat_cfg.get("segment_max_seconds", 32.0)),
        novelty_smooth_seconds=float(feat_cfg.get("novelty_smooth_seconds", 1.0)),
        max_segments=int(feat_cfg.get("max_segments", 12)),
    )

    # v2: MERT embedding + K-pop classifier when a checkpoint is configured.
    # If no checkpoint is available, a transparent acoustic-prior classifier keeps the package runnable.
    tag_result = predict_kpop_tags(audio_path, y, sr, features, config.get("models", {}))

    stem_result = None
    stem_features = None
    if bool(config.get("stems", {}).get("enabled", False)):
        stem_output_dir = output_dir / "stems"
        stem_result = separate_with_audio_separator(
            audio_path,
            stem_output_dir,
            model=str(config["stems"].get("model", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")),
            model_dir=config["stems"].get("model_dir"),
            device=str(config["stems"].get("device", "auto")),
        )
        if stem_result.get("available"):
            stem_features = analyze_stems(
                stem_result["stems"],
                sample_rate=int(config["audio"].get("sample_rate", 44100)),
                hop_length=int(feat_cfg.get("hop_length", 512)),
                n_fft=int(feat_cfg.get("n_fft", 2048)),
            )

    explanations = build_explanations(features, tag_result, stem_features=stem_features)
    contribution = build_stem_contribution(
        tag_result,
        stem_features=stem_features,
        segment_features=features.get("segments", {}),
    )

    figure_paths = {}
    if bool(config.get("output", {}).get("plots", True)):
        figure_paths = create_plots(y, sr, features, output_dir=output_dir, stem_features=stem_features)

    report = generate_report(
        audio_path=audio_path,
        features=features,
        tag_result=tag_result,
        explanations=explanations,
        stem_features=stem_features,
        figure_paths=figure_paths,
        title_prefix=str(config.get("report", {}).get("title_prefix", "KPopScope Analysis")),
        report_mode=str(config.get("report", {}).get("mode", "evidence_grounded")),
        stem_contribution=contribution,
        report_detail=str(config.get("report", {}).get("detail", "readable")),
    )

    result = {
        "audio_path": str(audio_path),
        "config": config,
        "features": features,
        "tag_result": tag_result,
        "stem_result": stem_result,
        "stem_features": stem_features,
        "stem_contribution": contribution,
        "explanations": explanations,
        "figures": figure_paths,
        "report_markdown": report,
    }

    if bool(config.get("output", {}).get("save_json", True)):
        write_json({k: v for k, v in result.items() if k != "report_markdown"}, output_dir / "analysis.json")
    if bool(config.get("output", {}).get("save_markdown", True)):
        write_text(report, output_dir / "report.md")
    return result
