from __future__ import annotations

from pathlib import Path

from kpop_scope.utils import human_time


def _fmt(x: float | int | None, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def generate_report(
    audio_path: str | Path,
    features: dict,
    tag_result: dict,
    explanations: dict,
    stem_features: dict | None = None,
    figure_paths: dict | None = None,
    title_prefix: str = "KPopScope Analysis",
    report_mode: str = "evidence_grounded",
    stem_contribution: dict | None = None,
) -> str:
    """Generate a Markdown report in Chinese."""
    audio_path = Path(audio_path)
    figure_paths = figure_paths or {}
    tempo = features.get("tempo", {})
    key = features.get("key", {})
    onset = features.get("onset", {})
    loudness = features.get("loudness", {})
    spectral = features.get("spectral", {})
    duration = features.get("duration_seconds", 0.0)

    lines: list[str] = []
    lines.append(f"# {title_prefix}: {audio_path.name}")
    lines.append("")
    lines.append("> 本报告由 KPopScope 自动生成。结构标签与赏析文字是基于音频特征和规则/模型的推测，不等同于人工音乐学标注。")
    if tag_result.get("source", "").startswith("acoustic_prior"):
        lines.append("> 当前标签来自声学先验和规则 fallback，不等同于训练好的 K-pop 模型预测。")
    lines.append(f"> 报告模式：`{report_mode}`。")
    lines.append("")

    lines.append("## 1. 总览")
    lines.append("")
    lines.append(f"- **时长**：{human_time(duration)}")
    lines.append(f"- **估计 BPM**：{_fmt(tempo.get('bpm'), 1)}")
    lines.append(f"- **估计调性**：{key.get('key', 'unknown')}，置信度约 {_fmt(key.get('confidence'), 2)}")
    lines.append(f"- **起音密度**：{_fmt(onset.get('onset_density_per_sec'), 2)} 次/秒")
    lines.append(f"- **平均频谱质心**：{_fmt(spectral.get('centroid_hz', {}).get('mean'), 1)} Hz")
    lines.append("")

    if figure_paths.get("waveform"):
        lines.append(f"![Waveform]({figure_paths['waveform']})")
        lines.append("")
    if figure_paths.get("loudness"):
        lines.append(f"![Loudness]({figure_paths['loudness']})")
        lines.append("")

    lines.append("## 2. K-pop 风格与情绪标签")
    lines.append("")
    lines.append(f"标签来源：`{tag_result.get('source', 'unknown')}`")
    lines.append("")
    lines.append("| 标签 | 分数 |")
    lines.append("|---|---:|")
    for item in tag_result.get("tags", [])[:12]:
        lines.append(f"| {item.get('tag')} | {_fmt(item.get('score'), 3)} |")
    lines.append("")

    if report_mode == "tag_only":
        lines.append("## 3. Tag-only baseline 摘要")
        lines.append("")
        lines.append(
            "本 baseline 只依据 style/mood/arrangement 标签生成简短描述，不使用段落、stem 或证据贡献；"
            "它用于和 evidence-grounded 报告做人工评测对比。"
        )
        top = [str(item.get("tag")) for item in tag_result.get("tags", [])[:8]]
        lines.append("")
        lines.append(f"- 主要候选标签：{'、'.join(top) if top else '无明确标签'}。")
        lines.append(f"- 标签来源：`{tag_result.get('source', 'unknown')}`。")
        lines.append("- 不确定性：该模式不解释标签为何成立，也不判断具体段落功能。")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 3. 自动赏析摘要")
    lines.append("")
    lines.append(explanations.get("summary", ""))
    lines.append("")

    lines.append("## 4. 全局听感线索")
    lines.append("")
    for cue in explanations.get("global_cues", []):
        lines.append(f"- {cue}")
    for cue in explanations.get("tag_cues", []):
        lines.append(f"- {cue}")
    lines.append("")

    lines.append("## 5. 段落与编曲推进")
    lines.append("")
    if figure_paths.get("segments"):
        lines.append(f"![Segments]({figure_paths['segments']})")
        lines.append("")
    lines.append("| 时间 | 推测段落 | 置信度 | 依据 |")
    lines.append("|---|---|---:|---|")
    for seg in explanations.get("segment_cues", []):
        lines.append(f"| {seg.get('time')} | {seg.get('label_guess')} | {_fmt(seg.get('confidence'), 2)} | {seg.get('reason')} |")
    lines.append("")
    for cue in explanations.get("arrangement_cues", []):
        lines.append(f"- {cue}")
    for cue in explanations.get("production_cues", []):
        lines.append(f"- {cue}")
    lines.append("")

    if stem_features:
        lines.append("## 6. 声部级分析")
        lines.append("")
        if figure_paths.get("stem_energy"):
            lines.append(f"![Stem energy]({figure_paths['stem_energy']})")
            lines.append("")
        lines.append("| Stem | RMS 占比 | 起音密度 | 低频占比 | 高频占比 |")
        lines.append("|---|---:|---:|---:|---:|")
        for stem in ["vocals", "drums", "bass", "other"]:
            info = stem_features.get(stem, {})
            if not info:
                continue
            lines.append(
                f"| {stem} | {_fmt(info.get('relative_rms_share'), 3)} | "
                f"{_fmt(info.get('onset_density_per_sec'), 2)} | "
                f"{_fmt(info.get('low_energy_ratio_mean'), 3)} | "
                f"{_fmt(info.get('high_energy_ratio_mean'), 3)} |"
            )
        lines.append("")
        for cue in explanations.get("stem_cues", []):
            lines.append(f"- {cue}")
        for cue in explanations.get("stem_section_cues", []):
            lines.append(f"- {cue}")
        lines.append("")

    if stem_contribution:
        lines.append("## 7. 声部贡献解释")
        lines.append("")
        lines.append(f"贡献模式：`{stem_contribution.get('mode', 'unknown')}`")
        lines.append("")
        lines.append("| 标签 | full mix | vocals | drums | bass | other | rhythm | structure |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in stem_contribution.get("tag_contributions", [])[:12]:
            c = item.get("contribution", {})
            lines.append(
                f"| {item.get('tag')} | {_fmt(c.get('full_mix'), 2)} | {_fmt(c.get('vocals'), 2)} | "
                f"{_fmt(c.get('drums'), 2)} | {_fmt(c.get('bass'), 2)} | {_fmt(c.get('other'), 2)} | "
                f"{_fmt(c.get('rhythm'), 2)} | {_fmt(c.get('structure'), 2)} |"
            )
        lines.append("")
        for item in stem_contribution.get("tag_contributions", [])[:6]:
            ev = item.get("evidence", [])
            if ev:
                lines.append(f"- **{item.get('tag')}**：{'；'.join(ev[:2])}")
        lines.append("")

    lines.append("## 8. 不确定性与可改进方向")
    lines.append("")
    lines.append("- 使用 `scripts/extract_embeddings.py` 与 `scripts/train_classifier.py --input-mode fusion` 训练 MERT + 声学特征融合分类器。")
    lines.append("- 构建小规模人工标注集，标注 dance break、pre-chorus build、drop chorus、vocal layering 等 K-pop 编曲标签。")
    lines.append("- 用人工评价比较 `full mix only` 与 `full mix + stems` 两种报告的准确性和可读性。")
    lines.append("")
    return "\n".join(lines)
