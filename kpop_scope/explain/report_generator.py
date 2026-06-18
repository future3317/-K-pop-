from __future__ import annotations

from pathlib import Path
from typing import Iterable

from kpop_scope.taxonomy import load_taxonomy
from kpop_scope.utils import human_time


STYLE_GROUP = "Style"
MOOD_GROUP = "Mood"
ARRANGEMENT_GROUP = "Arrangement"
STRUCTURE_GROUP = "Structure"
REPORT_GROUPS = [STYLE_GROUP, MOOD_GROUP, ARRANGEMENT_GROUP, STRUCTURE_GROUP]
STEM_NAMES = ["vocals", "drums", "bass", "other"]


def _fmt(x: float | int | None, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _score_label(score: float | None) -> str:
    try:
        value = float(score)
    except Exception:
        return "中等"
    if value >= 0.78:
        return "较高"
    if value >= 0.55:
        return "中等"
    return "较低"


def _md_path(path: str | Path | None) -> str:
    return str(path or "").replace("\\", "/")


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _tag_items(tag_result: dict, limit: int = 16) -> list[dict]:
    return list(tag_result.get("tags", []) or [])[:limit]


def _taxonomy_lookup() -> tuple[dict[str, dict], dict[str, list[str]]]:
    lookup: dict[str, dict] = {}
    groups: dict[str, list[str]] = {}
    for group, items in load_taxonomy().items():
        groups[group] = []
        for item in items:
            name = str(item.get("name", ""))
            lookup[name] = item
            groups[group].append(name)
    aliases = {
        "retro synth": "retro pop",
        "low-frequency drive": "sub-bass drive",
        "pre-chorus build-up": "pre-chorus buildup",
        "dance break likely": "dance break",
        "rap section likely": "rap section",
        "bright high-frequency synth": "electropop",
    }
    for src, dst in aliases.items():
        if dst in lookup:
            lookup[src] = lookup[dst]
    return lookup, groups


def _grouped_tags(tag_result: dict) -> dict[str, list[dict]]:
    lookup, groups = _taxonomy_lookup()
    group_sets = {g: set(v) for g, v in groups.items()}
    alias_group = {
        "retro synth": STYLE_GROUP,
        "low-frequency drive": ARRANGEMENT_GROUP,
        "pre-chorus build-up": ARRANGEMENT_GROUP,
        "dance break likely": ARRANGEMENT_GROUP,
        "rap section likely": ARRANGEMENT_GROUP,
        "bright high-frequency synth": ARRANGEMENT_GROUP,
    }
    out = {g: [] for g in REPORT_GROUPS}
    out["Other"] = []
    for item in _tag_items(tag_result, limit=20):
        tag = str(item.get("tag", ""))
        group = alias_group.get(tag)
        if not group:
            for g, names in group_sets.items():
                if tag in names:
                    group = g
                    break
        out[group or "Other"].append({**item, "info": lookup.get(tag, {})})
    return out


def _join_tag_names(items: list[dict], limit: int = 4) -> str:
    names = [str(i.get("tag")) for i in items[:limit]]
    return "、".join(names) if names else "未形成明确高置信候选"


def _top_tags_by_group(grouped: dict[str, list[dict]], group: str, limit: int = 4) -> list[str]:
    return [str(i.get("tag")) for i in grouped.get(group, [])[:limit]]


def _loudness_dynamic_range(features: dict) -> float:
    stats = features.get("loudness", {}).get("rms_db_relative_stats", {})
    return float((stats.get("p90", 0.0) or 0.0) - (stats.get("p10", 0.0) or 0.0))


def _low_frequency_ratio(features: dict) -> float:
    band = features.get("spectral", {}).get("band_energy_ratio", {})
    return float((band.get("low_20_250", {}) or {}).get("mean", 0.0) or 0.0)


def _time_range(seg: dict) -> str:
    return f"{human_time(seg.get('start', 0.0))}–{human_time(seg.get('end', 0.0))}"


def _clean_label(label: str) -> str:
    aliases = {
        "chorus/drop": "chorus/drop 候选",
        "pre-chorus": "pre-chorus / buildup",
        "dance break": "dance break 候选",
        "high-energy section": "高能段落",
        "refrain": "refrain / hook 回归",
        "bridge": "bridge / 对比段",
        "outro": "outro / 收束段",
        "intro": "intro / 进入段",
        "verse": "verse / 主歌段",
        "unknown": "功能不确定",
    }
    return aliases.get(label, label or "功能不确定")


def _segment_confidence(seg: dict) -> float:
    return float(seg.get("confidence", seg.get("label_confidence", 0.0)) or 0.0)


def _segment_reason(seg: dict) -> str:
    label = str(seg.get("label_guess", "unknown"))
    confidence = _segment_confidence(seg)
    evidence = _dedupe(seg.get("evidence", []) or [])
    if label == "chorus/drop":
        return "能量和重复性较突出，可能承担 hook 或高潮区域"
    if label == "high-energy section":
        return "整曲能量较高，但不足以确认具体 chorus/drop 功能"
    if label == "pre-chorus":
        return "与前后段相比存在蓄力或过渡倾向"
    if label == "dance break":
        return "节奏活跃度较高，可能偏表演段落"
    if label == "bridge":
        return "位置和能量变化显示一定对比功能"
    if label == "outro":
        return "位于尾部，可能承担收束功能"
    if label == "intro":
        return "位于开头，主要承担进入和铺垫功能"
    if label in {"unknown", "refrain"} and evidence:
        return evidence[0]
    if confidence < 0.45:
        return "证据不足以确认具体功能"
    return evidence[0] if evidence else "基于整曲声学特征的弱推测"


def _important_segments(features: dict, limit: int = 6) -> list[dict]:
    segments = list(features.get("segments", {}).get("segments", []) or [])
    if not segments:
        return []
    selected: list[dict] = []

    chorus = [s for s in segments if s.get("label_guess") == "chorus/drop"]
    chorus = sorted(chorus, key=_segment_confidence, reverse=True)[:3]
    selected.extend(chorus)

    for label in ["intro", "pre-chorus", "dance break", "bridge", "outro"]:
        candidates = [s for s in segments if s.get("label_guess") == label]
        if candidates:
            selected.append(max(candidates, key=_segment_confidence))

    if len(selected) < limit:
        extras = sorted(
            segments,
            key=lambda s: (_segment_confidence(s), float(s.get("energy_mean", 0.0) or 0.0)),
            reverse=True,
        )
        for seg in extras:
            if seg not in selected:
                selected.append(seg)
            if len(selected) >= limit:
                break
    return sorted(selected[:limit], key=lambda s: float(s.get("start", 0.0) or 0.0))


def _chorus_summary(features: dict) -> str:
    segments = list(features.get("segments", {}).get("segments", []) or [])
    candidates = [s for s in segments if s.get("label_guess") == "chorus/drop"]
    if not candidates:
        return "系统没有检测到足够稳定的 chorus/drop 候选，结构判断更适合保守解释为若干高能或过渡段。"
    candidates = sorted(candidates, key=_segment_confidence, reverse=True)[:3]
    strongest = candidates[0]
    ranges = "、".join(_time_range(s) for s in candidates[1:])
    if ranges:
        return f"系统检测到 {len(candidates)} 个 chorus/drop 候选，最强候选为 {_time_range(strongest)}；其余相近候选为 {ranges}。"
    return f"系统检测到 1 个主要 chorus/drop 候选，位置为 {_time_range(strongest)}。"


def _group_explanation(group: str, items: list[dict]) -> str:
    names = _join_tag_names(items)
    if group == STYLE_GROUP:
        return f"风格层面主要集中在 {names}，用于概括歌曲整体制作语汇和流行类型倾向。"
    if group == MOOD_GROUP:
        return f"情绪层面主要集中在 {names}，表示系统对整体听感色彩的判断。"
    if group == ARRANGEMENT_GROUP:
        return f"编曲层面主要集中在 {names}，更接近对节奏、音色、hook 和层次设计的粗粒度描述。"
    if group == STRUCTURE_GROUP:
        return f"结构层面主要集中在 {names}，只表示歌曲可能存在的整体结构倾向，不等同于人工段落标注。"
    return f"其他候选包括 {names}。"


def _tag_only_paragraphs(features: dict, tag_result: dict) -> list[str]:
    grouped = _grouped_tags(tag_result)
    bpm = float(features.get("tempo", {}).get("bpm", 0.0) or 0.0)
    onset_density = float(features.get("onset", {}).get("onset_density_per_sec", 0.0) or 0.0)
    centroid = float(features.get("spectral", {}).get("centroid_hz", {}).get("mean", 0.0) or 0.0)
    dyn = _loudness_dynamic_range(features)
    style_names = {str(i.get("tag")) for i in grouped[STYLE_GROUP]}
    mood_names = {str(i.get("tag")) for i in grouped[MOOD_GROUP]}
    arr_names = {str(i.get("tag")) for i in grouped[ARRANGEMENT_GROUP]}

    paragraphs: list[str] = []
    if {"dance-pop", "k-pop dance"} & style_names or "energetic" in mood_names:
        paragraphs.append(
            f"从标签层面看，这首歌更接近舞曲化的 K-pop 流行取向。系统给出的主要风格候选包括"
            f"{_join_tag_names(grouped[STYLE_GROUP])}，情绪候选包括{_join_tag_names(grouped[MOOD_GROUP])}。"
            f"BPM 约为 {_fmt(bpm, 1)}，起音密度约为 {_fmt(onset_density, 2)} 次/秒，"
            "这些基础特征支持一种偏节奏驱动、适合舞台表演和副歌记忆点塑造的整体听感。"
        )
    else:
        paragraphs.append(
            f"从标签层面看，系统没有把这首歌简单归入持续高能舞曲，而是给出"
            f"{_join_tag_names(grouped[STYLE_GROUP])} 等候选。BPM 约为 {_fmt(bpm, 1)}，"
            f"起音密度约为 {_fmt(onset_density, 2)} 次/秒，说明整体律动强度需要结合人工听感进一步确认。"
        )

    if {"confident", "dark", "dramatic"} & mood_names:
        paragraphs.append(
            f"情绪标签显示歌曲可能带有更强的力量感、冷色调或戏剧化张力。当前高分 mood 包括{_join_tag_names(grouped[MOOD_GROUP])}。"
            "在 tag-only 模式下，这种判断只代表整曲层面的情绪倾向，不能说明具体哪一段制造了这种张力。"
        )
    elif {"bright", "cute", "playful", "euphoric"} & mood_names:
        paragraphs.append(
            f"情绪标签更偏明亮、轻快或上扬。平均频谱质心约为 {_fmt(centroid, 1)} Hz；若同时出现 bright/cute/energetic 等标签，"
            "可以理解为系统倾向于把它归为更清爽、更容易形成正向听感的 K-pop 表达。"
        )
    elif grouped[MOOD_GROUP]:
        paragraphs.append(
            f"情绪候选主要集中在{_join_tag_names(grouped[MOOD_GROUP])}。这些标签适合作为人工听感复核的入口，而不是最终音乐学结论。"
        )

    if {"electropop", "synth-pop", "retro pop", "retro synth"} & style_names or "bright high-frequency synth" in arr_names:
        paragraphs.append(
            "风格标签中出现 electropop、synth-pop 或 retro pop 倾向时，说明系统认为整曲音色可能更接近合成器流行语境。"
            f"当前平均频谱质心为 {_fmt(centroid, 1)} Hz，响度动态范围约为 {_fmt(dyn, 1)} dB。"
            "tag-only 报告只能描述这种整体音色倾向，不能进一步定位具体 synth hook 或伴奏层次。"
        )

    if {"pre-chorus build-up", "pre-chorus buildup", "drop chorus", "chorus energy lift"} & arr_names:
        paragraphs.append(
            "编曲标签层面还显示出“蓄力-释放”结构的可能性，例如 pre-chorus build-up、drop chorus 或 chorus energy lift。"
            "这里的表述刻意保持在整体倾向层面：系统倾向于认为歌曲可能存在 K-pop 常见的副歌前铺垫与高潮释放，"
            "但本 baseline 不使用具体段落边界，因此不标注任何时间点。"
        )

    return paragraphs[:4]


def _generate_tag_only_report(audio_path: Path, features: dict, tag_result: dict, title_prefix: str) -> str:
    tempo = features.get("tempo", {})
    key = features.get("key", {})
    onset = features.get("onset", {})
    loudness = features.get("loudness", {})
    spectral = features.get("spectral", {})
    duration = features.get("duration_seconds", 0.0)
    grouped = _grouped_tags(tag_result)

    lines: list[str] = []
    lines.append(f"# KPopScope Tag-only Baseline: {audio_path.name}")
    lines.append("")
    lines.append("> 本报告是标签驱动 baseline：它使用基础音频特征、标签分数和 taxonomy 描述生成整体赏析，不使用段落时间线、Demucs stems 或 stem contribution。")
    if tag_result.get("source", "").startswith("acoustic_prior"):
        lines.append("> 当前标签来自声学先验和规则 fallback，不等同于训练好的 K-pop 模型预测。")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.append(f"- **时长**：{human_time(duration)}")
    lines.append(f"- **估计 BPM**：{_fmt(tempo.get('bpm'), 1)}")
    lines.append(f"- **估计调性**：{key.get('key', 'unknown')}，置信度约 {_fmt(key.get('confidence'), 2)}")
    lines.append(f"- **起音密度**：{_fmt(onset.get('onset_density_per_sec'), 2)} 次/秒")
    lines.append(f"- **平均频谱质心**：{_fmt(spectral.get('centroid_hz', {}).get('mean'), 1)} Hz")
    lines.append(f"- **响度动态范围**：{_fmt(_loudness_dynamic_range(features), 1)} dB")
    lines.append("")
    lines.append("## 2. 标签预测结果")
    lines.append("")
    lines.append(f"标签来源：`{tag_result.get('source', 'unknown')}`")
    for group, title in [
        (STYLE_GROUP, "风格标签"),
        (MOOD_GROUP, "情绪标签"),
        (ARRANGEMENT_GROUP, "编曲标签"),
        (STRUCTURE_GROUP, "结构倾向标签"),
    ]:
        lines.append("")
        lines.append(f"### {title}")
        items = grouped.get(group, [])
        if not items:
            lines.append("暂无高分候选。")
            continue
        lines.append("| 标签 | 分数 | 标签说明 | 常见正向线索 |")
        lines.append("|---|---:|---|---|")
        for item in items[:6]:
            info = item.get("info", {}) or {}
            cues = "、".join(str(x) for x in info.get("positive_cues", [])[:3])
            lines.append(
                f"| {item.get('tag')} | {_fmt(item.get('score'), 3)} | "
                f"{info.get('description_zh', '标签层面的候选倾向')} | {cues or '需结合听感复核'} |"
            )
    lines.append("")
    lines.append("## 3. 基于标签的整体听感分析")
    lines.append("")
    for paragraph in _tag_only_paragraphs(features, tag_result):
        lines.append(paragraph)
        lines.append("")
    lines.append("## 4. K-pop 制作语境下的解释")
    lines.append("")
    lines.append(
        f"在 K-pop 制作语境下，{_join_tag_names(grouped[STYLE_GROUP])} 与 {_join_tag_names(grouped[MOOD_GROUP])} "
        "共同描述了歌曲的整体概念轮廓。如果 dance-pop、k-pop dance、energetic 等标签分数较高，"
        "它通常意味着歌曲更强调舞台表演、稳定律动和易记 hook；如果 confident、dark 或 dramatic 较高，"
        "则说明系统在标签层面捕捉到更强的力量感或对比张力。"
    )
    lines.append("")
    lines.append(
        f"编曲候选包括{_join_tag_names(grouped[ARRANGEMENT_GROUP])}。"
        "例如 heavy drums 或 low-frequency drive 倾向可以被理解为节奏和低频推动更重要；"
        "vocal layering 倾向表示系统认为人声层次可能更厚；chant hook 倾向则表示 hook 设计可能更偏口号式和记忆点导向。"
        "这些解释都只来自标签层面，不能声称来自具体 stem 或具体段落。"
    )
    lines.append("")
    lines.append("## 5. 局限性")
    lines.append("")
    lines.append("- 本报告是 tag-only baseline，不使用声部分离、段落边界或 stem contribution。")
    lines.append("- 因此它只能给出整曲层面的标签倾向，不能解释具体哪一段、哪一个声部导致这些判断。")
    lines.append("- 结构类标签只表示整体倾向，不包含 chorus/drop、dance break 或 bridge 的具体时间点。")
    lines.append("- 若需要可追溯的段落证据、声部贡献和更细粒度编曲解释，应阅读 evidence-grounded 报告。")
    lines.append("")
    return "\n".join(lines)


def _readable_conclusion(features: dict, tag_result: dict, stem_features: dict | None) -> str:
    grouped = _grouped_tags(tag_result)
    styles = _join_tag_names(grouped[STYLE_GROUP], limit=2)
    moods = _join_tag_names(grouped[MOOD_GROUP], limit=2)
    bpm = _fmt(features.get("tempo", {}).get("bpm"), 1)
    onset = _fmt(features.get("onset", {}).get("onset_density_per_sec"), 2)
    chorus_text = _chorus_summary(features)
    stem_text = (
        "本次启用了 stems，可进一步参考声部贡献解释。"
        if stem_features
        else "由于本次未启用或未成功生成 stems，声部级结论只能作为整曲声学推测。"
    )
    confidence = _score_label((tag_result.get("tags") or [{}])[0].get("score"))
    return (
        f"系统认为这首歌整体偏 {styles}，主要情绪倾向为 {moods}，速度约 {bpm} BPM，起音密度约 {onset} 次/秒。"
        f"{chorus_text} 当前分析置信度为{confidence}；{stem_text}"
    )


def _readable_tag_section(tag_result: dict) -> list[str]:
    grouped = _grouped_tags(tag_result)
    lines: list[str] = []
    lines.append("## 3. 标签解释")
    lines.append("")
    lines.append(f"标签来源：`{tag_result.get('source', 'unknown')}`。下表只展示 top tags，避免把弱候选当作确定结论。")
    lines.append("")
    lines.append("| 组别 | Top tags | 解释 |")
    lines.append("|---|---|---|")
    for group, title in [
        (STYLE_GROUP, "风格"),
        (MOOD_GROUP, "情绪"),
        (ARRANGEMENT_GROUP, "编曲"),
        (STRUCTURE_GROUP, "结构"),
    ]:
        items = grouped.get(group, [])[:4]
        lines.append(f"| {title} | {_join_tag_names(items)} | {_group_explanation(group, items)} |")
    lines.append("")
    lines.append("Top 8 标签分数：")
    lines.append("")
    lines.append("| 标签 | 分数 | 置信描述 |")
    lines.append("|---|---:|---|")
    for item in _tag_items(tag_result, limit=8):
        lines.append(f"| {item.get('tag')} | {_fmt(item.get('score'), 3)} | {_score_label(item.get('score'))} |")
    lines.append("")
    return lines


def _structure_paragraphs(features: dict) -> list[str]:
    segments = list(features.get("segments", {}).get("segments", []) or [])
    if not segments:
        return ["当前未得到稳定段落边界，因此结构赏析只保留整曲层面的节奏和能量判断。"]
    intro = next((s for s in segments if s.get("label_guess") == "intro"), segments[0])
    outro = next((s for s in reversed(segments) if s.get("label_guess") == "outro"), segments[-1])
    chorus = [s for s in segments if s.get("label_guess") == "chorus/drop"]
    pre = [s for s in segments if s.get("label_guess") == "pre-chorus"]
    bridge = [s for s in segments if s.get("label_guess") == "bridge"]

    paragraphs = [
        f"开头的 {_time_range(intro)} 更像进入和铺垫区域，系统主要依据其位置、能量水平和后续变化来做保守判断。"
    ]
    if pre:
        paragraphs.append(
            f"在 {_time_range(pre[0])} 附近，系统观察到一定的蓄力或过渡倾向；这适合解释为进入高潮前的准备，但仍需人工听感确认。"
        )
    if chorus:
        strongest = sorted(chorus, key=_segment_confidence, reverse=True)[0]
        paragraphs.append(
            f"最强 chorus/drop 候选位于 {_time_range(strongest)}，其能量和重复性比周边段落更突出，因此更可能承担 hook 或高潮功能。"
        )
    else:
        paragraphs.append("系统没有强行指定 chorus/drop；若只出现高能和重复线索，报告会把它保守描述为高能段落或功能不确定。")
    if bridge:
        paragraphs.append(f"{_time_range(bridge[0])} 可能承担 bridge 或对比功能，作用更接近在后段制造听感变化。")
    else:
        paragraphs.append(f"尾部的 {_time_range(outro)} 更像收束区域，当前证据不足以展开更细的人工段落命名。")
    return paragraphs[:4]


def _full_mix_acoustic_evidence(features: dict) -> list[str]:
    onset_density = float(features.get("onset", {}).get("onset_density_per_sec", 0.0) or 0.0)
    low_ratio = _low_frequency_ratio(features)
    centroid = float(features.get("spectral", {}).get("centroid_hz", {}).get("mean", 0.0) or 0.0)
    evidence: list[str] = []
    if onset_density >= 2.0:
        evidence.append(f"起音密度较高（{_fmt(onset_density, 2)} 次/秒）→ 可能存在较活跃的鼓点、切分或节奏型。")
    else:
        evidence.append(f"起音密度为 {_fmt(onset_density, 2)} 次/秒 → 节奏活跃度不宜过度解读。")
    if low_ratio >= 0.18:
        evidence.append(f"低频能量占比约 {_fmt(low_ratio, 3)} → 可能存在较明显的低频推进。")
    if centroid >= 2600:
        evidence.append(f"平均频谱质心约 {_fmt(centroid, 1)} Hz → 高频 synth、hi-hat 或 FX 可能更突出。")
    elif centroid > 0:
        evidence.append(f"平均频谱质心约 {_fmt(centroid, 1)} Hz → 整体音色亮度处于可参考范围。")
    return evidence[:3]


def _readable_stem_section(features: dict, stem_features: dict | None, stem_contribution: dict | None) -> list[str]:
    lines: list[str] = []
    lines.append("## 5. 声部分析状态")
    lines.append("")
    if not stem_features:
        lines.append(
            "本次未启用或未成功生成 Demucs stems，因此无法判断 vocals、drums、bass、other 的独立贡献。"
            "以下关于鼓组、低频或人声的说法仅来自整曲声学特征，而不是声部分离结果。"
        )
        lines.append("")
        for item in _full_mix_acoustic_evidence(features):
            lines.append(f"- {item}")
        lines.append("")
        return lines

    lines.append("本次检测到 Demucs stems，因此只展示高分标签中最有信息量的 top contributing sources，不显示完整矩阵或接近 0 的贡献值。")
    lines.append("")
    tag_rows = list((stem_contribution or {}).get("tag_contributions", []) or [])[:3]
    if not tag_rows:
        lines.append("- 当前没有稳定的声部贡献结果。")
        lines.append("")
        return lines
    for item in tag_rows:
        contrib = item.get("contribution", {}) or {}
        ranked = [
            (name, float(contrib.get(name, 0.0) or 0.0))
            for name in [*STEM_NAMES, "rhythm", "structure"]
            if float(contrib.get(name, 0.0) or 0.0) > 0.05
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        sources = "、".join(f"{name} {_fmt(value, 2)}" for name, value in ranked[:3]) or "未形成明显声部贡献"
        evidence = "；".join(_dedupe(item.get("evidence", []) or [])[:2])
        lines.append(f"- **{item.get('tag')}**：{sources}。{evidence or '依据为 stem 级声学特征的相对强弱。'}")
    lines.append("")
    return lines


def _generate_readable_evidence_report(
    audio_path: Path,
    features: dict,
    tag_result: dict,
    explanations: dict,
    stem_features: dict | None,
    figure_paths: dict,
    stem_contribution: dict | None,
) -> str:
    tempo = features.get("tempo", {})
    key = features.get("key", {})
    onset = features.get("onset", {})
    spectral = features.get("spectral", {})
    duration = features.get("duration_seconds", 0.0)
    important = _important_segments(features, limit=6)

    lines: list[str] = []
    lines.append(f"# KPopScope Evidence-grounded Analysis: {audio_path.name}")
    lines.append("")
    lines.append("> readable 模式面向人评和论文正文：保留关键证据，省略长段落表、完整贡献矩阵和重复 debug 句子。")
    if tag_result.get("source", "").startswith("acoustic_prior"):
        lines.append("> 当前标签来自声学先验和规则 fallback，不等同于训练好的 K-pop 模型预测。")
    lines.append("")
    lines.append("## 1. 一句话结论")
    lines.append("")
    lines.append(_readable_conclusion(features, tag_result, stem_features))
    lines.append("")
    lines.append("## 2. 基础信息")
    lines.append("")
    lines.append(f"- **时长**：{human_time(duration)}")
    lines.append(f"- **估计 BPM**：{_fmt(tempo.get('bpm'), 1)}")
    lines.append(f"- **估计调性**：{key.get('key', 'unknown')}，置信度约 {_fmt(key.get('confidence'), 2)}")
    lines.append(f"- **起音密度**：{_fmt(onset.get('onset_density_per_sec'), 2)} 次/秒")
    lines.append(f"- **平均频谱质心**：{_fmt(spectral.get('centroid_hz', {}).get('mean'), 1)} Hz")
    lines.append(f"- **响度动态范围**：{_fmt(_loudness_dynamic_range(features), 1)} dB")
    lines.append("")
    for key_name, title in [("waveform", "Waveform"), ("loudness", "Loudness")]:
        if figure_paths.get(key_name):
            lines.append(f"![{title}]({_md_path(figure_paths[key_name])})")
            lines.append("")
    lines.extend(_readable_tag_section(tag_result))
    lines.append("## 4. 结构与段落摘要")
    lines.append("")
    lines.append(_chorus_summary(features))
    lines.append("")
    if figure_paths.get("segments"):
        lines.append(f"![Segments]({_md_path(figure_paths['segments'])})")
        lines.append("")
    if important:
        lines.append("| 时间 | 功能推测 | 置信度 | 主要依据 |")
        lines.append("|---|---|---:|---|")
        chorus_seen = 0
        for seg in important:
            reason = _segment_reason(seg)
            if seg.get("label_guess") == "chorus/drop":
                chorus_seen += 1
                if chorus_seen > 1:
                    reason = "同类高能 hook 候选，已在摘要中聚合，不再重复展开"
            lines.append(
                f"| {_time_range(seg)} | {_clean_label(str(seg.get('label_guess', 'unknown')))} | "
                f"{_fmt(_segment_confidence(seg), 2)} | {reason} |"
            )
        lines.append("")
    else:
        lines.append("当前没有稳定段落候选。")
        lines.append("")
    for paragraph in _dedupe(_structure_paragraphs(features)):
        lines.append(paragraph)
        lines.append("")
    lines.extend(_readable_stem_section(features, stem_features, stem_contribution))
    lines.append("## 6. 不确定性说明")
    lines.append("")
    lines.append("- 本报告将自动段落、声学特征和标签结果作为证据线索，不等同于人工音乐学标注。")
    lines.append("- 若未启用 stems，关于鼓组、低频和人声的描述均来自整曲特征，不能解释为独立声部贡献。")
    lines.append("- readable 模式会主动合并相邻重复段落，并只展示最重要的候选；需要完整调试信息时可使用 `--report-detail technical`。")
    lines.append("")
    return "\n".join(lines)


def _generate_technical_evidence_report(
    audio_path: Path,
    features: dict,
    tag_result: dict,
    explanations: dict,
    stem_features: dict | None,
    figure_paths: dict,
    title_prefix: str,
    report_mode: str,
    report_detail: str,
    stem_contribution: dict | None,
) -> str:
    tempo = features.get("tempo", {})
    key = features.get("key", {})
    onset = features.get("onset", {})
    spectral = features.get("spectral", {})
    duration = features.get("duration_seconds", 0.0)

    lines: list[str] = []
    lines.append(f"# {title_prefix}: {audio_path.name}")
    lines.append("")
    lines.append("> 本报告由 KPopScope 自动生成。结构标签与赏析文字是基于音频特征和规则/模型的推测，不等同于人工音乐学标注。")
    if tag_result.get("source", "").startswith("acoustic_prior"):
        lines.append("> 当前标签来自声学先验和规则 fallback，不等同于训练好的 K-pop 模型预测。")
    lines.append(f"> 报告模式：`{report_mode}`；细节级别：`{report_detail}`。")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.append(f"- **时长**：{human_time(duration)}")
    lines.append(f"- **估计 BPM**：{_fmt(tempo.get('bpm'), 1)}")
    lines.append(f"- **估计调性**：{key.get('key', 'unknown')}，置信度约 {_fmt(key.get('confidence'), 2)}")
    lines.append(f"- **起音密度**：{_fmt(onset.get('onset_density_per_sec'), 2)} 次/秒")
    lines.append(f"- **平均频谱质心**：{_fmt(spectral.get('centroid_hz', {}).get('mean'), 1)} Hz")
    lines.append("")
    for key_name, title in [("waveform", "Waveform"), ("loudness", "Loudness")]:
        if figure_paths.get(key_name):
            lines.append(f"![{title}]({_md_path(figure_paths[key_name])})")
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
    lines.append("## 3. 自动赏析摘要")
    lines.append("")
    lines.append(explanations.get("summary", ""))
    lines.append("")
    lines.append("## 4. 全局听感线索")
    lines.append("")
    for cue in _dedupe([*explanations.get("global_cues", []), *explanations.get("tag_cues", [])]):
        lines.append(f"- {cue}")
    lines.append("")
    lines.append("## 5. 段落与编曲推断")
    lines.append("")
    if figure_paths.get("segments"):
        lines.append(f"![Segments]({_md_path(figure_paths['segments'])})")
        lines.append("")
    lines.append("| 时间 | 推测段落 | 置信度 | 依据 |")
    lines.append("|---|---|---:|---|")
    for seg in explanations.get("segment_cues", []):
        lines.append(f"| {seg.get('time')} | {seg.get('label_guess')} | {_fmt(seg.get('confidence'), 2)} | {seg.get('reason')} |")
    lines.append("")
    for cue in _dedupe([*explanations.get("arrangement_cues", []), *explanations.get("production_cues", [])]):
        lines.append(f"- {cue}")
    lines.append("")
    lines.extend(_readable_stem_section(features, stem_features, stem_contribution))
    if stem_features and stem_contribution:
        lines.append("## 6. 声部贡献矩阵")
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
    lines.append("## 7. 不确定性与可改进方向")
    lines.append("")
    lines.append("- 使用 `scripts/extract_embeddings.py` 与 `scripts/train_classifier.py --input-mode fusion` 训练 MERT + 声学特征融合分类器。")
    lines.append("- 构建小规模人工标注集，标注 dance break、pre-chorus build、drop chorus、vocal layering 等 K-pop 编曲标签。")
    lines.append("- 用人工评价比较 `tag_only` 与 `evidence_grounded` 两种报告的准确性、信息量和可读性。")
    lines.append("")
    return "\n".join(lines)


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
    report_detail: str = "readable",
) -> str:
    """Generate a Markdown report in Chinese."""
    audio_path = Path(audio_path)
    figure_paths = figure_paths or {}
    report_detail = (report_detail or "readable").lower()

    if report_mode == "tag_only":
        return _generate_tag_only_report(audio_path, features, tag_result, title_prefix)
    if report_detail == "readable":
        return _generate_readable_evidence_report(
            audio_path=audio_path,
            features=features,
            tag_result=tag_result,
            explanations=explanations,
            stem_features=stem_features,
            figure_paths=figure_paths,
            stem_contribution=stem_contribution,
        )
    return _generate_technical_evidence_report(
        audio_path=audio_path,
        features=features,
        tag_result=tag_result,
        explanations=explanations,
        stem_features=stem_features,
        figure_paths=figure_paths,
        title_prefix=title_prefix,
        report_mode=report_mode,
        report_detail=report_detail,
        stem_contribution=stem_contribution,
    )
