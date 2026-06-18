from __future__ import annotations

from collections import Counter
from typing import Any

from kpop_scope.utils import human_time


def _top_tags(tag_result: dict, limit: int = 10, min_score: float = 0.15) -> list[str]:
    tags = []
    for item in tag_result.get("tags", []):
        if item.get("score", 0.0) >= min_score:
            tags.append(str(item.get("tag")))
        if len(tags) >= limit:
            break
    return tags or tag_result.get("top_tags", [])[:limit]


def _score(tag_result: dict, tag: str) -> float:
    for item in tag_result.get("tags", []):
        if str(item.get("tag")) == tag:
            return float(item.get("score", 0.0) or 0.0)
    return 0.0


def _band(features: dict, name: str) -> float:
    return float(
        features.get("spectral", {})
        .get("band_energy_ratio", {})
        .get(name, {})
        .get("mean", 0.0)
        or 0.0
    )


def _tempo_zone(tempo: float) -> tuple[str, str]:
    t = tempo
    while 0 < t < 70:
        t *= 2
    while t > 190:
        t /= 2
    if t >= 128:
        return "club_fast", "速度落在偏快舞曲区间，容易承载 house、jersey club、EDM drop 或高密度编舞段。"
    if t >= 116:
        return "dance_midfast", "速度处在 K-pop dance-pop 常见的中高速区间，适合稳定四拍律动和副歌齐舞。"
    if t >= 96:
        return "midtempo", "速度属于中速区间，通常会在 groove、人声旋律和段落对比之间取得平衡。"
    if t >= 75:
        return "slow_mid", "速度偏慢或可能采用 half-time 听感，适合 R&B、trap-pop 或更强调人声的段落。"
    return "ballad", "速度偏慢，更接近 ballad / 抒情取向，情绪推进通常依赖旋律、和声和动态变化。"


def _segment_by_label(segments: list[dict], label: str) -> list[dict]:
    return [s for s in segments if label in str(s.get("label_guess", ""))]


def _energy_lift_between(prev: dict | None, cur: dict | None) -> float:
    if not prev or not cur:
        return 0.0
    return float(cur.get("mean_energy", 0.0) or 0.0) - float(prev.get("mean_energy", 0.0) or 0.0)


def build_explanations(features: dict, tag_result: dict, stem_features: dict | None = None) -> dict:
    """Turn structured features into higher-level K-pop interpretation cues.

    The rules are intentionally explicit so the report can explain *why* a tag or
    structure hypothesis was produced instead of only showing a probability.
    """
    tempo = float(features.get("tempo", {}).get("bpm", 0.0) or 0.0)
    key = features.get("key", {})
    onset_density = float(features.get("onset", {}).get("onset_density_per_sec", 0.0) or 0.0)
    centroid = float(features.get("spectral", {}).get("centroid_hz", {}).get("mean", 0.0) or 0.0)
    low_ratio = _band(features, "low_20_250")
    lowmid_ratio = _band(features, "lowmid_250_500")
    highmid_ratio = _band(features, "highmid_2000_6000")
    high_ratio = _band(features, "high_6000_12000")
    rms_db = features.get("loudness", {}).get("rms_db_relative_stats", {})
    dynamic_range = float((rms_db.get("p90", 0.0) or 0.0) - (rms_db.get("p10", 0.0) or 0.0))
    segments = list(features.get("segments", {}).get("segments", []) or [])

    tags = _top_tags(tag_result)
    tag_set = set(tags)
    zone, tempo_cue = _tempo_zone(tempo)

    global_cues: list[str] = [tempo_cue]
    if onset_density >= 2.8:
        global_cues.append("全曲起音密度较高，说明鼓组、切分合成器或打击性采样很活跃，听感会更偏舞台表演/编舞导向。")
    elif onset_density >= 1.8:
        global_cues.append("起音密度处于中等偏高水平，节奏推进比较稳定，但仍给人声和 hook 留出空间。")
    else:
        global_cues.append("起音密度相对克制，歌曲更可能依靠旋律线、和声铺陈或音色氛围塑造情绪。")

    if low_ratio >= 0.36:
        global_cues.append("20–250 Hz 低频占比较高，kick 与 bass 可能承担主要的重量感和推进感，适合 drop chorus 或 confident/dark 概念。")
    elif low_ratio <= 0.18:
        global_cues.append("低频占比不高，编曲重心可能更多放在人声、合成器高频或和声织体上。")
    if high_ratio >= 0.10 or centroid >= 3200:
        global_cues.append("高频亮度突出，可能存在明亮 synth、FX、hi-hat 或空气感人声处理，容易形成 bright/retro/futuristic 听感。")
    elif centroid < 2200 and low_ratio >= 0.28:
        global_cues.append("频谱整体偏暗且低频较重，听感更容易靠近 dark、trap-pop 或强概念化舞曲。")
    if dynamic_range >= 14:
        global_cues.append("响度动态范围较大，段落之间有明显的收放，适合制造 pre-chorus 蓄力到 chorus 释放的结构。")

    tag_cues: list[str] = []
    if "dance-pop" in tag_set or "k-pop dance" in tag_set:
        tag_cues.append("dance-pop/K-pop dance 标签通常对应稳定节拍、清晰重拍和副歌段能量释放；报告中的 chorus/drop 段可重点听鼓组与低频是否同步抬升。")
    if "electropop" in tag_set or "bright high-frequency synth" in tag_set:
        tag_cues.append("electropop/高频 synth 倾向说明音色设计可能比传统乐队伴奏更突出，尤其要关注副歌或 post-chorus 的合成器 hook。")
    if "hip-hop pop" in tag_set or "trap-pop" in tag_set or "rap section likely" in tag_set:
        tag_cues.append("hip-hop/trap-pop 倾向往往体现在更重的低频、切分鼓组和 rap/spoken rhythm 段落，可在 verse 或 break 中寻找节奏型变化。")
    if "ballad" in tag_set or "sentimental" in tag_set:
        tag_cues.append("ballad/sentimental 倾向说明歌曲可能更依赖人声旋律、和声层次和动态铺陈，而不是持续高密度鼓点。")
    if "jersey club influence" in tag_set:
        tag_cues.append("jersey club influence 标签常与更快的 club 速度和高密度切分律动相关；若出现在 K-pop 中，通常会强化舞蹈段的弹跳感。")
    if "vocal layering" in tag_set:
        tag_cues.append("vocal layering 倾向意味着副歌可能通过叠唱、和声或 ad-lib 增厚，而不是只靠伴奏变响来制造冲击。")

    segment_cues: list[dict[str, Any]] = []
    arrangement_cues: list[str] = []
    labels = Counter(str(s.get("label_guess", "unknown")) for s in segments)
    chorus_segments = _segment_by_label(segments, "chorus")
    pre_segments = _segment_by_label(segments, "pre-chorus")
    dance_segments = _segment_by_label(segments, "dance break")
    bridge_segments = _segment_by_label(segments, "bridge")

    for idx, seg in enumerate(segments):
        label = seg.get("label_guess", "segment")
        start = human_time(seg.get("start", 0.0))
        end = human_time(seg.get("end", 0.0))
        energy = float(seg.get("mean_energy", 0.0) or 0.0)
        onset = float(seg.get("mean_onset", 0.0) or 0.0)
        slope = float(seg.get("energy_slope", 0.0) or 0.0)
        rep = float(seg.get("repetition_score", 0.0) or 0.0)
        conf = float(seg.get("label_confidence", 0.0) or 0.0)
        reason_parts = []
        reason_parts.extend(seg.get("label_evidence", [])[:2])
        if energy >= 0.70:
            reason_parts.append("平均能量很高")
        elif energy >= 0.52:
            reason_parts.append("平均能量中高")
        elif energy <= 0.32:
            reason_parts.append("平均能量较低")
        if onset >= 0.62:
            reason_parts.append("鼓点/瞬态非常活跃")
        elif onset >= 0.42:
            reason_parts.append("节奏活动明显")
        elif onset <= 0.22:
            reason_parts.append("节奏相对稀疏")
        if slope > 0.08:
            reason_parts.append("段内能量呈上升趋势")
        elif slope < -0.08:
            reason_parts.append("段内能量呈下降趋势")
        if rep >= 0.88:
            reason_parts.append("与其他段落高度相似，可能承担重复 hook/副歌功能")
        reason = "；".join(dict.fromkeys(reason_parts)) if reason_parts else "根据能量、节奏、频谱和位置综合推断"
        segment_cues.append(
            {
                "time": f"{start}–{end}",
                "label_guess": label,
                "confidence": conf,
                "reason": reason,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            }
        )

        if "pre-chorus" in str(label) and idx + 1 < len(segments):
            lift = _energy_lift_between(seg, segments[idx + 1])
            if lift > 0.08:
                arrangement_cues.append(
                    f"{start}–{end} 的 pre-chorus 后接能量更高的段落，呈现 K-pop 常见的“蓄力—释放”结构。"
                )
        if "chorus" in str(label):
            arrangement_cues.append(
                f"{start}–{end} 被推测为 chorus/drop：这一段可重点听 hook 是否重复、鼓组是否更满、低频是否更强。"
            )
        if "dance break" in str(label):
            arrangement_cues.append(
                f"{start}–{end} 具有 dance break/post-chorus 倾向：高密度瞬态和较短段落时长会强化舞蹈展示感。"
            )
        if "bridge" in str(label):
            arrangement_cues.append(
                f"{start}–{end} 类似 bridge：它通常通过能量回落、和声变化或音色减法，为最后一次副歌做对比。"
            )

    if chorus_segments:
        best_chorus = max(chorus_segments, key=lambda s: float(s.get("mean_energy", 0.0) or 0.0))
        arrangement_cues.append(
            f"全曲最强副歌候选出现在 {human_time(best_chorus.get('start', 0.0))}–{human_time(best_chorus.get('end', 0.0))}，"
            "可以作为分析 hook、drop、群舞冲击力的重点片段。"
        )
    if pre_segments and chorus_segments:
        arrangement_cues.append("检测到 pre-chorus 与 chorus/drop 的组合，这种结构常用于先削弱/悬置鼓组，再在副歌释放低频和完整鼓组。")
    if labels.get("intro", 0) and labels.get("verse", 0):
        arrangement_cues.append("intro 到 verse 的过渡可以观察人声首次进入、鼓组是否从铺底变为明确律动。")
    if not arrangement_cues:
        arrangement_cues.append("当前段落边界较粗，建议结合可视化能量曲线人工确认 intro、主歌、副歌和 bridge 的具体位置。")

    stem_cues: list[str] = []
    stem_section_cues: list[str] = []
    if stem_features:
        vocals = stem_features.get("vocals", {})
        drums = stem_features.get("drums", {})
        bass = stem_features.get("bass", {})
        other = stem_features.get("other", {})
        if drums:
            d_onset = float(drums.get("onset_density_per_sec", 0.0) or 0.0)
            d_share = float(drums.get("relative_rms_share", 0.0) or 0.0)
            if d_onset >= 2.4 or d_share >= 0.22:
                stem_cues.append("drums stem 起音密度/能量占比较高，鼓组很可能是舞蹈律动和副歌冲击力的核心来源。")
            else:
                stem_cues.append("drums stem 相对克制，歌曲可能更多依靠人声、和声铺底或合成器音色推进。")
        if bass:
            b_low = float(bass.get("low_energy_ratio_mean", 0.0) or 0.0)
            b_share = float(bass.get("relative_rms_share", 0.0) or 0.0)
            if b_low >= 0.45 or b_share >= 0.18:
                stem_cues.append("bass stem 低频集中度/能量占比较高，适合制造 drop、dark concept 或 confident 取向的重量感。")
            else:
                stem_cues.append("bass stem 不算特别突出，低频可能更多与 kick 共同铺底，而非独立形成 hook。")
        if vocals:
            v_share = float(vocals.get("relative_rms_share", 0.0) or 0.0)
            v_onset = float(vocals.get("onset_density_per_sec", 0.0) or 0.0)
            if v_share >= 0.28:
                stem_cues.append("vocals stem 能量占比较高，说明歌曲较强调人声存在感；副歌可能依赖叠唱或 ad-lib 增厚。")
            if v_onset >= 1.8:
                stem_cues.append("vocals stem 起音也较密，可能存在 rap、chant hook 或切分式人声节奏。")
        if other:
            o_high = float(other.get("high_energy_ratio_mean", 0.0) or 0.0)
            o_share = float(other.get("relative_rms_share", 0.0) or 0.0)
            if o_high >= 0.08 or o_share >= 0.35:
                stem_cues.append("other stem 高频或能量占比较明显，可能包含 synth hook、pad、FX、吉他或编曲装饰层，是 K-pop 音色识别的重要来源。")

        # If stem feature curves are available, compare average stem energy inside structural sections.
        for seg in chorus_segments[:3] + dance_segments[:2] + bridge_segments[:2]:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)
            label = str(seg.get("label_guess", "section"))
            dominant = _dominant_stem_in_range(stem_features, start, end)
            if dominant:
                stem_section_cues.append(
                    f"{human_time(start)}–{human_time(end)} 的 {label} 中，{dominant[0]} stem 相对更突出，可作为该段编曲重心的线索。"
                )
    else:
        stem_cues.append("未启用或未成功生成 stems；若要分析鼓、贝斯、人声和伴奏在副歌中的作用，建议使用 `--stems` 运行 audio-separator。")

    production_cues: list[str] = []
    if _score(tag_result, "drop chorus") >= 0.45 or _score(tag_result, "chorus energy lift") >= 0.50:
        production_cues.append("模型认为 chorus energy lift/drop chorus 倾向较强：可重点比较 pre-chorus 末尾与 chorus 开头的低频、鼓组和整体响度差异。")
    if _score(tag_result, "pre-chorus build-up") >= 0.45:
        production_cues.append("pre-chorus build-up 倾向较强：常见手法包括鼓组减法、上行旋律、riser/FX、和声逐步增厚。")
    if _score(tag_result, "dance break likely") >= 0.45:
        production_cues.append("dance break likely 分数较高：K-pop 中这类段落常减少歌词信息，转而突出鼓组、bass、synth riff 和编舞节奏点。")
    if _score(tag_result, "sparse verse") >= 0.45:
        production_cues.append("sparse verse 倾向说明主歌可能采用减法编曲，让 rap/人声节奏更清晰，并为副歌制造对比。")
    if _score(tag_result, "vocal layering") >= 0.42:
        production_cues.append("vocal layering 倾向提示副歌或桥段可能存在叠唱、和声垫底、ad-lib 或 call-and-response。")

    summary = _build_summary(tags, tempo, key, global_cues, arrangement_cues, stem_cues, tag_result)

    return {
        "top_tags": tags,
        "global_cues": global_cues,
        "tag_cues": tag_cues,
        "segment_cues": segment_cues,
        "arrangement_cues": arrangement_cues,
        "stem_cues": stem_cues,
        "stem_section_cues": stem_section_cues,
        "production_cues": production_cues,
        "summary": summary,
    }


def _dominant_stem_in_range(stem_features: dict, start: float, end: float) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for stem, info in stem_features.items():
        times = info.get("frame_times", []) or []
        curve = info.get("loudness_curve", []) or []
        if not times or not curve:
            continue
        vals = [float(v) for t, v in zip(times, curve) if start <= float(t) <= end]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if best is None or mean > best[1]:
            best = (stem, float(mean))
    return best


def _build_summary(
    tags: list[str],
    tempo: float,
    key: dict,
    global_cues: list[str],
    arrangement_cues: list[str],
    stem_cues: list[str],
    tag_result: dict,
) -> str:
    tag_text = "、".join(tags[:6]) if tags else "未形成明确标签"
    key_text = key.get("key", "unknown")
    source = tag_result.get("source", "unknown")
    pieces = [
        f"这首歌的主要标签倾向为：{tag_text}。",
        f"系统估计 BPM 约为 {tempo:.1f}，调性倾向为 {key_text}。",
    ]
    if source == "mert_kpop_classifier":
        pieces.append("标签由 MERT embedding 与训练好的 K-pop 多标签分类器给出。")
    elif "mert" in str(source):
        pieces.append("系统已抽取 MERT embedding，但当前未配置训练好的分类器权重，因此标签仍以声学先验为主。")
    else:
        pieces.append("当前标签来自声学先验分类器；配置训练好的 MERT K-pop classifier 后可获得更可靠的风格/编曲标签。")
    if global_cues:
        pieces.append(global_cues[0])
    if arrangement_cues:
        pieces.append(arrangement_cues[0])
    if stem_cues:
        pieces.append(stem_cues[0])
    pieces.append("整体解读应把标签、段落、声部分布结合起来看：标签描述风格倾向，节拍/能量/声部变化解释这种听感是如何被编曲制造出来的。")
    return "".join(pieces)
