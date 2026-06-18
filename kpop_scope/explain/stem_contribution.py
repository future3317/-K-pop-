from __future__ import annotations

from typing import Any


STEMS = ["vocals", "drums", "bass", "other"]


def _score_tag_scores(tag_result: dict[str, Any]) -> dict[str, float]:
    return {str(item.get("tag")): float(item.get("score", 0.0) or 0.0) for item in tag_result.get("tags", [])}


def _stem_share(stem_features: dict[str, Any] | None, stem: str, key: str = "relative_rms_share") -> float:
    if not stem_features:
        return 0.0
    try:
        return float(stem_features.get(stem, {}).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _segment_signal(segments: list[dict[str, Any]]) -> dict[str, float]:
    chorus = [s for s in segments if "chorus" in str(s.get("label", s.get("label_guess", "")))]
    dance = [s for s in segments if "dance break" in str(s.get("label", s.get("label_guess", "")))]
    pre = [s for s in segments if "pre-chorus" in str(s.get("label", s.get("label_guess", "")))]
    max_energy = max([float(s.get("energy_mean", s.get("mean_energy", 0.0)) or 0.0) for s in segments] or [0.0])
    max_onset = max([float(s.get("onset_density", s.get("mean_onset", 0.0)) or 0.0) for s in segments] or [0.0])
    return {
        "chorus": min(1.0, len(chorus) / 2.0),
        "dance": min(1.0, len(dance)),
        "pre": min(1.0, len(pre)),
        "energy": max_energy,
        "onset": max_onset,
    }


def acoustic_evidence_contribution(
    tag_result: dict[str, Any],
    stem_features: dict[str, Any] | None = None,
    segment_features: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate tag evidence contribution without a learned classifier.

    This is a rule-based fallback. Values are normalized evidence weights, not
    causal model attributions.
    """
    if isinstance(segment_features, dict):
        segments = list(segment_features.get("segments", []) or [])
    else:
        segments = list(segment_features or [])
    seg_signal = _segment_signal(segments)
    tag_scores = _score_tag_scores(tag_result)

    vocals = _stem_share(stem_features, "vocals")
    drums = _stem_share(stem_features, "drums")
    bass = _stem_share(stem_features, "bass")
    other = _stem_share(stem_features, "other")
    has_stems = bool(stem_features)

    rows: list[dict[str, Any]] = []
    for tag, score in tag_scores.items():
        t = tag.lower()
        c = {
            "full_mix": float(score),
            "vocals": 0.0,
            "drums": 0.0,
            "bass": 0.0,
            "other": 0.0,
            "rhythm": 0.0,
            "structure": 0.0,
        }
        evidence: list[str] = []
        if any(k in t for k in ["vocal", "chant", "rap", "r&b"]):
            c["vocals"] = vocals if has_stems else 0.35 * score
            evidence.append("人声相关标签主要参考 vocals stem 或人声节奏线索")
        if any(k in t for k in ["drum", "dance", "four-on", "jersey", "syncopated", "energetic"]):
            c["drums"] = drums if has_stems else 0.30 * score
            c["rhythm"] = max(c["rhythm"], seg_signal["onset"])
            evidence.append("律动/舞蹈标签参考起音密度和鼓组能量")
        if any(k in t for k in ["bass", "low", "drop", "trap", "confident", "dark"]):
            c["bass"] = bass if has_stems else 0.25 * score
            evidence.append("低频相关标签参考 bass/low-frequency 能量")
        if any(k in t for k in ["synth", "electro", "bright", "retro", "tropical", "post-chorus"]):
            c["other"] = other if has_stems else 0.25 * score
            evidence.append("合成器/伴奏标签参考 other stem、高频亮度或频谱质心")
        if any(k in t for k in ["chorus", "pre-chorus", "bridge", "intro", "outro", "structure"]):
            c["structure"] = max(seg_signal["chorus"], seg_signal["pre"], seg_signal["dance"], seg_signal["energy"])
            evidence.append("结构类标签参考自动分段、能量抬升和重复性")
        if not evidence:
            c["rhythm"] = 0.25 * score
            c["structure"] = 0.25 * score
            evidence.append("该标签由整曲声学先验给出，缺少更细粒度声部证据")

        rows.append(
            {
                "tag": tag,
                "score": float(score),
                "contribution": {k: float(max(0.0, min(1.0, v))) for k, v in c.items()},
                "evidence": evidence,
            }
        )

    return {
        "mode": "rule_based_acoustic_fallback",
        "has_stems": bool(stem_features),
        "note": "Rule-based evidence weights for interpretation; not causal model attribution.",
        "tag_contributions": rows,
    }


def perturbation_contribution(
    labels: list[str],
    full_scores: dict[str, float],
    removed_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows = []
    for tag in labels:
        full = float(full_scores.get(tag, 0.0))
        c = {"full_mix": full, "rhythm": 0.0, "structure": 0.0}
        for stem in STEMS:
            c[stem] = float(full - removed_scores.get(stem, {}).get(tag, full))
        rows.append(
            {
                "tag": tag,
                "score": full,
                "contribution": c,
                "evidence": ["基于移除声部后的分数变化计算，contribution = full score - removed score"],
            }
        )
    return {
        "mode": "model_based_perturbation",
        "has_stems": True,
        "tag_contributions": rows,
    }


def build_stem_contribution(
    tag_result: dict[str, Any],
    stem_features: dict[str, Any] | None = None,
    segment_features: dict[str, Any] | list[dict[str, Any]] | None = None,
    perturbation_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if perturbation_scores:
        labels = list(perturbation_scores.get("labels", []))
        return perturbation_contribution(
            labels=labels,
            full_scores=dict(perturbation_scores.get("full_scores", {})),
            removed_scores=dict(perturbation_scores.get("removed_scores", {})),
        )
    return acoustic_evidence_contribution(tag_result, stem_features, segment_features)
