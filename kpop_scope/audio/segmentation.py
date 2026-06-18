from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "slope": 0.0}
    if values.size >= 2:
        slope = float(np.polyfit(np.linspace(0.0, 1.0, values.size), values, 1)[0])
    else:
        slope = 0.0
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "std": float(np.std(values)),
        "slope": slope,
    }


def _safe_mean(feature_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(feature_matrix.shape[0], dtype=float)
    return np.mean(feature_matrix[:, mask], axis=1)


def _boundary_strength(times: np.ndarray, novelty: np.ndarray, t: float) -> float:
    if times.size == 0 or novelty.size == 0:
        return 0.0
    idx = int(np.argmin(np.abs(times - t)))
    return float(novelty[min(idx, len(novelty) - 1)])


def _merge_close_boundaries(boundaries: list[float], min_seconds: float, duration: float) -> list[float]:
    out = [0.0]
    for t in sorted(float(b) for b in boundaries if 0.0 < float(b) < duration):
        if t - out[-1] >= min_seconds:
            out.append(t)
    if duration - out[-1] < min_seconds and len(out) > 1:
        out[-1] = duration
    else:
        out.append(duration)
    return out


def _insert_long_segment_boundaries(boundaries: list[float], max_seconds: float, duration: float) -> list[float]:
    out = [boundaries[0]]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        while end - out[-1] > max_seconds:
            out.append(float(out[-1] + max_seconds))
        if end > out[-1]:
            out.append(float(end))
    out[-1] = duration
    return out


def _dedupe_boundary_sequence(boundaries: list[float], duration: float) -> list[float]:
    vals = sorted(set(round(float(b), 4) for b in boundaries if 0 <= b <= duration))
    if not vals or vals[0] != 0.0:
        vals = [0.0] + vals
    if vals[-1] != round(duration, 4):
        vals.append(float(duration))
    return [float(v) for v in vals]


def _make_feature_matrix(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    n = min(len(rms), len(onset), len(centroid), len(bandwidth), len(rolloff), len(zcr), chroma.shape[1], mfcc.shape[1])
    if n <= 1:
        times = librosa.frames_to_time(np.arange(max(n, 1)), sr=sr, hop_length=hop_length)
        curves = {
            "energy": np.zeros(max(n, 1)),
            "onset": np.zeros(max(n, 1)),
            "centroid": np.zeros(max(n, 1)),
            "bandwidth": np.zeros(max(n, 1)),
            "rolloff": np.zeros(max(n, 1)),
            "zcr": np.zeros(max(n, 1)),
            "chroma_stability": np.zeros(max(n, 1)),
        }
        return np.zeros((1, max(n, 1))), curves, times

    rms = rms[:n]
    onset = onset[:n]
    centroid = centroid[:n]
    bandwidth = bandwidth[:n]
    rolloff = rolloff[:n]
    zcr = zcr[:n]
    chroma = chroma[:, :n]
    mfcc = mfcc[:, :n]
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

    energy = _normalize(rms)
    onset_norm = _normalize(onset)
    centroid_norm = _normalize(centroid)
    bandwidth_norm = _normalize(bandwidth)
    rolloff_norm = _normalize(rolloff)
    zcr_norm = _normalize(zcr)
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    chroma_stability = 1.0 - _normalize(np.r_[0.0, np.linalg.norm(np.diff(chroma_norm, axis=1), axis=0)])

    mfcc_norm = np.vstack([_normalize(row) for row in mfcc])
    matrix = np.vstack(
        [
            energy * 1.4,
            onset_norm * 1.2,
            centroid_norm * 0.7,
            bandwidth_norm * 0.4,
            rolloff_norm * 0.4,
            zcr_norm * 0.3,
            chroma_norm * 0.9,
            mfcc_norm * 0.55,
        ]
    )
    curves = {
        "energy": energy,
        "onset": onset_norm,
        "centroid": centroid_norm,
        "bandwidth": bandwidth_norm,
        "rolloff": rolloff_norm,
        "zcr": zcr_norm,
        "chroma_stability": chroma_stability,
    }
    return matrix.astype(float), curves, times


def _novelty_from_features(matrix: np.ndarray, curves: dict[str, np.ndarray], sr: int, hop_length: int, smooth_seconds: float) -> np.ndarray:
    if matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[1])
    diff = np.linalg.norm(np.diff(matrix, axis=1), axis=0)
    diff = np.r_[0.0, diff]
    energy_jump = np.r_[0.0, np.abs(np.diff(curves["energy"]))]
    onset_jump = np.r_[0.0, np.abs(np.diff(curves["onset"]))]
    chroma_change = np.r_[0.0, 1.0 - curves["chroma_stability"][1:]]
    novelty = _normalize(diff) * 0.50 + _normalize(energy_jump) * 0.25 + _normalize(onset_jump) * 0.20 + _normalize(chroma_change) * 0.05
    sigma_frames = max(1, int(smooth_seconds * sr / hop_length))
    novelty = gaussian_filter1d(novelty, sigma=sigma_frames)
    return _normalize(novelty)


def _segment_similarity(means: list[np.ndarray], idx: int) -> tuple[float, int | None]:
    if idx >= len(means) or len(means) <= 1:
        return 0.0, None
    target = means[idx]
    best_score = -1.0
    best_j: int | None = None
    for j, vec in enumerate(means):
        if j == idx:
            continue
        denom = float(np.linalg.norm(target) * np.linalg.norm(vec) + 1e-8)
        score = float(np.dot(target, vec) / denom)
        if score > best_score:
            best_score = score
            best_j = j
    return max(0.0, best_score), best_j


def _infer_labels(segments: list[dict], duration: float) -> list[dict]:
    if not segments:
        return segments
    n = len(segments)
    energies = np.array([s["mean_energy"] for s in segments], dtype=float)
    onsets = np.array([s["mean_onset"] for s in segments], dtype=float)
    centroids = np.array([s["mean_centroid"] for s in segments], dtype=float)
    slopes = np.array([s["energy_slope"] for s in segments], dtype=float)
    mean_e = float(np.mean(energies))
    mean_o = float(np.mean(onsets))
    high_e = float(np.percentile(energies, 70)) if n > 1 else mean_e
    low_e = float(np.percentile(energies, 35)) if n > 1 else mean_e
    high_o = float(np.percentile(onsets, 70)) if n > 1 else mean_o
    high_c = float(np.percentile(centroids, 65)) if n > 1 else float(np.mean(centroids))

    # First pass: loud/repeated sections are likely chorus/drop; short dense post-chorus sections can be dance breaks.
    chorus_candidates = set()
    for i, s in enumerate(segments):
        strong = s["mean_energy"] >= high_e and s["mean_onset"] >= max(mean_o, 0.35)
        repeated = s.get("repetition_score", 0.0) >= 0.88 and s["mean_energy"] >= mean_e
        if strong or repeated:
            chorus_candidates.add(i)
    if not chorus_candidates:
        chorus_candidates.add(int(np.argmax(energies)))

    labels = ["unknown"] * n
    confidences = [0.45] * n
    evidence: list[list[str]] = [[] for _ in range(n)]

    for i, s in enumerate(segments):
        start = s["start"]
        end = s["end"]
        dur = s["duration"]
        pos = (start + end) / max(duration * 2.0, 1e-8)
        e = s["mean_energy"]
        o = s["mean_onset"]
        c = s["mean_centroid"]
        slope = s["energy_slope"]
        rep = s.get("repetition_score", 0.0)

        if i == 0 and (e <= mean_e * 0.95 or dur <= 18 or o <= mean_o):
            labels[i] = "intro"
            confidences[i] = 0.72
            evidence[i].append("开头段且能量/节奏较克制")
            continue
        if i == n - 1 and (e <= mean_e * 0.90 or slope < -0.05):
            labels[i] = "outro"
            confidences[i] = 0.68
            evidence[i].append("末段能量较低或呈收束趋势")
            continue
        if i in chorus_candidates:
            labels[i] = "chorus/drop"
            confidences[i] = 0.70 + 0.12 * (rep >= 0.88)
            evidence[i].append("段落能量较高且节奏活跃")
            if rep >= 0.88:
                evidence[i].append("与其他高能段相似，可能是重复副歌")
            continue
        if e >= high_e and o >= high_o and c >= high_c and dur <= 24 and i > 0:
            labels[i] = "dance break"
            confidences[i] = 0.66
            evidence[i].append("短时高能、瞬态密集且频谱更亮")
            continue
        if i > 0 and i + 1 < n and (i + 1 in chorus_candidates) and (slope > 0.06 or segments[i + 1]["mean_energy"] - e > 0.12):
            labels[i] = "pre-chorus"
            confidences[i] = 0.72
            evidence[i].append("位于高能段之前并呈明显蓄力/上升")
            continue
        if e <= low_e and o <= max(mean_o, high_o) and pos > 0.45 and rep < 0.82:
            labels[i] = "bridge"
            confidences[i] = 0.58
            evidence[i].append("中后段能量回落且与其他段落重复性较低")
            continue
        if e <= mean_e and o <= high_o:
            labels[i] = "verse"
            confidences[i] = 0.58
            evidence[i].append("能量和节奏密度低于副歌候选段")
        else:
            labels[i] = "transition"
            confidences[i] = 0.52
            evidence[i].append("能量或节奏介于主歌与副歌之间")

    # Second pass: a dense segment immediately after chorus is often post-chorus/dance break in K-pop.
    for i in range(1, n):
        if labels[i - 1] == "chorus/drop" and labels[i] in {"transition", "unknown"}:
            if onsets[i] >= high_o and energies[i] >= mean_e and segments[i]["duration"] <= 24:
                labels[i] = "post-chorus" if segments[i].get("mean_energy", 0.0) >= mean_e else "dance break"
                confidences[i] = max(confidences[i], 0.62)
                evidence[i].append("紧跟副歌后的高密度器乐推进，符合 dance break/post-chorus 特征")

    for i, s in enumerate(segments):
        prev_energy = float(segments[i - 1].get("mean_energy", 0.0) or 0.0) if i > 0 else float(s.get("mean_energy", 0.0) or 0.0)
        boundary_strength = max(
            float(s.get("boundary_strength_start", 0.0) or 0.0),
            float(s.get("boundary_strength_end", 0.0) or 0.0),
        )
        s["label_guess"] = labels[i]
        s["label"] = labels[i]
        s["label_confidence"] = float(min(0.95, confidences[i]))
        if not evidence[i]:
            evidence[i].append("根据相对能量、起音密度、频谱亮度和位置综合推断")
        s["label_evidence"] = evidence[i]
        s["evidence"] = evidence[i]
        s["boundary_strength"] = float(boundary_strength)
        s["energy_mean"] = float(s.get("mean_energy", 0.0) or 0.0)
        s["energy_delta"] = float((s.get("mean_energy", 0.0) or 0.0) - prev_energy)
        s["onset_density"] = float(s.get("mean_onset", 0.0) or 0.0)
        s["spectral_brightness"] = float(s.get("mean_centroid", 0.0) or 0.0)
    return segments


def _unique_evidence(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in out:
            out.append(text)
    return out


def _merge_adjacent_same_label(segments: list[dict], max_gap: float = 2.0) -> list[dict]:
    if not segments:
        return segments
    merged: list[dict] = []
    for seg in segments:
        label = str(seg.get("label", seg.get("label_guess", "unknown")))
        if merged:
            prev = merged[-1]
            prev_label = str(prev.get("label", prev.get("label_guess", "unknown")))
            gap = float(seg.get("start", 0.0) or 0.0) - float(prev.get("end", 0.0) or 0.0)
            if label == prev_label and gap <= max_gap:
                prev_dur = max(float(prev.get("duration", 0.0) or 0.0), 1e-8)
                seg_dur = max(float(seg.get("duration", 0.0) or 0.0), 1e-8)
                total = prev_dur + seg_dur
                for key in ["mean_energy", "energy_mean", "mean_onset", "onset_density", "mean_centroid", "spectral_brightness", "chroma_stability", "repetition_score", "label_confidence"]:
                    prev[key] = float((float(prev.get(key, 0.0) or 0.0) * prev_dur + float(seg.get(key, 0.0) or 0.0) * seg_dur) / total)
                prev["end"] = float(seg.get("end", prev.get("end", 0.0)) or 0.0)
                prev["duration"] = float(prev["end"] - float(prev.get("start", 0.0) or 0.0))
                prev["boundary_strength_end"] = float(seg.get("boundary_strength_end", prev.get("boundary_strength_end", 0.0)) or 0.0)
                prev["boundary_strength"] = max(float(prev.get("boundary_strength", 0.0) or 0.0), float(seg.get("boundary_strength", 0.0) or 0.0))
                ev = list(prev.get("evidence", prev.get("label_evidence", [])) or []) + list(seg.get("evidence", seg.get("label_evidence", [])) or [])
                prev["evidence"] = _unique_evidence(ev)[:4]
                prev["label_evidence"] = prev["evidence"]
                continue
        merged.append(dict(seg))
    for i, seg in enumerate(merged):
        seg["index"] = int(i)
    return merged


def _postprocess_chorus_candidates(segments: list[dict]) -> list[dict]:
    if not segments:
        return segments
    energies = np.array([float(s.get("energy_mean", s.get("mean_energy", 0.0)) or 0.0) for s in segments], dtype=float)
    onsets = np.array([float(s.get("onset_density", s.get("mean_onset", 0.0)) or 0.0) for s in segments], dtype=float)
    high_e = float(np.percentile(energies, 65)) if energies.size else 0.0
    high_o = float(np.percentile(onsets, 60)) if onsets.size else 0.0
    candidates: list[tuple[float, int]] = []
    for i, seg in enumerate(segments):
        if "chorus" not in str(seg.get("label", seg.get("label_guess", ""))):
            continue
        prev = segments[i - 1] if i > 0 else {}
        e = float(seg.get("energy_mean", seg.get("mean_energy", 0.0)) or 0.0)
        o = float(seg.get("onset_density", seg.get("mean_onset", 0.0)) or 0.0)
        rep = float(seg.get("repetition_score", 0.0) or 0.0)
        energy_delta = e - float(prev.get("energy_mean", prev.get("mean_energy", e)) or e)
        onset_delta = o - float(prev.get("onset_density", prev.get("mean_onset", o)) or o)
        structurally_plausible = i > 0 and str(prev.get("label", prev.get("label_guess", ""))) in {"intro", "verse", "pre-chorus", "transition", "high-energy section", "refrain"}
        repeated_later = sum(1 for s in segments if float(s.get("repetition_score", 0.0) or 0.0) >= 0.82 and float(s.get("energy_mean", s.get("mean_energy", 0.0)) or 0.0) >= high_e) >= 2
        strict = e >= high_e and (rep >= 0.82 or energy_delta >= 0.08 or onset_delta >= 0.08) and (structurally_plausible or repeated_later)
        score = e * 0.45 + rep * 0.30 + max(0.0, energy_delta) * 0.15 + max(0.0, onset_delta) * 0.10
        if strict:
            candidates.append((float(score), i))
        else:
            seg["label"] = seg["label_guess"] = "high-energy section"
            seg["label_confidence"] = min(float(seg.get("label_confidence", 0.5) or 0.5), 0.56)
            seg["evidence"] = ["能量或重复性较高，但缺少足够的能量抬升/结构位置证据，暂不确认具体功能。"]
            seg["label_evidence"] = seg["evidence"]

    keep = {i for _, i in sorted(candidates, reverse=True)[:3]}
    for _, i in candidates:
        if i in keep:
            continue
        seg = segments[i]
        rep = float(seg.get("repetition_score", 0.0) or 0.0)
        seg["label"] = seg["label_guess"] = "refrain" if rep >= 0.82 else "high-energy section"
        seg["label_confidence"] = min(float(seg.get("label_confidence", 0.5) or 0.5), 0.60)
        seg["evidence"] = ["与主要 chorus/drop 候选相似，但为避免过度标注，归为重复高能段。"]
        seg["label_evidence"] = seg["evidence"]

    for i, seg in enumerate(segments):
        prev_e = float(segments[i - 1].get("energy_mean", segments[i - 1].get("mean_energy", 0.0)) or 0.0) if i > 0 else float(seg.get("energy_mean", seg.get("mean_energy", 0.0)) or 0.0)
        prev_o = float(segments[i - 1].get("onset_density", segments[i - 1].get("mean_onset", 0.0)) or 0.0) if i > 0 else float(seg.get("onset_density", seg.get("mean_onset", 0.0)) or 0.0)
        seg["energy_delta"] = float(float(seg.get("energy_mean", seg.get("mean_energy", 0.0)) or 0.0) - prev_e)
        seg["onset_density_delta"] = float(float(seg.get("onset_density", seg.get("mean_onset", 0.0)) or 0.0) - prev_o)
    return segments


def _postprocess_segments(segments: list[dict]) -> list[dict]:
    segments = _merge_adjacent_same_label(segments, max_gap=2.0)
    segments = _postprocess_chorus_candidates(segments)
    for i, seg in enumerate(segments):
        seg["index"] = int(i)
    return segments


def _structure_summary(segments: list[dict]) -> dict:
    labels = [str(s.get("label", s.get("label_guess", "unknown"))) for s in segments]
    chorus = [
        {
            "index": int(s.get("index", i)),
            "start": float(s.get("start", 0.0) or 0.0),
            "end": float(s.get("end", 0.0) or 0.0),
            "confidence": float(s.get("label_confidence", 0.0) or 0.0),
        }
        for i, s in enumerate(segments)
        if "chorus" in str(s.get("label", s.get("label_guess", "")))
    ]
    dance = [
        {
            "index": int(s.get("index", i)),
            "start": float(s.get("start", 0.0) or 0.0),
            "end": float(s.get("end", 0.0) or 0.0),
            "confidence": float(s.get("label_confidence", 0.0) or 0.0),
        }
        for i, s in enumerate(segments)
        if "dance break" in str(s.get("label", s.get("label_guess", "")))
    ]
    confidences = [float(s.get("label_confidence", 0.0) or 0.0) for s in segments]
    return {
        "num_segments": int(len(segments)),
        "has_intro": "intro" in labels,
        "has_prechorus": "pre-chorus" in labels,
        "has_chorus_drop": any("chorus" in label for label in labels),
        "has_dance_break": any("dance break" in label for label in labels),
        "chorus_candidates": chorus,
        "dance_break_candidates": dance,
        "structure_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def segment_track(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_seconds: float = 8.0,
    max_seconds: float = 32.0,
    smooth_seconds: float = 1.0,
    max_segments: int = 12,
) -> dict:
    """Estimate K-pop-oriented coarse structure from multi-feature novelty.

    The algorithm remains lightweight and explainable:
    1. compute RMS, onset, spectral brightness, chroma, and MFCC features;
    2. detect boundary candidates from smoothed multi-feature novelty;
    3. enforce min/max segment duration constraints;
    4. label sections with K-pop-aware rules for intro/verse/pre-chorus/
       chorus/drop/bridge/dance break/outro.

    The labels are hypotheses for a report, not ground-truth annotations.
    """
    duration = float(len(y) / sr)
    if duration <= 0:
        return {"boundaries": [0.0], "segments": [], "novelty_curve": [], "frame_times": [], "method": "kpop_multifeature_v2"}

    matrix, curves, times = _make_feature_matrix(y, sr, hop_length)
    novelty_norm = _novelty_from_features(matrix, curves, sr, hop_length, smooth_seconds)

    min_distance_frames = max(1, int(min_seconds * sr / hop_length))
    prominence = 0.06 if duration < 80 else 0.08
    peaks, _ = find_peaks(novelty_norm, distance=min_distance_frames, prominence=prominence)
    candidate_times = [float(times[p]) for p in peaks if min_seconds <= times[p] <= duration - min_seconds]

    # Add soft musical-grid candidates based on estimated tempo. They are kept only
    # when they land near some novelty, helping common 8/16-bar K-pop sections.
    try:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=curves["onset"], sr=sr, hop_length=hop_length, units="frames")
        if hasattr(tempo, "item"):
            tempo = float(tempo.item())
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        if tempo and len(beat_times) >= 16:
            bar_seconds = 60.0 / tempo * 4.0
            for bars in (8, 16):
                step = bar_seconds * bars
                t = step
                while min_seconds <= t <= duration - min_seconds:
                    strength = _boundary_strength(times, novelty_norm, t)
                    if strength >= 0.35:
                        candidate_times.append(float(t))
                    t += step
    except Exception:
        pass

    boundaries = _merge_close_boundaries([0.0] + candidate_times + [duration], min_seconds=min_seconds, duration=duration)
    boundaries = _insert_long_segment_boundaries(boundaries, max_seconds=max_seconds, duration=duration)
    boundaries = _dedupe_boundary_sequence(boundaries, duration=duration)

    # Limit segment count by keeping strongest internal boundaries, but preserve start/end.
    if len(boundaries) - 1 > max_segments:
        internal = boundaries[1:-1]
        strengths = [_boundary_strength(times, novelty_norm, t) for t in internal]
        keep_n = max(1, max_segments - 1)
        keep = sorted([b for _, b in sorted(zip(strengths, internal), reverse=True)[:keep_n]])
        boundaries = _dedupe_boundary_sequence([0.0] + keep + [duration], duration=duration)

    means: list[np.ndarray] = []
    segments: list[dict] = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (times >= start) & (times < end)
        if i == len(boundaries) - 2:
            mask = (times >= start) & (times <= end)
        e_stats = _summary(curves["energy"][mask])
        o_stats = _summary(curves["onset"][mask])
        c_stats = _summary(curves["centroid"][mask])
        z_stats = _summary(curves["zcr"][mask])
        chroma_stability = _summary(curves["chroma_stability"][mask])
        mean_vec = _safe_mean(matrix, mask)
        means.append(mean_vec)
        segments.append(
            {
                "index": int(i),
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "label_guess": "unknown",
                "mean_energy": e_stats["mean"],
                "max_energy": e_stats["max"],
                "min_energy": e_stats["min"],
                "energy_lift": float(e_stats["max"] - e_stats["min"]),
                "energy_slope": e_stats["slope"],
                "mean_onset": o_stats["mean"],
                "max_onset": o_stats["max"],
                "onset_slope": o_stats["slope"],
                "mean_centroid": c_stats["mean"],
                "spectral_brightness": c_stats["mean"],
                "mean_zcr": z_stats["mean"],
                "chroma_stability": chroma_stability["mean"],
                "boundary_strength_start": _boundary_strength(times, novelty_norm, start),
                "boundary_strength_end": _boundary_strength(times, novelty_norm, end),
            }
        )

    for i, seg in enumerate(segments):
        rep_score, rep_idx = _segment_similarity(means, i)
        seg["repetition_score"] = float(rep_score)
        seg["nearest_repeated_segment"] = None if rep_idx is None else int(rep_idx)

    segments = _postprocess_segments(_infer_labels(segments, duration))

    return {
        "boundaries": [float(b) for b in boundaries],
        "segments": segments,
        "structure_summary": _structure_summary(segments),
        "novelty_curve": [float(v) for v in novelty_norm],
        "frame_times": [float(v) for v in times],
        "method": "kpop_multifeature_structure_v2",
        "notes": [
            "Labels are rule-based hypotheses for explainable K-pop arrangement analysis.",
            "Boundary detection uses RMS, onset, spectral, chroma and MFCC novelty with duration constraints.",
        ],
    }
