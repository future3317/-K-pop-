from kpop_scope.explain.report_generator import generate_report


def minimal_features():
    return {
        "duration_seconds": 30,
        "tempo": {"bpm": 124},
        "key": {"key": "C major", "confidence": 0.4},
        "onset": {"onset_density_per_sec": 2.2},
        "loudness": {"rms_db_relative_stats": {"p10": -18.0, "p90": -6.0}},
        "spectral": {
            "centroid_hz": {"mean": 3000},
            "band_energy_ratio": {"low_20_250": {"mean": 0.2}},
        },
        "segments": {
            "segments": [
                {"start": 0, "end": 8, "label_guess": "intro", "confidence": 0.55, "evidence": ["start"]},
                {"start": 8, "end": 20, "label_guess": "chorus/drop", "confidence": 0.82, "energy_mean": 0.9},
                {"start": 20, "end": 30, "label_guess": "outro", "confidence": 0.5},
            ]
        },
    }


def minimal_tags():
    return {
        "source": "acoustic_prior_classifier_v2",
        "tags": [
            {"tag": "dance-pop", "score": 0.8},
            {"tag": "energetic", "score": 0.7},
            {"tag": "chorus energy lift", "score": 0.6},
        ],
    }


def test_tag_only_report_mode_omits_segment_table():
    md = generate_report("song.flac", minimal_features(), minimal_tags(), {"summary": "x"}, report_mode="tag_only")
    assert "Tag-only Baseline" in md
    assert "结构与段落摘要" not in md
    assert "| 标签 | full mix | vocals | drums | bass | other |" not in md


def test_readable_evidence_omits_stem_matrix_without_stems():
    md = generate_report(
        "song.flac",
        minimal_features(),
        minimal_tags(),
        {"summary": "x"},
        figure_paths={"waveform": r"figures\waveform.png"},
        report_mode="evidence_grounded",
        report_detail="readable",
    )
    assert "Evidence-grounded Analysis" in md
    assert "figures/waveform.png" in md
    assert "无法判断 vocals、drums、bass、other 的独立贡献" in md
    assert "| 标签 | full mix | vocals | drums | bass | other |" not in md


def test_technical_evidence_keeps_debug_table():
    md = generate_report(
        "song.flac",
        minimal_features(),
        minimal_tags(),
        {"summary": "x", "segment_cues": [{"time": "0:00-0:08", "label_guess": "intro", "confidence": 0.5, "reason": "start"}]},
        report_mode="evidence_grounded",
        report_detail="technical",
    )
    assert "细节级别：`technical`" in md
    assert "| 时间 | 推测段落 | 置信度 | 依据 |" in md
