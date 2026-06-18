from kpop_scope.explain.report_generator import generate_report


def minimal_features():
    return {
        "duration_seconds": 30,
        "tempo": {"bpm": 124},
        "key": {"key": "C major", "confidence": 0.4},
        "onset": {"onset_density_per_sec": 2.2},
        "loudness": {"rms_stats": {"p90": 0.1}},
        "spectral": {"centroid_hz": {"mean": 3000}},
        "segments": {"segments": []},
    }


def test_tag_only_report_mode_omits_segment_table():
    tags = {"source": "acoustic_prior_classifier_v2", "tags": [{"tag": "dance-pop", "score": 0.8}]}
    md = generate_report("song.flac", minimal_features(), tags, {"summary": "x"}, report_mode="tag_only")
    assert "Tag-only baseline" in md
    assert "段落与编曲推进" not in md
