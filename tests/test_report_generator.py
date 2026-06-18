from kpop_scope.explain.report_generator import generate_report


def test_generate_report_minimal():
    features = {
        "duration_seconds": 60,
        "tempo": {"bpm": 120},
        "key": {"key": "C major", "confidence": 0.5},
        "onset": {"onset_density_per_sec": 2.0},
        "spectral": {"centroid_hz": {"mean": 2000}},
        "segments": {"segments": []},
    }
    tags = {"source": "test", "tags": [{"tag": "dance-pop", "score": 0.9}]}
    exp = {"summary": "summary", "global_cues": [], "segment_cues": []}
    md = generate_report("song.mp3", features, tags, exp)
    assert "KPopScope" in md
    assert "dance-pop" in md
