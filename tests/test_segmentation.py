import numpy as np

from kpop_scope.audio.segmentation import segment_track


def test_segment_track_runs_on_noise():
    sr = 22050
    y = np.random.default_rng(0).normal(0, 0.01, sr * 10).astype("float32")
    result = segment_track(y, sr, min_seconds=2, max_seconds=5, max_segments=5)
    assert "segments" in result
    assert len(result["segments"]) >= 1
