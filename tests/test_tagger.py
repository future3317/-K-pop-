from kpop_scope.models.feature_vector import acoustic_feature_vector
from kpop_scope.models.tagger import predict_kpop_tags


def minimal_features():
    return {
        "tempo": {"bpm": 124, "beat_density_per_sec": 2.0},
        "onset": {"onset_density_per_sec": 2.5},
        "loudness": {
            "rms_stats": {"mean": 0.08, "std": 0.02, "p90": 0.12},
            "rms_db_relative_stats": {"p10": -24, "p90": -8},
        },
        "spectral": {
            "centroid_hz": {"mean": 3000, "std": 800},
            "bandwidth_hz": {"mean": 2500},
            "rolloff_hz": {"mean": 6000},
            "zero_crossing_rate": {"mean": 0.08},
            "band_energy_ratio": {
                "low_20_250": {"mean": 0.25},
                "lowmid_250_500": {"mean": 0.15},
                "mid_500_2000": {"mean": 0.25},
                "highmid_2000_6000": {"mean": 0.25},
                "high_6000_12000": {"mean": 0.10},
            },
        },
        "segments": {
            "segments": [
                {"label_guess": "intro", "mean_energy": 0.25, "mean_onset": 0.2},
                {"label_guess": "pre-chorus", "mean_energy": 0.45, "mean_onset": 0.4},
                {"label_guess": "chorus/drop", "mean_energy": 0.8, "mean_onset": 0.7},
            ]
        },
    }


def test_acoustic_feature_vector_shape():
    vec, names = acoustic_feature_vector(minimal_features())
    assert vec.shape[0] == len(names)
    assert vec.shape[0] > 10


def test_predict_tags_fallback_runs_without_optional_models():
    result = predict_kpop_tags("song.mp3", y=__import__("numpy").zeros(1000), sr=1000, features=minimal_features(), model_config={})
    assert result["source"] == "acoustic_prior_classifier_v2"
    assert result["tags"]
