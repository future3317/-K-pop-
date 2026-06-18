from kpop_scope.explain.stem_contribution import build_stem_contribution


def test_stem_contribution_fallback_without_stems():
    tags = {"tags": [{"tag": "dance break", "score": 0.8}, {"tag": "vocal layering", "score": 0.7}]}
    segments = {"segments": [{"label": "dance break", "energy_mean": 0.8, "onset_density": 0.9}]}
    result = build_stem_contribution(tags, stem_features=None, segment_features=segments)
    assert result["mode"] == "rule_based_acoustic_fallback"
    assert result["tag_contributions"]
    assert result["tag_contributions"][0]["contribution"]["full_mix"] > 0
