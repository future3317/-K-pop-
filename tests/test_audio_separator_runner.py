from kpop_scope.stems.audio_separator_runner import _normalized_stem_name


def test_audio_separator_stem_name_normalization():
    assert _normalized_stem_name("song_(Vocals)_model.wav") == "vocals"
    assert _normalized_stem_name("song_(Drums)_model.wav") == "drums"
    assert _normalized_stem_name("song_(Bass)_model.wav") == "bass"
    assert _normalized_stem_name("song_(Other)_model.wav") == "other"
    assert _normalized_stem_name("song_(Instrumental)_model.wav") == "other"
