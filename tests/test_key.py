import numpy as np

from kpop_scope.audio.key import estimate_key


def test_estimate_key_shape():
    chroma = np.zeros(12)
    chroma[0] = 1.0
    result = estimate_key(chroma)
    assert "key" in result
    assert "confidence" in result
