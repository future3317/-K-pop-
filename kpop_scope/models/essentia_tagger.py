from __future__ import annotations


class EssentiaTagger:
    """Thin placeholder for optional Essentia-based model inference.

    The main v2 pipeline uses ``models.tagger.predict_kpop_tags``. This class is
    kept as an extension point for users who want to wire local Essentia ONNX or
    TensorFlow models into the same tag-result dictionary format.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            import essentia  # noqa: F401
            import essentia.standard as es  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Essentia is not installed or failed to import. Install Essentia and configure "
                "a model path, or use the default KPopScope MERT/acoustic tagger."
            ) from exc

    def predict(self, audio_path: str) -> dict:
        raise NotImplementedError(
            "Connect your local Essentia model here. The rest of KPopScope will accept "
            "a dict with {'source': ..., 'tags': [{'tag': str, 'score': float}, ...]}."
        )
