from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .feature_vector import acoustic_feature_vector
from .kpop_classifier import AcousticPriorKPopClassifier, KPopClassifier
from .mert_embedder import MERTEmbedder


def _checkpoint_needs_mert(classifier: KPopClassifier, acoustic_dim: int) -> bool:
    if classifier.input_mode in {"mert", "fusion"}:
        return True
    if classifier.input_mode == "acoustic":
        return False
    # Auto mode: if acoustic vector alone matches, MERT is not required.
    return classifier.input_dim != acoustic_dim


def _extract_mert_embedding(
    y: np.ndarray,
    sr: int,
    model_config: dict[str, Any],
) -> tuple[np.ndarray | None, str | None]:
    try:
        embedder = MERTEmbedder(
            model_name_or_path=str(model_config.get("mert_model_name_or_path", "m-a-p/MERT-v1-95M")),
            device=str(model_config.get("mert_device", "auto")),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
            dtype=str(model_config.get("mert_dtype", "auto")),
        )
        max_duration = model_config.get("mert_max_duration", 45.0)
        if max_duration is not None:
            max_samples = int(float(max_duration) * sr)
            y_for_mert = y[:max_samples]
        else:
            y_for_mert = y
        embedding = embedder.embed_audio(
            y_for_mert,
            sr=sr,
            chunk_seconds=model_config.get("mert_chunk_seconds", 20.0),
        )
        return embedding, None
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        return None, f"{type(exc).__name__}: {exc}"


def predict_kpop_tags(
    audio_path: str | Path,
    y: np.ndarray,
    sr: int,
    features: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Predict K-pop tags using MERT embeddings and a lightweight classifier.

    Recommended production path:
        1. extract MERT embeddings with ``scripts/extract_embeddings.py``;
        2. train a checkpoint with ``scripts/train_classifier.py``;
        3. set ``models.kpop_classifier_path`` in config.

    Without a checkpoint, the function returns an acoustic-prior classifier result
    unless ``fallback_to_acoustic_prior`` is false.
    """
    del audio_path  # reserved for future file-level model APIs
    checkpoint = model_config.get("kpop_classifier_path")
    top_k = int(model_config.get("top_k", 12))
    threshold = float(model_config.get("threshold", 0.20))
    fallback = bool(model_config.get("fallback_to_acoustic_prior", True))
    explicit_use_mert = bool(model_config.get("use_mert", False))
    acoustic, acoustic_names = acoustic_feature_vector(features)

    classifier: KPopClassifier | None = None
    classifier_error: str | None = None
    if checkpoint:
        try:
            classifier = KPopClassifier(checkpoint, device=str(model_config.get("classifier_device", "auto")))
        except Exception as exc:
            classifier_error = f"{type(exc).__name__}: {exc}"
            if not fallback:
                raise

    needs_mert = explicit_use_mert
    if classifier is not None:
        needs_mert = needs_mert or _checkpoint_needs_mert(classifier, acoustic_dim=acoustic.size)

    embedding: np.ndarray | None = None
    embedder_error: str | None = None
    if needs_mert:
        embedding, embedder_error = _extract_mert_embedding(y, sr, model_config)
        if embedder_error and classifier is not None and _checkpoint_needs_mert(classifier, acoustic_dim=acoustic.size) and not fallback:
            raise RuntimeError(embedder_error)

    if classifier is not None:
        try:
            result = classifier.predict(embedding=embedding, acoustic=acoustic, top_k=top_k, threshold=threshold)
            result["mert_embedding_available"] = embedding is not None
            result["acoustic_feature_names"] = acoustic_names
            if embedder_error:
                result["mert_error"] = embedder_error
            return result
        except Exception as exc:
            classifier_error = f"{type(exc).__name__}: {exc}"
            if not fallback:
                raise

    if not fallback:
        raise RuntimeError(
            "No usable K-pop classifier checkpoint configured and fallback_to_acoustic_prior=false. "
            "Set models.kpop_classifier_path or enable fallback."
        )

    result = AcousticPriorKPopClassifier().predict(features, top_k=top_k, threshold=threshold)
    result["mert_embedding_available"] = embedding is not None
    result["acoustic_feature_names"] = acoustic_names
    if embedding is not None:
        result["source"] = "mert_embedding_plus_acoustic_prior"
        result["embedding_dim"] = int(embedding.size)
        result["note"] = "MERT embedding was extracted, but no usable trained classifier checkpoint was configured."
    if embedder_error:
        result["mert_error"] = embedder_error
    if classifier_error:
        result["classifier_error"] = classifier_error
    return result
