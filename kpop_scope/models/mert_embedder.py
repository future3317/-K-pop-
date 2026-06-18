from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


class MERTEmbedder:
    """MERT feature extractor using Hugging Face Transformers.

    The default checkpoint is ``m-a-p/MERT-v1-95M``. The Hugging Face model card
    shows direct Transformers loading with ``AutoModel.from_pretrained(...,
    trust_remote_code=True)``; this wrapper keeps that optional dependency out of
    the core install and returns a single pooled numpy embedding.
    """

    target_sample_rate = 24000

    def __init__(
        self,
        model_name_or_path: str = "m-a-p/MERT-v1-95M",
        device: str = "auto",
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install optional deps with `pip install -e .[models]`.") from exc

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name_or_path = str(model_name_or_path)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

        kwargs = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs).to(device)
        self.model.eval()

    def embed_file(
        self,
        path: str | Path,
        max_duration: float | None = 30.0,
        chunk_seconds: float | None = 20.0,
    ) -> np.ndarray:
        y, sr = librosa.load(path, sr=self.target_sample_rate, mono=True, duration=max_duration)
        return self.embed_audio(y, sr=sr, chunk_seconds=chunk_seconds)

    def embed_audio(
        self,
        y: np.ndarray,
        sr: int = 24000,
        chunk_seconds: float | None = 20.0,
    ) -> np.ndarray:
        """Return mean-pooled MERT embedding.

        Long tracks can exceed GPU memory. When ``chunk_seconds`` is set, we run
        MERT on chunks and average chunk embeddings weighted by chunk length.
        """
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            raise ValueError("Cannot embed empty audio.")
        if sr != self.target_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate

        if chunk_seconds is None or chunk_seconds <= 0:
            return self._embed_chunk(y, sr)

        chunk_size = int(chunk_seconds * sr)
        if y.size <= chunk_size:
            return self._embed_chunk(y, sr)

        embs: list[np.ndarray] = []
        weights: list[float] = []
        hop = chunk_size
        for start in range(0, len(y), hop):
            chunk = y[start : start + chunk_size]
            if chunk.size < sr * 2 and embs:
                break
            embs.append(self._embed_chunk(chunk, sr))
            weights.append(float(chunk.size))
        W = np.asarray(weights, dtype=np.float32)
        E = np.stack(embs).astype(np.float32)
        return np.average(E, axis=0, weights=W).astype(np.float32)

    def _embed_chunk(self, y: np.ndarray, sr: int) -> np.ndarray:
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        hidden = out.last_hidden_state.squeeze(0)
        emb = hidden.mean(dim=0).detach().cpu().numpy().astype("float32")
        return emb
