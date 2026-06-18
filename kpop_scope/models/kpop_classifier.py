from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .feature_vector import acoustic_feature_vector


DEFAULT_KPOP_TAGS = [
    "k-pop dance",
    "dance-pop",
    "electropop",
    "hip-hop pop",
    "r&b pop",
    "ballad",
    "retro synth",
    "trap-pop",
    "jersey club influence",
    "bright",
    "dark",
    "dreamy",
    "sentimental",
    "confident",
    "cute",
    "energetic",
    "dramatic",
    "heavy drums",
    "synth bass",
    "low-frequency drive",
    "bright high-frequency synth",
    "vocal layering",
    "rap section likely",
    "drop chorus",
    "chorus energy lift",
    "pre-chorus build-up",
    "dance break likely",
    "sparse verse",
]


class KPopClassifier:
    """A small multi-label classifier wrapper.

    The checkpoint is created by ``scripts/train_classifier.py``. It may consume
    one of three input layouts:

    - MERT embedding only
    - acoustic feature vector only
    - concatenated ``[MERT embedding, acoustic feature vector]``

    The wrapper infers the right layout from checkpoint metadata and available
    inputs, so older checkpoints remain usable.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install optional deps with `pip install -e .[models]`.") from exc

        self.torch = torch
        self.nn = nn
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.labels: list[str] = list(payload["labels"])
        self.input_dim = int(payload["input_dim"])
        self.embedding_dim = payload.get("embedding_dim")
        self.acoustic_dim = payload.get("acoustic_dim")
        self.input_mode = str(payload.get("input_mode", "auto"))
        hidden_dim = int(payload.get("hidden_dim", 256))
        dropout = float(payload.get("dropout", 0.2))
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.labels)),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _prepare_input(self, embedding: np.ndarray | None, acoustic: np.ndarray | None) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.input_mode == "mert":
            if embedding is None:
                raise ValueError("This classifier expects a MERT embedding, but embedding=None.")
            parts = [embedding]
        elif self.input_mode == "acoustic":
            if acoustic is None:
                raise ValueError("This classifier expects acoustic features, but acoustic=None.")
            parts = [acoustic]
        elif self.input_mode == "fusion":
            if embedding is None or acoustic is None:
                raise ValueError("This classifier expects both MERT embedding and acoustic features.")
            parts = [embedding, acoustic]
        else:
            candidates: list[np.ndarray] = []
            if embedding is not None:
                candidates.append(embedding)
            if acoustic is not None:
                candidates.append(acoustic)
            if embedding is not None and acoustic is not None:
                candidates.insert(0, np.concatenate([embedding, acoustic]).astype(np.float32))
            for c in candidates:
                if c.size == self.input_dim:
                    return c.astype(np.float32)
            raise ValueError(
                f"Cannot match classifier input_dim={self.input_dim}; "
                f"embedding_dim={None if embedding is None else embedding.size}, "
                f"acoustic_dim={None if acoustic is None else acoustic.size}."
            )
        x = np.concatenate(parts).astype(np.float32) if len(parts) > 1 else parts[0].astype(np.float32)
        if x.size != self.input_dim:
            raise ValueError(f"Classifier expects input_dim={self.input_dim}, got {x.size}.")
        return x

    def predict_proba(self, embedding: np.ndarray | None = None, acoustic: np.ndarray | None = None) -> np.ndarray:
        x_np = self._prepare_input(embedding, acoustic)
        x = self.torch.tensor(x_np, dtype=self.torch.float32).reshape(1, -1).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        return probs

    def predict(
        self,
        embedding: np.ndarray | None = None,
        acoustic: np.ndarray | None = None,
        top_k: int = 12,
        threshold: float = 0.20,
    ) -> dict[str, Any]:
        probs = self.predict_proba(embedding=embedding, acoustic=acoustic)
        order = np.argsort(-probs)
        selected = [i for i in order if float(probs[i]) >= threshold][:top_k]
        if not selected:
            selected = list(order[:top_k])
        tags = [{"tag": self.labels[i], "score": float(probs[i])} for i in selected]
        return {
            "source": "mert_kpop_classifier",
            "tags": tags,
            "top_tags": [t["tag"] for t in tags],
            "classifier_input_mode": self.input_mode,
        }


class AcousticPriorKPopClassifier:
    """Deterministic fallback that uses the same tag schema as the learned model.

    It behaves as a transparent prior classifier over K-pop labels when a
    trained MERT checkpoint is not available,
    so the package remains runnable after installation. The report will clearly
    mark the source as ``acoustic_prior_classifier``.
    """

    labels = DEFAULT_KPOP_TAGS

    def predict(self, features: dict[str, Any], top_k: int = 12, threshold: float = 0.20) -> dict[str, Any]:
        v, _ = acoustic_feature_vector(features)
        f = dict(zip(_, v))
        bpm = float(features.get("tempo", {}).get("bpm", 0.0) or 0.0)
        while 0 < bpm < 70:
            bpm *= 2
        while bpm > 190:
            bpm /= 2
        onset = f["onset_density_norm"]
        beat = f["beat_density_norm"]
        low = f["low_20_250_mean"]
        high = f["high_6000_12000_mean"]
        centroid = f["centroid_mean_norm"]
        energy = f["rms_p90"]
        contrast = f["energy_contrast"]
        chorus = f["chorus_lift_proxy"]
        pre = f["prechorus_build_proxy"]
        dance = f["dancebreak_proxy"]

        scores: dict[str, float] = {}

        def score(name: str, value: float) -> None:
            scores[name] = float(max(0.0, min(1.0, value)))

        fast = 1.0 if bpm >= 118 else max(0.0, (bpm - 92.0) / 36.0)
        mid = max(0.0, 1.0 - abs(bpm - 100.0) / 38.0)
        slow = max(0.0, (105.0 - bpm) / 45.0) if bpm > 0 else 0.25
        score("k-pop dance", 0.20 + 0.30 * fast + 0.25 * beat + 0.20 * chorus)
        score("dance-pop", 0.18 + 0.34 * fast + 0.22 * onset + 0.20 * energy)
        score("electropop", 0.16 + 0.30 * centroid + 0.28 * high + 0.16 * beat)
        score("hip-hop pop", 0.12 + 0.24 * mid + 0.28 * low + 0.18 * contrast)
        score("r&b pop", 0.14 + 0.35 * mid + 0.18 * low + 0.12 * (1.0 - onset))
        score("ballad", 0.18 + 0.45 * slow + 0.20 * (1.0 - onset) + 0.10 * (1.0 - energy))
        score("retro synth", 0.12 + 0.25 * high + 0.20 * centroid + 0.12 * (112 <= bpm <= 128))
        score("trap-pop", 0.10 + 0.25 * low + 0.18 * contrast + 0.14 * (bpm < 115))
        score("jersey club influence", 0.08 + 0.28 * (125 <= bpm <= 150) + 0.25 * onset)
        score("energetic", 0.18 + 0.35 * fast + 0.20 * energy + 0.20 * onset)
        score("bright", 0.12 + 0.36 * high + 0.25 * centroid + 0.10 * fast)
        score("dark", 0.12 + 0.32 * low + 0.25 * (1.0 - centroid) + 0.08 * slow)
        score("dreamy", 0.10 + 0.28 * high + 0.25 * (1.0 - onset) + 0.12 * mid)
        score("sentimental", 0.10 + 0.35 * slow + 0.20 * (1.0 - energy) + 0.15 * mid)
        score("confident", 0.12 + 0.30 * low + 0.22 * onset + 0.14 * fast)
        score("cute", 0.10 + 0.25 * high + 0.18 * fast + 0.12 * (1.0 - low))
        score("dramatic", 0.10 + 0.30 * contrast + 0.20 * chorus + 0.15 * low)
        score("heavy drums", 0.10 + 0.45 * onset + 0.20 * beat)
        score("synth bass", 0.10 + 0.46 * low + 0.12 * centroid)
        score("low-frequency drive", 0.10 + 0.55 * low + 0.10 * fast)
        score("bright high-frequency synth", 0.10 + 0.42 * high + 0.25 * centroid)
        score("vocal layering", 0.18 + 0.22 * chorus + 0.16 * high + 0.12 * contrast)
        score("rap section likely", 0.10 + 0.20 * mid + 0.20 * low + 0.15 * onset)
        score("drop chorus", 0.10 + 0.45 * chorus + 0.18 * low + 0.16 * onset)
        score("chorus energy lift", 0.10 + 0.62 * chorus)
        score("pre-chorus build-up", 0.10 + 0.62 * pre)
        score("dance break likely", 0.08 + 0.60 * dance + 0.12 * onset)
        score("sparse verse", 0.12 + 0.30 * (1.0 - onset) + 0.25 * (1.0 - energy))

        tags = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        picked = [(t, s) for t, s in tags if s >= threshold][:top_k]
        if not picked:
            picked = tags[:top_k]
        return {
            "source": "acoustic_prior_classifier_v2",
            "tags": [{"tag": t, "score": float(s)} for t, s in picked],
            "top_tags": [t for t, _ in picked],
            "note": "Fallback prior. Configure models.kpop_classifier_path to use a trained MERT K-pop classifier.",
        }


def load_classifier(path: str | Path | None, device: str = "auto") -> KPopClassifier | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    return KPopClassifier(path, device=device)


def save_label_map(labels: Sequence[str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(labels), f, ensure_ascii=False, indent=2)
