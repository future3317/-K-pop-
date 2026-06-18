#!/usr/bin/env python
"""Train a lightweight K-pop multi-label classifier on extracted features.

Input NPZ expected from extract_embeddings.py:
    X: MERT embeddings [N, D]
    A: acoustic features [N, F]
    tags_text: semicolon-separated labels

Typical training:
    python scripts/train_classifier.py \
      --embeddings data/kpop_embeddings.npz \
      --output checkpoints/kpop_mert_fusion.pt \
      --input-mode fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def parse_tags(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(";") if t.strip()]


def choose_input(payload, mode: str) -> tuple[np.ndarray, int, int, str]:
    X = payload["X"].astype("float32") if "X" in payload.files else np.zeros((0, 0), dtype="float32")
    A = payload["A"].astype("float32") if "A" in payload.files else np.zeros((len(X), 0), dtype="float32")
    if mode == "mert":
        if X.shape[1] == 0:
            raise ValueError("input-mode=mert requires non-empty X embeddings")
        return X, int(X.shape[1]), 0, "mert"
    if mode == "acoustic":
        if A.shape[1] == 0:
            raise ValueError("input-mode=acoustic requires A acoustic features")
        return A, 0, int(A.shape[1]), "acoustic"
    if mode == "fusion":
        if X.shape[1] == 0 or A.shape[1] == 0:
            raise ValueError("input-mode=fusion requires both X embeddings and A acoustic features")
        return np.concatenate([X, A], axis=1).astype("float32"), int(X.shape[1]), int(A.shape[1]), "fusion"
    raise ValueError(f"Unknown input mode: {mode}")


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise SystemExit("Install torch first: pip install -e .[models]") from exc

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-mode", choices=["mert", "acoustic", "fusion"], default="fusion")
    ap.add_argument("--mode", choices=["mert", "acoustic", "fusion"], default=None, help="Alias for --input-mode")
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.mode:
        args.input_mode = args.mode

    if Path(args.embeddings).is_dir():
        emb_dir = Path(args.embeddings)
        X_arr = np.load(emb_dir / "twice_mert.npy") if (emb_dir / "twice_mert.npy").exists() else np.zeros((0, 0), dtype="float32")
        A_arr = np.load(emb_dir / "twice_acoustic.npy") if (emb_dir / "twice_acoustic.npy").exists() else np.zeros((len(X_arr), 0), dtype="float32")
        meta = json.loads((emb_dir / "twice_ids.json").read_text(encoding="utf-8")) if (emb_dir / "twice_ids.json").exists() else {}
        tags_text = meta.get("tags_text", [""] * len(A_arr))
        if args.labels:
            import csv
            label_rows = list(csv.DictReader(open(args.labels, "r", encoding="utf-8", newline="")))
            tags_text = []
            for r in label_rows:
                vals = []
                for key in ["human_style_tags", "human_mood_tags", "human_arrangement_tags", "human_structure_tags", "candidate_style_tags", "candidate_mood_tags", "candidate_arrangement_tags", "candidate_structure_tags"]:
                    vals.extend(parse_tags(r.get(key, "")))
                tags_text.append(";".join(sorted(set(vals))))
        class Payload:
            files = ["X", "A", "tags_text", "acoustic_feature_names"]
            def __getitem__(self, key):
                if key == "X":
                    return X_arr
                if key == "A":
                    return A_arr
                if key == "tags_text":
                    return np.array(tags_text, dtype=object)
                if key == "acoustic_feature_names":
                    return np.array(meta.get("acoustic_feature_names", []), dtype=object)
                raise KeyError(key)
        payload = Payload()
    else:
        payload = np.load(args.embeddings, allow_pickle=True)
    X, embedding_dim, acoustic_dim, input_mode = choose_input(payload, args.input_mode)
    tags = [parse_tags(t) for t in payload["tags_text"]]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags).astype("float32")
    if Y.shape[1] == 0:
        raise ValueError("No labels found in tags_text.")
    if len(Y) < 30:
        print("WARNING: tiny dataset; this is smoke training / overfit sanity check, not evidence of generalization.")

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=args.test_size, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_x = torch.tensor(X_val)
    val_y = torch.tensor(y_val)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(X.shape[1], args.hidden_dim),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden_dim, Y.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_macro = -1.0
    best_state = None
    best_auc = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(val_x.to(device)).cpu()
            probs = torch.sigmoid(logits).numpy()
        preds = (probs >= args.threshold).astype("int32")
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(y_val, preds, average="micro", zero_division=0)
        try:
            auc = roc_auc_score(y_val, probs, average="macro")
        except Exception:
            auc = float("nan")
        print(
            f"epoch={epoch:03d} loss={total_loss/len(train_ds):.4f} "
            f"macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f} macro_auc={auc:.4f}"
        )
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    acoustic_feature_names = payload["acoustic_feature_names"].tolist() if "acoustic_feature_names" in payload.files else []
    torch.save(
        {
            "model_state_dict": best_state or model.state_dict(),
            "labels": list(mlb.classes_),
            "input_dim": int(X.shape[1]),
            "embedding_dim": int(embedding_dim),
            "acoustic_dim": int(acoustic_dim),
            "input_mode": input_mode,
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "threshold": float(args.threshold),
            "best_macro_f1": float(best_macro),
            "best_macro_auc": float(best_auc),
            "acoustic_feature_names": acoustic_feature_names,
        },
        out,
    )
    config_path = out.with_name(out.stem + "_config.json")
    config_path.write_text(
        json.dumps(
            {
                "labels": list(mlb.classes_),
                "input_mode": input_mode,
                "input_dim": int(X.shape[1]),
                "embedding_dim": int(embedding_dim),
                "acoustic_dim": int(acoustic_dim),
                "small_sample_warning": len(Y) < 30,
                "label_support": {label: int(Y[:, i].sum()) for i, label in enumerate(mlb.classes_)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
