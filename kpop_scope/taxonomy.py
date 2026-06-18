from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "configs" / "kpop_taxonomy.yaml"


def load_taxonomy(path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load the K-pop taxonomy used by pseudo labels, reports, and annotation UI."""
    taxonomy_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    groups = data.get("groups", data)
    normalized: dict[str, list[dict[str, Any]]] = {}
    for group, items in groups.items():
        normalized[str(group)] = []
        for item in items or []:
            row = dict(item)
            row.setdefault("group", str(group))
            row.setdefault("name", "")
            row.setdefault("description_zh", "")
            row.setdefault("positive_cues", [])
            row.setdefault("negative_cues", [])
            row.setdefault("audio_evidence", [])
            row.setdefault("annotation_guideline_zh", "")
            row.setdefault("uncertainty_policy", "证据不足时保持不确定，不作为人工真值。")
            normalized[str(group)].append(row)
    return normalized


@lru_cache(maxsize=4)
def cached_taxonomy(path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    return load_taxonomy(path)


def labels_by_group(path: str | Path | None = None) -> dict[str, list[str]]:
    return {group: [str(item["name"]) for item in items] for group, items in load_taxonomy(path).items()}


def all_labels(path: str | Path | None = None) -> list[str]:
    labels: list[str] = []
    for items in load_taxonomy(path).values():
        labels.extend(str(item["name"]) for item in items)
    return labels


def label_info(name: str, path: str | Path | None = None) -> dict[str, Any] | None:
    for items in load_taxonomy(path).values():
        for item in items:
            if item.get("name") == name:
                return item
    return None
