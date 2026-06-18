from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from kpop_scope.taxonomy import labels_by_group, label_info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Human-in-the-loop label assistant for local K-pop audio.")
    ap.add_argument("--manifest", default="data/local/twice_manifest.csv")
    ap.add_argument("--pseudo-labels", default="data/local/twice_pseudo_labels.csv")
    ap.add_argument("--analysis-dir", default="outputs/twice")
    ap.add_argument("--output", default="data/local/twice_human_labels.csv")
    return ap.parse_args()


def safe_name(rel_path: str) -> str:
    stem = Path(rel_path).stem.lower()
    text = re.sub(r"[^a-z0-9]+", "_", stem)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "track"


def split_values(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(";") if x.strip()]


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path).fillna("")
    return pd.DataFrame()


def save_row(output: Path, row: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        with open(output, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("track_id") != row.get("track_id")]
    rows.append(row)
    fields = list(row.keys())
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    pseudo_path = Path(args.pseudo_labels)
    analysis_dir = Path(args.analysis_dir)
    output_path = Path(args.output)

    st.set_page_config(page_title="KPopScope Annotation", layout="wide")
    st.title("KPopScope Annotation")

    manifest = read_csv(manifest_path)
    if manifest.empty:
        st.error(f"Manifest not found or empty: {manifest_path}")
        return
    pseudo = read_csv(pseudo_path)
    human = read_csv(output_path)

    options = [f"{r.filename} | {r.track_id}" for r in manifest.itertuples()]
    selected = st.sidebar.selectbox("Track", options)
    track_id = selected.split("|")[-1].strip()
    mrow = manifest[manifest["track_id"] == track_id].iloc[0].to_dict()
    prow = pseudo[pseudo["track_id"] == track_id].iloc[0].to_dict() if not pseudo.empty and (pseudo["track_id"] == track_id).any() else {}
    hrow = human[human["track_id"] == track_id].iloc[0].to_dict() if not human.empty and (human["track_id"] == track_id).any() else {}

    track_dir = analysis_dir / safe_name(str(mrow.get("rel_path", mrow.get("filename", ""))))
    analysis_path = track_dir / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}

    c1, c2, c3, c4 = st.columns(4)
    features = analysis.get("features", {})
    c1.metric("Filename", str(mrow.get("filename", "")))
    c2.metric("Duration", f"{float(mrow.get('duration', 0) or features.get('duration_seconds', 0)):.1f}s")
    c3.metric("BPM", f"{float(features.get('tempo', {}).get('bpm', 0.0) or prow.get('bpm', 0) or 0):.1f}")
    c4.metric("Key", str(features.get("key", {}).get("key", prow.get("key", "unknown"))))

    audio_path = manifest_path.parent.parent.parent / str(mrow.get("rel_path", ""))
    # Prefer input-dir relative to cwd when manifest is data/local/*.csv.
    cwd_audio = Path("twice") / str(mrow.get("rel_path", ""))
    if cwd_audio.exists():
        audio_path = cwd_audio
    if audio_path.exists():
        st.audio(str(audio_path))

    report_path = track_dir / "report.md"
    if report_path.exists():
        st.subheader("Report")
        st.markdown(report_path.read_text(encoding="utf-8"))

    fig_dir = track_dir / "figures"
    for name in ["waveform.png", "loudness_curve.png", "segment_timeline.png", "stem_energy.png"]:
        p = fig_dir / name
        if p.exists():
            st.image(str(p), caption=name, use_container_width=True)

    taxonomy = labels_by_group()
    st.subheader("Labels")
    result: dict[str, str] = {
        "track_id": track_id,
        "rel_path": str(mrow.get("rel_path", "")),
        "filename": str(mrow.get("filename", "")),
        "source": "human",
    }
    pseudo_cols = {
        "Style": "candidate_style_tags",
        "Mood": "candidate_mood_tags",
        "Arrangement": "candidate_arrangement_tags",
        "Structure": "candidate_structure_tags",
    }
    human_cols = {
        "Style": "human_style_tags",
        "Mood": "human_mood_tags",
        "Arrangement": "human_arrangement_tags",
        "Structure": "human_structure_tags",
    }
    for group, labels in taxonomy.items():
        default = split_values(hrow.get(human_cols[group], "")) or split_values(prow.get(pseudo_cols[group], ""))
        selected_labels = st.multiselect(group, labels, default=[x for x in default if x in labels])
        result[human_cols[group]] = ";".join(selected_labels)
        for lab in selected_labels:
            info = label_info(lab) or {}
            evidence = info.get("audio_evidence", [])
            st.caption(f"{lab}: {info.get('annotation_guideline_zh', '')} Evidence: {', '.join(evidence[:3])}")

    st.text_area("Pseudo evidence", value=str(prow.get("evidence", "")), disabled=True)
    result["human_notes"] = st.text_area("Human notes", value=str(hrow.get("human_notes", "")))
    result["pseudo_label_confidence"] = str(prow.get("pseudo_label_confidence", ""))
    result["needs_human_review"] = str(prow.get("needs_human_review", ""))

    if st.button("Save labels", type="primary"):
        save_row(output_path, result)
        st.success(f"Saved {output_path}")


if __name__ == "__main__":
    main()
