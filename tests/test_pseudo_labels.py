import csv
import json
import subprocess
import sys


def test_bootstrap_pseudo_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "rel_path", "filename"])
        writer.writeheader()
        writer.writerow({"track_id": "a", "rel_path": "Song.wav", "filename": "Song.wav"})
    analysis_dir = tmp_path / "outputs"
    track = analysis_dir / "song"
    track.mkdir(parents=True)
    (track / "analysis.json").write_text(json.dumps({
        "features": {
            "tempo": {"bpm": 120},
            "key": {"key": "C major"},
            "onset": {"onset_density_per_sec": 2.5},
            "loudness": {"rms_stats": {"p90": 0.1}},
            "spectral": {"centroid_hz": {"mean": 3000}},
            "segments": {"segments": [{"label": "chorus/drop", "energy_delta": 0.2}], "structure_summary": {"has_chorus_drop": True, "structure_confidence": 0.7}},
        },
        "tag_result": {"source": "acoustic_prior_classifier_v2", "tags": [{"tag": "dance-pop", "score": 0.8}, {"tag": "bright", "score": 0.7}]},
    }), encoding="utf-8")
    out = tmp_path / "pseudo.csv"
    subprocess.run([sys.executable, "scripts/bootstrap_pseudo_labels.py", "--manifest", str(manifest), "--analysis-dir", str(analysis_dir), "--output", str(out)], check=True)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["source"] == "pseudo"
    assert "dance-pop" in rows[0]["candidate_style_tags"]
