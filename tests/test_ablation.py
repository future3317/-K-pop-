import csv
import subprocess
import sys


def test_run_ablation_tiny(tmp_path):
    labels = tmp_path / "labels.csv"
    with open(labels, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "candidate_style_tags", "candidate_mood_tags", "candidate_arrangement_tags", "candidate_structure_tags"])
        writer.writeheader()
        writer.writerow({"track_id": "a", "candidate_style_tags": "dance-pop", "candidate_mood_tags": "bright", "candidate_arrangement_tags": "", "candidate_structure_tags": ""})
        writer.writerow({"track_id": "b", "candidate_style_tags": "ballad", "candidate_mood_tags": "sentimental", "candidate_arrangement_tags": "", "candidate_structure_tags": ""})
    out = tmp_path / "research"
    subprocess.run([sys.executable, "scripts/run_ablation.py", "--labels", str(labels), "--output-dir", str(out), "--modes", "acoustic"], check=True)
    assert (out / "ablation_results.csv").exists()
