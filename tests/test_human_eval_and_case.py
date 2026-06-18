import csv
import json
import subprocess
import sys


def test_human_eval_sheet_and_case_study(tmp_path):
    track = tmp_path / "outputs" / "track"
    track.mkdir(parents=True)
    (track / "report_tag_only.md").write_text("tag", encoding="utf-8")
    (track / "report_evidence_grounded.md").write_text("evidence", encoding="utf-8")
    sheet = tmp_path / "research" / "human_eval_sheet.csv"
    subprocess.run([sys.executable, "scripts/build_human_eval_sheet.py", "--analysis-dir", str(tmp_path / "outputs"), "--output", str(sheet)], check=True)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert rows and rows[0]["system_A"] in {"tag_only", "evidence_grounded"}

    analysis = track / "analysis.json"
    analysis.write_text(json.dumps({
        "features": {
            "duration_seconds": 60,
            "tempo": {"bpm": 120},
            "key": {"key": "C major"},
            "segments": {"segments": [{"start": 0, "end": 10, "label": "chorus/drop", "label_confidence": 0.8, "energy_mean": 0.8, "onset_density": 0.7, "evidence": ["high energy"]}]},
        },
        "tag_result": {"tags": [{"tag": "dance-pop", "score": 0.9}]},
        "stem_contribution": {"tag_contributions": [{"tag": "dance-pop", "evidence": ["rhythm"]}]},
    }), encoding="utf-8")
    case = tmp_path / "case.md"
    subprocess.run([sys.executable, "scripts/make_case_study.py", "--analysis", str(analysis), "--output", str(case)], check=True)
    assert "chorus/drop" in case.read_text(encoding="utf-8")
