import subprocess
import sys


def test_required_scripts_help():
    for script in [
        "scripts/build_manifest.py",
        "scripts/batch_analyze.py",
        "scripts/bootstrap_pseudo_labels.py",
        "scripts/analyze_stem_contribution.py",
        "scripts/run_ablation.py",
    ]:
        proc = subprocess.run([sys.executable, script, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, proc.stderr
