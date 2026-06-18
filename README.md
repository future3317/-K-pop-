# KPopScope

**KPopScope** is a K-pop oriented Music Information Retrieval toolkit. It turns a local audio file into structured MIR features, K-pop style/mood/arrangement tags, stem-wise arrangement analysis, plots, and a Chinese music appreciation report.

> 中文定位：输入一首 mp3，输出 BPM、调性、段落能量、鼓/贝斯/人声/伴奏声部分析、K-pop 风格/情绪/编曲标签，以及一份中文赏析报告。

This repository is designed for a course project in **Audio Information Processing / Music Information Retrieval** and as a practical open-source tool that other users can run locally.

## What it can do now

- Load `mp3/wav/flac/m4a` with `librosa`.
- Extract MIR features:
  - tempo / beat positions
  - onset envelope and onset density
  - chroma and estimated key
  - RMS loudness and energy curve
  - spectral centroid, bandwidth, rolloff, zero-crossing rate
  - coarse low/mid/high-band energy
- Estimate K-pop-oriented song structure from multi-feature novelty:
  - `intro`, `verse`, `pre-chorus`, `chorus/drop`, `bridge`, `dance break`, `outro`
  - each segment includes confidence and evidence.
- Predict K-pop tags through the v2 tagging stack:
  - trained path: **MERT embedding + lightweight K-pop multi-label classifier**
  - fallback path: transparent acoustic-prior classifier when no checkpoint is configured
  - tags include `dance-pop`, `electropop`, `trap-pop`, `jersey club influence`, `vocal layering`, `drop chorus`, `pre-chorus build-up`, `dance break likely`, etc.
- Optionally call **audio-separator** to split audio into `vocals/drums/bass/other` or the stems supported by the selected model, then analyze each stem.
- Generate:
  - `analysis.json`
  - `report.md`
  - optional plots: waveform, loudness, novelty, segment timeline, stem energy.
- Provide training utilities for:
  - MERT embedding extraction
  - acoustic feature extraction
  - MERT + acoustic feature fusion classifier training
  - Streamlit web demo.

## What it intentionally does not include

This repo does **not** ship K-pop audio files or copyrighted datasets. You should use local audio that you are allowed to analyze. For research, prepare datasets yourself according to their licenses.

## Installation

Core analyzer:

```bash
pip install -e .
```

With optional model/stem/app dependencies:

```bash
pip install -e ".[all]"
```

Stem separation uses `audio-separator`, which can download UVR/RoFormer/MDX-style models automatically. If installation fails, install PyTorch matching your CUDA environment first, then:

```bash
pip install audio-separator
```

## Quick start

Analyze one local file:

```bash
kpop-scope analyze ./song.mp3 --output outputs/song --plots
```

With stem-wise analysis using audio-separator:

```bash
kpop-scope analyze ./song.mp3 --output outputs/song --stems --plots
```

You can override the separator model:

```bash
kpop-scope analyze ./song.mp3 --output outputs/song --stems --stem-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

With MERT embedding and a trained classifier checkpoint:

```bash
kpop-scope analyze ./song.mp3 \
  --output outputs/song \
  --mert \
  --classifier checkpoints/kpop_mert_fusion.pt
```

The tool will write:

```text
outputs/song/
  analysis.json
  report.md
  figures/
    waveform.png
    loudness_curve.png
    novelty_curve.png
    segment_timeline.png
    stem_energy.png      # when --stems succeeds
```

Use as Python API:

```python
from kpop_scope import analyze

result = analyze(
    "song.mp3",
    output_dir="outputs/song",
    use_stems=True,
    make_plots=True,
    use_mert=True,
    classifier_path="checkpoints/kpop_mert_fusion.pt",
)
print(result["report_markdown"])
```

## Training your K-pop classifier

Prepare a CSV manifest. Do not commit copyrighted audio.

```csv
path,tags
/path/to/song1.mp3,"dance-pop;bright;chorus energy lift;vocal layering"
/path/to/song2.mp3,"trap-pop;dark;rap section likely;dance break likely"
```

Extract embeddings and acoustic features:

```bash
python scripts/extract_embeddings.py \
  --manifest data/manifest.csv \
  --output data/kpop_embeddings.npz \
  --model m-a-p/MERT-v1-95M \
  --max-duration 45
```

Train a fusion classifier:

```bash
python scripts/train_classifier.py \
  --embeddings data/kpop_embeddings.npz \
  --output checkpoints/kpop_mert_fusion.pt \
  --input-mode fusion \
  --epochs 40
```

Other modes are available:

```bash
python scripts/train_classifier.py --embeddings data/kpop_embeddings.npz --output checkpoints/kpop_mert.pt --input-mode mert
python scripts/train_classifier.py --embeddings data/kpop_embeddings.npz --output checkpoints/kpop_acoustic.pt --input-mode acoustic
```

A single RTX 4060 Ti is enough for embedding extraction and the lightweight MLP classifier in the local course-project setting. The recommended route is to freeze MERT and train only the classifier head.

## Streamlit demo

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

## Local TWICE Demo

`twice/` or `TWICE/` is treated as a private local audio directory. Do not commit, copy, package, or upload these audio files. The demo uses analysis outputs only.

```bash
python scripts/build_manifest.py --input-dir twice --output data/local/twice_manifest.csv

python scripts/batch_analyze.py \
  --input-dir twice \
  --output-dir outputs/twice \
  --plots \
  --report md

python scripts/bootstrap_pseudo_labels.py \
  --manifest data/local/twice_manifest.csv \
  --analysis-dir outputs/twice \
  --output data/local/twice_pseudo_labels.csv

streamlit run app/annotation_app.py -- \
  --manifest data/local/twice_manifest.csv \
  --pseudo-labels data/local/twice_pseudo_labels.csv \
  --analysis-dir outputs/twice \
  --output data/local/twice_human_labels.csv
```

Optional CUDA/MERT path:

```bash
python scripts/extract_embeddings.py \
  --manifest data/local/twice_manifest.csv \
  --output-dir data/local/embeddings \
  --mert \
  --device cuda \
  --resume

python scripts/train_classifier.py \
  --labels data/local/twice_human_labels.csv \
  --embeddings data/local/embeddings \
  --mode fusion \
  --output checkpoints/kpop_classifier_tiny.pt
```

The 13-song local demo is only suitable for demo, smoke testing, and qualitative case studies. It is not enough to prove generalization. For real training, add legal labeled data such as MTG-Jamendo, MusicCaps-derived metadata, Melon Playlist Dataset representations, or a self-built licensed annotation set.

Do not publish local copyrighted artifacts:

- `TWICE/` or `twice/`
- `outputs/`, especially separated stems under `outputs/twice_stems/`
- `data/local/`
- MERT embeddings such as `twice_mert.npy` / `twice_acoustic.npy`
- tiny classifier weights trained from private audio-derived labels

GitHub releases should contain code, configs, docs, tests, schemas, and synthetic or properly licensed examples only.

## Course Paper Experiment Workflow

```bash
python scripts/build_manifest.py --input-dir twice --output data/local/twice_manifest.csv

python scripts/batch_analyze.py --input-dir twice --output-dir outputs/twice --plots --both-report-modes

python scripts/bootstrap_pseudo_labels.py --manifest data/local/twice_manifest.csv --analysis-dir outputs/twice --output data/local/twice_pseudo_labels.csv

streamlit run app/annotation_app.py -- --manifest data/local/twice_manifest.csv --pseudo-labels data/local/twice_pseudo_labels.csv --analysis-dir outputs/twice --output data/local/twice_human_labels.csv

python scripts/run_ablation.py --manifest data/local/twice_manifest.csv --labels data/local/twice_human_labels.csv --analysis-dir outputs/twice --output-dir outputs/research --loo

python scripts/build_human_eval_sheet.py --analysis-dir outputs/twice --output outputs/research/human_eval_sheet.csv

python scripts/make_paper_tables.py --research-dir outputs/research --output outputs/research/paper_tables.md
```

Report modes:

```bash
kpop-scope analyze song.flac --report-mode tag_only
kpop-scope analyze song.flac --report-mode evidence_grounded
```

`tag_only` is the baseline. `evidence_grounded` uses tags, segmentation, optional stems, and stem contribution evidence.

## Recommended course-project framing

Suggested title:

> 基于预训练音乐表征与声部级特征融合的 K-pop 自动赏析系统

Core idea:

1. Build a complete K-pop MIR pipeline from raw audio.
2. Use source separation to make arrangement analysis interpretable.
3. Use MERT embeddings plus acoustic features to train a lightweight K-pop tag classifier.
4. Convert structured analysis into a readable Chinese appreciation report.

Suggested comparisons:

| Method | Input | Output | Explainability |
|---|---|---|---|
| Baseline A | full mix acoustic features | acoustic-prior tags | medium |
| Baseline B | full mix MERT embedding | learned tags | medium |
| Ours | MERT embedding + acoustic features + stems/segments | K-pop tags + arrangement report | strong |

## Dataset notes

Useful public resources to investigate:

- MTG-Jamendo: Creative Commons music tagging dataset.
- Melon Playlist Dataset: Korean music platform playlist/tag dataset with mel-spectrogram representations.
- MIREX Audio K-POP Genre Classification: classic K-pop genre task definition.
- MusicCaps / LP-MusicCaps: music captioning direction.

See [`docs/datasets.md`](docs/datasets.md) for a practical data plan.

## Project structure

```text
kpop_scope/
  audio/       audio loading, features, key, beat, K-pop-aware segmentation
  stems/       audio-separator integration and stem-wise feature analysis
  models/      MERT embedder, classifier, acoustic features, tagger orchestration
  explain/     K-pop interpretation rules and report generation
  visualize/   plots
scripts/       embedding extraction, training, evaluation utilities
app/           Streamlit demo
docs/          design and research notes
```

## License

MIT for code. This license does not grant rights to any third-party music/audio datasets.
