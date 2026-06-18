# KPopScope Design Notes

## Goal

KPopScope is a practical K-pop MIR system, not an end-to-end black-box music LLM. It emphasizes explainable audio features, MERT-based music representation, and stem-wise arrangement analysis.

## Pipeline

```text
local audio
  -> loader/resampling
  -> full-mix MIR features
  -> K-pop-aware structure segmentation
  -> optional audio-separator stems
  -> stem-wise MIR features
  -> MERT embedding + K-pop classifier checkpoint
       or acoustic-prior fallback
  -> K-pop explanation rules
  -> JSON / Markdown / plots
```

## v2 tagging stack

Recommended learned path:

```text
audio -> MERT embedding
      -> concatenate acoustic feature vector
      -> MLP multi-label classifier
      -> K-pop style/mood/arrangement tags
```

The classifier supports three input modes:

- `mert`: MERT embedding only
- `acoustic`: explainable acoustic features only
- `fusion`: MERT embedding + acoustic feature vector

When no checkpoint is configured, the package uses `acoustic_prior_classifier_v2` so that the project remains runnable immediately. This fallback should be treated as a baseline, not the final method.

## Segmentation v2

`audio/segmentation.py` now uses:

- RMS energy
- onset strength
- spectral centroid / bandwidth / rolloff / zero-crossing rate
- chroma stability
- MFCC changes
- duration constraints and soft beat/bar candidates

It outputs coarse K-pop labels:

- intro
- verse
- pre-chorus
- chorus/drop
- bridge
- dance break
- outro

Each label has confidence and human-readable evidence.

## Research contribution candidates

1. **MERT + acoustic feature fusion**: compare MERT-only, acoustic-only and fusion classifiers.
2. **Stem-aware feature fusion**: compare full-mix features with full-mix + stem features.
3. **K-pop tag schema**: define K-pop-specific arrangement tags such as drop chorus, dance break, pre-chorus build-up, vocal layering.
4. **Explainable report generation**: transform tag predictions into interpretable evidence linked to BPM, onset density, energy curves, segments and stems.

## Recommended experiments

- Classification metrics: macro-F1, micro-F1, ROC-AUC, PR-AUC.
- Ablation:
  - acoustic-prior baseline
  - MERT-only classifier
  - acoustic-only classifier
  - MERT + acoustic fusion classifier
  - report without stems vs report with stems
- Human evaluation:
  - factual accuracy
  - musical usefulness
  - K-pop specificity
  - readability

## What to implement after v2

- Downbeat tracking and bar-level feature pooling.
- Better repeated-section detection with self-similarity matrices.
- Optional chord/key estimation with more robust models.
- Learned segment labeling for intro/verse/pre-chorus/chorus/bridge/dance break.
- Add stem-level features into the classifier input after source separation.
- LLM-based report rewriting constrained by structured JSON.
