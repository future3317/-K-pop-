# Course Project Research Plan

## Proposed title

基于预训练音乐表征与声部级特征融合的 K-pop 自动赏析系统

## Background

K-pop production often features clear section-level arrangement, strong rhythm design, pre-chorus build-ups, drop/chorus energy release, vocal layering and concept-oriented timbres. Standard music tagging systems usually output high-level genre or mood tags but give limited explanations about how these perceptual impressions arise from audio signals.

## Motivation

A listener may not only ask "what genre is this song?" but also "why does the chorus feel powerful?" and "what role do drums, bass, vocals and synth layers play?" This project moves from automatic music tagging to explainable K-pop appreciation.

## Contributions

1. A complete local MP3-to-report K-pop MIR pipeline.
2. A MERT-based K-pop multi-label classifier interface with acoustic feature fusion.
3. A K-pop-aware segment labeling algorithm for intro/verse/pre-chorus/chorus/drop/bridge/dance break.
4. A stem-wise analysis module using vocals/drums/bass/other features.
5. A K-pop-oriented explanation schema that links labels to BPM, energy, onset, spectral and stem evidence.

## Baselines

- Acoustic-prior classifier using full-mix handcrafted features.
- MERT embedding + MLP classifier.
- Acoustic-only MLP classifier.

## Proposed method

The proposed model freezes MERT and trains a lightweight multi-label classifier. The recommended representation is:

```text
[MERT full-mix embedding ; acoustic feature vector]
```

The acoustic feature vector contains tempo, beat density, onset density, RMS dynamics, spectral brightness, frequency-band ratios and segment-level proxies for chorus lift, pre-chorus build and dance break. The final report uses structured evidence from features, segments and optional stems.

## Evaluation

- Multi-label metrics: macro-F1, micro-F1, ROC-AUC, PR-AUC.
- Ablation: acoustic-prior vs MERT-only vs acoustic-only vs MERT+acoustic fusion.
- Explanation evaluation: report without stems vs report with stems.
- Human evaluation: accuracy, usefulness, readability, K-pop specificity.

## Implementation milestones

1. v0.1: Working acoustic analyzer and report generator.
2. v0.2: MERT embedding extraction and classifier training.
3. v0.3: Improved K-pop-aware segmentation and detailed arrangement rules.
4. v0.4: Stem-aware classifier and ablation.
5. v0.5: Streamlit demo and polished README.
