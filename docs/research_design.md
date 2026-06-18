# Research Design

KPopScope is framed as an explainable MIR system for K-pop appreciation, not as an end-to-end generative music model.

## Research Contributions

1. K-pop-specific arrangement taxonomy
   - Labels cover style, mood, arrangement, and structure.
   - Arrangement labels include `pre-chorus buildup`, `chorus energy lift`, `drop chorus`, `instrumental post-chorus`, `dance break`, and `vocal layering`.

2. Stem-aware interpretable tagging
   - audio-separator stems are optional.
   - When available, vocals, drums, bass, and other stems provide evidence for arrangement tags.
   - Without stems, the system falls back to acoustic evidence and marks it as rule-based.

3. Structure-aware evidence-grounded appreciation generation
   - Reports bind claims to BPM, onset density, loudness change, spectral brightness, segment labels, and stem energy.
   - `tag_only` is the baseline report mode.
   - `evidence_grounded` is the full system mode.

4. Weak-label + human-in-the-loop low-resource adaptation
   - Pseudo labels are explicitly marked as `source=pseudo`.
   - The Streamlit annotation app lets users confirm, delete, or edit candidates.
   - Tiny classifier training is a smoke/overfit sanity check when only 13 local tracks are available.

## Limits

The local TWICE folder is private copyrighted audio. It is useful for demo, debugging, and case studies only. It must not be committed, copied into examples, or distributed.
