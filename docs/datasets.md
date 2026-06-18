# Dataset Plan

KPopScope does not bundle copyrighted music. Use these notes to prepare data yourself.

## Option A: MTG-Jamendo for a legal open baseline

Use MTG-Jamendo to train a general music tagging baseline. It contains Creative Commons music with genre/instrument/mood tags. It is not K-pop-specific, but it is useful for reproducible MIR experiments.

Suggested use:

1. Download audio and split files according to the dataset license.
2. Train a generic tag classifier.
3. Use it as a baseline before K-pop domain adaptation.

## Option B: Melon Playlist Dataset for Korean music context

The Melon Playlist Dataset provides playlist/tag metadata and mel-spectrogram representations from a Korean music platform. It is useful for Korean music tagging/recommendation context, but you need to follow its data format and licensing carefully.

Suggested use:

- Learn a tag classifier from provided representations.
- Use tags as weak labels for Korean/K-pop style categories.

## Option C: Small local K-pop annotation set

For the course project, a small private/local annotation set can be enough:

```csv
path,tags
/path/to/song1.mp3,"dance-pop;energetic;heavy drums;chorus energy lift"
/path/to/song2.mp3,"r&b pop;dreamy;vocal layering;sparse verse"
```

Do not redistribute the audio. You can publish:

- the tag schema
- the annotation tool/script
- feature extraction code
- trained weights only if data/license permits
- evaluation protocol

## Suggested K-pop tags

Style:

- dance-pop
- electropop
- hip-hop pop
- r&b pop
- ballad
- rock pop
- retro synth
- trap-pop
- jersey club influence

Mood:

- energetic
- bright
- dark
- dreamy
- sentimental
- confident
- cute
- dramatic
- chill

Arrangement:

- heavy drums
- synth bass
- vocal layering
- rap section likely
- drop chorus
- chorus energy lift
- pre-chorus build-up
- dance break likely
- sparse verse
- bright high-frequency synth
- low-frequency drive
