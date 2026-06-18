# Literature Review Notes

## Music Foundation Models

MERT-style models provide frozen music representations that can be reused for tagging and retrieval. In this project, MERT is treated as an optional feature extractor, not a model trained from scratch.

## K-pop Genre and Korean Music Data

K-pop classification work such as MIREX Audio K-POP Genre Classification and Korean platform resources such as the Melon Playlist Dataset motivate domain-specific tags, but KPopScope focuses more on arrangement and appreciation evidence than plain genre classification.

## Music Structure Analysis

Music structure analysis commonly uses novelty curves, self-similarity, repetition, and boundary strength. KPopScope uses a lightweight explainable variant tuned for K-pop-like sections: intro, verse, pre-chorus, chorus/drop, post-chorus, dance break, bridge, outro, transition, and unknown.

## Source Separation

Demucs can separate a mix into vocals, drums, bass, and other stems. These stems are used as optional evidence units for arrangement interpretation.

## Explainable MIR

audioLIME motivates using musically meaningful components rather than arbitrary spectrogram patches. KPopScope follows this idea at a practical level by reporting stem contribution and segment evidence.

## Music Captioning

MusicCaps, LP-MusicCaps, and MusiLingo show the value of music-to-language systems. KPopScope uses evidence-grounded template generation so claims remain traceable under low-resource conditions.
