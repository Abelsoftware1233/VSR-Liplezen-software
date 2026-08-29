# Accuracy Notes

This document exists so nobody using this repo is surprised by its output
quality. Read it before you rely on this for anything important.

## Why lip reading can't be "perfect"

Many distinct words produce nearly identical lip movement — this is called
a **viseme clash**. Classic examples in English:

- "pat" / "bat" / "mat"
- "pack" / "back" / "mac"
- Nasal sounds like "m" vs "n" vs "ng" are often visually indistinguishable
- Words that differ mainly by tongue position (invisible on camera), e.g.
  many "t"/"d"/"n" contrasts

No amount of better training data fully resolves this, because the
information genuinely isn't present in the visual signal — it's an
information bottleneck, not a modeling weakness. Deaf and hard-of-hearing
expert lip readers face the exact same ceiling, which is why lip reading is
normally combined with context, hearing aids, or captions in real-world use
— never relied on alone.

## Current state of the art (what this repo uses)

| Model | Visual-only WER (LRS3) | Notes |
|---|---|---|
| Auto-AVSR | ~19.5–20.3% | Best open-source result as of the papers referenced below |
| AV-HuBERT Large | ~28.6% | Self-supervised pretraining, LRS3 fine-tuned |
| SLAM-LLM (LLM decoder) | ~28.3% | LLM-based decoding, similar ceiling |

These numbers are measured on **LRS3**: clean, front-facing, well-lit TED
talk footage, native English speakers, single speaker per clip. Your actual
video will very likely perform worse than this if it differs from those
conditions.

## Things that will make results noticeably worse

- **Camera angle**: anything beyond ~30° off from front-facing degrades
  detection and accuracy sharply.
- **Lighting**: shadows across the mouth, backlighting, low light.
- **Facial hair**: mustaches/beards that obscure lip contours.
- **Speaking speed**: fast or slurred speech is harder than deliberate,
  clear speech.
- **Resolution**: the mouth region needs to be reasonably large in the
  frame — a face that's a small part of a wide shot won't have enough
  detail after cropping.
- **Multiple people in frame**: the face tracker in this repo only tracks
  one face (the first one detected) and does not handle multiple speakers.
- **Non-English speech**: all bundled models are English-only. Running
  Dutch, French, etc. video through this will produce garbage output.
- **Background motion / occlusion**: hands near the face, objects passing
  in front of the mouth.

## What the language model rescoring step can and can't fix

The LM correction pass (see `models/lm_rescore.py`) can:
- Fix grammar and punctuation
- Nudge locally ambiguous word choices toward what's contextually more
  likely (this works better with the optional local-LLM rescoring path
  than the lightweight n-gram pass)

It cannot:
- Recover a word that the visual model got completely wrong with high
  confidence
- Know facts about your specific video (names, jargon, unusual phrases)
  unless they happen to be statistically common in its training data
- Guarantee the corrected sentence matches what was actually said — it is
  optimizing for "plausible English," not "true to the source"

## Ensembling (AV-HuBERT + Auto-AVSR)

Enabling the secondary AV-HuBERT model in the app roughly doubles CPU time
and, in the current implementation, uses a very simple heuristic (prefer
the longer output) to combine results — this is not a proper confusion
network / ROVER merge, which would require word-level timing and
confidence scores from both decoders. Treat the ensembling option as
experimental; it will not reliably outperform Auto-AVSR alone, and in some
cases may not help at all.

## Bottom line

Use this for: exploration, prototyping, accessibility experiments,
research, or as a component in a larger system that combines lip reading
with other signals (context, partial audio, user correction).

Do not use this for: legal transcription, medical communication, security
authentication, or any situation where an incorrect transcript could cause
real harm. If accurate transcription genuinely matters, use audio-based
speech recognition when any audio is available at all — it is dramatically
more reliable than lip reading (Auto-AVSR's own audio-based WER on the same
benchmark is ~1%, versus ~20% for visual-only).
