# License Notes

This repository's own code is provided under whatever license you choose to
apply to your fork/repo. However, it **downloads and depends on
pretrained weights and datasets from other projects**, each with their own
license terms. Check these before using this for anything beyond personal
experimentation, and especially before any commercial use.

## Auto-AVSR (github.com/mpc001/auto_avsr)
- Code: Apache 2.0
- Pretrained weights: derived from training on LRS3, which is distributed
  under TED's terms of use plus **CC BY-NC-ND 4.0** (non-commercial, no
  derivatives). This means the *weights* inherit a non-commercial
  restriction even though the *code* is Apache 2.0.

## AV-HuBERT (github.com/facebookresearch/av_hubert)
- Code: MIT-style (see repo for exact terms)
- Pretrained checkpoints: Meta requires accepting a separate license
  agreement before downloading the LRS3-finetuned model — this repo's
  `download_models.py` cannot do that for you, you must do it manually via
  the link it prints.

## LRS2 / LRS3 datasets
- Not redistributed by this repo — only used indirectly via pretrained
  weights that were trained on them.
- If you ever fine-tune further on LRS2/LRS3 yourself, note both are
  restricted to non-commercial research use and require separate access
  requests from Oxford's VGG group.

## MediaPipe (Google)
- Apache 2.0 — no restrictions relevant to this project.

## Bottom line

**Treat any transcription output from this app, and this project as a
whole, as non-commercial / research use only**, unless you replace the
Auto-AVSR and AV-HuBERT weights with ones trained on data you have
commercial rights to.
