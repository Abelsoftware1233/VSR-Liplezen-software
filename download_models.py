"""
One-time setup script: downloads existing pretrained VSR model weights and
clones the upstream inference code they depend on. This script does NOT
train anything — it only fetches checkpoints that were already trained by
their original authors.

Usage:
    python download_models.py                  # Auto-AVSR only (recommended default)
    python download_models.py --with-av-hubert  # also fetch AV-HuBERT for ensembling
    python download_models.py --with-lm         # also fetch the subword n-gram language model

Sources:
    - Auto-AVSR weights & code: https://github.com/mpc001/auto_avsr (Apache 2.0 code license;
      weights carry LRS3 dataset terms — non-commercial, see docs/LICENSES.md)
    - AV-HuBERT weights & code: https://github.com/facebookresearch/av_hubert
    - Language model: bundled with the Auto-AVSR / Chaplin release on Hugging Face Hub
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
THIRD_PARTY = ROOT / "third_party"
WEIGHTS = ROOT / "model_weights"

AUTO_AVSR_REPO = "https://github.com/mpc001/auto_avsr.git"
AV_HUBERT_REPO = "https://github.com/facebookresearch/av_hubert.git"

# Hugging Face Hub repo used by the Chaplin project to host ready-to-use
# Auto-AVSR checkpoints (avoids needing the original gated dataset access
# just to get inference weights).
HF_AUTO_AVSR_REPO_ID = "amanvirparhar/chaplin-models"


def run(cmd: list[str], cwd: Path | None = None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_if_missing(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"[skip] {target_dir} already exists")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def download_auto_avsr():
    print("\n=== Auto-AVSR (primary model) ===")
    clone_if_missing(AUTO_AVSR_REPO, THIRD_PARTY / "auto_avsr")

    dest = WEIGHTS / "auto_avsr"
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    print("Downloading pretrained checkpoint from Hugging Face Hub...")
    for filename in ["lrs3_v_wer19.1.pth", "lrs3_v_wer19.1.ini"]:
        local_path = hf_hub_download(
            repo_id=HF_AUTO_AVSR_REPO_ID,
            filename=filename,
            local_dir=str(dest),
        )
        print(f"  -> {local_path}")

    print("Auto-AVSR ready.")


def download_av_hubert():
    print("\n=== AV-HuBERT (optional secondary model) ===")
    clone_if_missing(AV_HUBERT_REPO, THIRD_PARTY / "av_hubert")

    fairseq_dir = THIRD_PARTY / "av_hubert" / "fairseq"
    if not fairseq_dir.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/pytorch/fairseq.git", str(fairseq_dir)])
        run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=fairseq_dir)

    dest = WEIGHTS / "av_hubert"
    dest.mkdir(parents=True, exist_ok=True)
    print(
        "NOTE: AV-HuBERT's official LRS3-finetuned checkpoint requires "
        "accepting Meta's model license via their form:\n"
        "  https://github.com/facebookresearch/av_hubert#model-checkpoints\n"
        "This script cannot auto-accept that license for you. Download "
        "'large_lrs3.pt' manually from the link above and place it at:\n"
        f"  {dest / 'large_lrs3.pt'}"
    )


def download_language_model():
    print("\n=== Subword n-gram language model ===")
    dest = WEIGHTS / "language_models" / "lm_en_subword"
    dest.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=HF_AUTO_AVSR_REPO_ID,
        filename="lm_en_subword.arpa",
        local_dir=str(dest.parent),
    )
    print(f"  -> {local_path}")
    print("Language model ready.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-av-hubert", action="store_true",
                         help="Also download AV-HuBERT for ensemble transcription")
    parser.add_argument("--with-lm", action="store_true", default=True,
                         help="Also download the n-gram language model (default: on)")
    parser.add_argument("--skip-lm", action="store_true",
                         help="Skip language model download")
    args = parser.parse_args()

    WEIGHTS.mkdir(exist_ok=True)
    THIRD_PARTY.mkdir(exist_ok=True)

    download_auto_avsr()

    if args.with_av_hubert:
        download_av_hubert()

    if args.with_lm and not args.skip_lm:
        download_language_model()

    print("\nAll done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
