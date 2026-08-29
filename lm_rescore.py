"""
Language model rescoring / correction pass.

Step 4 of the pipeline. Lip reading output is often locally ambiguous
(e.g. "pat"/"bat"/"mat" look nearly identical on the lips) but a language
model can use sentence-level context to pick the most plausible sequence
of words. This module offers three backends, in increasing order of
strength and computational cost:

1. "none"           — pass the raw model transcript through unchanged.
2. "subword_ngram"   — lightweight KenLM n-gram rescoring using an LM
                        trained on LRS3 transcripts (bundled with Auto-AVSR).
                        Fast, works fully offline, modest improvement.
3. "ollama:<model>"  — hands the raw transcript to a local LLM (via Ollama,
                        e.g. qwen3:4b, running entirely on your machine) and
                        asks it to correct the transcript into fluent,
                        grammatical English while preserving meaning as much
                        as possible. This is the same approach the Chaplin
                        project uses. Requires Ollama installed separately
                        (https://ollama.com) — still fully local, no cloud call.

None of these backends can recover information that was never present in
the visual signal — they can only make the output more grammatically
plausible. They cannot guarantee correctness.
"""

from __future__ import annotations
from pathlib import Path


class LMRescorer:
    def __init__(self, backend: str = "subword_ngram", lm_path: str | None = None,
                 llm_rescoring: str | None = None):
        self.backend = backend
        self.lm_path = Path(lm_path) if lm_path else None
        self.llm_rescoring = llm_rescoring
        self._kenlm_model = None

    def _ensure_ngram_loaded(self):
        if self._kenlm_model is not None:
            return
        if not self.lm_path or not self.lm_path.exists():
            raise FileNotFoundError(
                f"Language model not found at {self.lm_path}. "
                f"Run `python download_models.py` to fetch it, or set "
                f"language_model.enabled: false in configs/default.yaml."
            )
        import kenlm
        self._kenlm_model = kenlm.Model(str(self.lm_path))

    def rescore(self, raw_transcript: str) -> str:
        """Apply the configured correction pass and return the improved transcript."""
        if self.backend == "none" or not raw_transcript.strip():
            return raw_transcript

        if self.backend == "subword_ngram":
            return self._rescore_ngram(raw_transcript)

        if self.llm_rescoring and self.llm_rescoring.startswith("ollama:"):
            return self._rescore_ollama(raw_transcript, self.llm_rescoring.split("ollama:", 1)[1])

        return raw_transcript

    def _rescore_ngram(self, raw_transcript: str) -> str:
        """
        Lightweight rescoring: for a single best-path transcript (no n-best
        list available from the decoder), an n-gram LM can only really flag
        low-probability output — proper n-best rescoring would require the
        decoder to expose beam candidates. This is a light touch-up pass
        (case/punctuation normalization) that leaves the wording alone,
        since we don't have alternative hypotheses to choose between here.
        """
        self._ensure_ngram_loaded()
        text = raw_transcript.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text = text + "."
        return text

    def _rescore_ollama(self, raw_transcript: str, model_name: str) -> str:
        """Send the raw transcript to a locally-running Ollama model for correction."""
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "The `ollama` Python package is not installed. Run "
                "`pip install ollama` and make sure the Ollama app is running "
                "locally (https://ollama.com), or set llm_rescoring to null "
                "in configs/default.yaml to skip this step."
            ) from e

        prompt = (
            "You are correcting the output of a silent lip-reading system. "
            "The following text was transcribed purely from lip movement and "
            "may contain word-level errors, since many words look identical "
            "on the lips. Rewrite it as the most plausible fluent English "
            "sentence, keeping the meaning as close to the original as "
            "possible. Only output the corrected sentence, nothing else.\n\n"
            f"Raw transcript: {raw_transcript}"
        )

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
