"""
Utilities for KenLM rescoring.
"""

from dataclasses import dataclass
from pathlib import Path


try:
    import kenlm
except ImportError:
    kenlm = None


@dataclass
class KenLMConfig:
    model_path: str | Path
    alpha: float = 0.5
    beta: float = 0.0


class KenLMScorer:
    def __init__(self, config: KenLMConfig):
        if kenlm is None:
            raise ImportError(
                "KenLM is not installed. Install it before using --kenlm-model."
            )

        self.config = config
        self.model_path = str(config.model_path)
        self.model = kenlm.Model(self.model_path)
        self.alpha = config.alpha
        self.beta = config.beta

    def score_text(self, text: str) -> float:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return float("-inf")
        return float(self.model.score(normalized, bos=True, eos=True))

    def combined_score(self, acoustic_score: float, text: str) -> float:
        lm_score = self.score_text(text)
        word_bonus = len(text.split())
        return acoustic_score + self.alpha * lm_score + self.beta * word_bonus
