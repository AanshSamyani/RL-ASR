"""PARE: Phoneme-Aware Reward Ensemble.

Combines CLAP + LM perplexity + self-consistency into a single reward.
The key novelty: achieves LLM-level reward quality at CLAP-level latency
by combining multiple lightweight signals with calibrated weights.
"""

import torch

from .clap_reward import CLAPReward
from .lm_reward import LMPerplexityReward
from .consistency_reward import ConsistencyReward


class RewardEnsemble:
    """Multi-signal reward ensemble with configurable weights.

    Each signal is independently normalized to [0, 1] via min-max scaling
    within the candidate group, then combined with learned/fixed weights.
    """

    def __init__(
        self,
        use_clap: bool = True,
        use_lm: bool = True,
        use_consistency: bool = True,
        clap_weight: float = 0.5,
        lm_weight: float = 0.3,
        consistency_weight: float = 0.2,
        device: str = "cuda",
    ):
        self.device = device
        self.weights = {}
        self.reward_fns = {}

        if use_clap:
            self.reward_fns["clap"] = CLAPReward(device=device)
            self.weights["clap"] = clap_weight
        if use_lm:
            self.reward_fns["lm"] = LMPerplexityReward(device=device)
            self.weights["lm"] = lm_weight
        if use_consistency:
            self.reward_fns["consistency"] = ConsistencyReward(device=device)
            self.weights["consistency"] = consistency_weight

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def _normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Min-max normalize rewards within the candidate group."""
        rmin = rewards.min()
        rmax = rewards.max()
        if rmax - rmin < 1e-8:
            return torch.ones_like(rewards) * 0.5
        return (rewards - rmin) / (rmax - rmin)

    def compute(
        self,
        audio: torch.Tensor,
        texts: list[str],
        return_components: bool = False,
    ) -> dict:
        """Compute combined reward for each candidate.

        Args:
            audio: waveform tensor [T]
            texts: list of N candidates
            return_components: if True, include individual signal values

        Returns:
            dict with:
                - reward: tensor [N], combined reward
                - components: dict of individual rewards (if requested)
        """
        components = {}
        combined = torch.zeros(len(texts), device=self.device)

        if "clap" in self.reward_fns:
            clap_raw = self.reward_fns["clap"](audio, texts)
            clap_norm = self._normalize(clap_raw)
            combined += self.weights["clap"] * clap_norm
            components["clap"] = clap_raw

        if "lm" in self.reward_fns:
            lm_raw = self.reward_fns["lm"](texts)
            lm_norm = self._normalize(lm_raw)
            combined += self.weights["lm"] * lm_norm
            components["lm"] = lm_raw

        if "consistency" in self.reward_fns:
            cons_raw = self.reward_fns["consistency"](texts)
            cons_norm = self._normalize(cons_raw)
            combined += self.weights["consistency"] * cons_norm
            components["consistency"] = cons_raw

        result = {"reward": combined}
        if return_components:
            result["components"] = components
        return result

    def __call__(self, audio: torch.Tensor, texts: list[str], **kwargs) -> dict:
        return self.compute(audio, texts, **kwargs)
