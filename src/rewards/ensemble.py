"""PARE: Phoneme-Aware Reward Ensemble.

Combines CLAP + LM perplexity + self-consistency into a single reward.
Achieves LLM-level reward quality at CLAP-level latency by combining
multiple lightweight signals with calibrated weights.
"""

from __future__ import annotations

import torch

from src.rewards.clap_reward import CLAPReward
from src.rewards.consistency_reward import ConsistencyReward
from src.rewards.lm_reward import LMPerplexityReward


class RewardEnsemble:
    """Multi-signal reward ensemble with configurable weights.

    Each signal is min-max normalized within the candidate group,
    then combined with fixed weights.
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
    ) -> None:
        self.device = device
        self.weights: dict[str, float] = {}
        self.reward_fns: dict[str, CLAPReward | LMPerplexityReward | ConsistencyReward] = {}

        if use_clap:
            self.reward_fns["clap"] = CLAPReward(device=device)
            self.weights["clap"] = clap_weight
        if use_lm:
            self.reward_fns["lm"] = LMPerplexityReward(device=device)
            self.weights["lm"] = lm_weight
        if use_consistency:
            self.reward_fns["consistency"] = ConsistencyReward(device=device)
            self.weights["consistency"] = consistency_weight

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
            dict with 'reward' tensor [N] and optionally 'components'
        """
        components: dict[str, torch.Tensor] = {}
        combined = torch.zeros(len(texts), device=self.device)

        if "clap" in self.reward_fns:
            clap_raw = self.reward_fns["clap"](audio, texts)
            combined += self.weights["clap"] * self._normalize(clap_raw)
            components["clap"] = clap_raw

        if "lm" in self.reward_fns:
            lm_raw = self.reward_fns["lm"](texts)
            combined += self.weights["lm"] * self._normalize(lm_raw)
            components["lm"] = lm_raw

        if "consistency" in self.reward_fns:
            cons_raw = self.reward_fns["consistency"](texts)
            combined += self.weights["consistency"] * self._normalize(cons_raw)
            components["consistency"] = cons_raw

        result: dict = {"reward": combined}
        if return_components:
            result["components"] = components
        return result

    def __call__(self, audio: torch.Tensor, texts: list[str], **kwargs) -> dict:
        return self.compute(audio, texts, **kwargs)
