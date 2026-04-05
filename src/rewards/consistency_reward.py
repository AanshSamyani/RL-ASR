"""Self-consistency reward based on agreement among candidates.

If multiple stochastic decodings agree, the transcription is likely correct.
This is a FREE reward signal (no external model) that measures consensus
among candidates via normalized edit distance.
"""

from __future__ import annotations

import torch

from src.utils import compute_edit_distance


class ConsistencyReward:
    """Pairwise agreement reward among transcription candidates."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def compute(self, texts: list[str]) -> torch.Tensor:
        """Compute consistency reward for each candidate.

        Args:
            texts: list of N transcription candidates

        Returns:
            rewards: tensor [N] in [0, 1], higher = more consistent
        """
        n = len(texts)
        if n <= 1:
            return torch.ones(n, device=self.device)

        similarities = torch.zeros(n, n, device=self.device)
        for i in range(n):
            similarities[i, i] = 1.0
            for j in range(i + 1, n):
                max_len = max(len(texts[i]), len(texts[j]), 1)
                edit_dist = compute_edit_distance(texts[i], texts[j])
                sim = 1.0 - edit_dist / max_len
                similarities[i, j] = sim
                similarities[j, i] = sim

        mask = 1.0 - torch.eye(n, device=self.device)
        return (similarities * mask).sum(dim=1) / mask.sum(dim=1)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.compute(texts)
