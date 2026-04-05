"""Self-consistency reward based on agreement among candidates.

Key insight: If multiple stochastic decodings agree, the transcription is
likely correct. This is a FREE reward signal (no external model needed)
that measures consensus among candidates via normalized edit distance.
"""

import torch

from ..utils import compute_edit_distance


class ConsistencyReward:
    """Computes pairwise agreement reward among transcription candidates.

    For each candidate, the reward is the mean normalized similarity
    to all other candidates. High agreement = high confidence.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute(self, texts: list[str]) -> torch.Tensor:
        """Compute consistency reward for each candidate.

        Args:
            texts: list of N transcription candidates

        Returns:
            rewards: tensor [N] in [0, 1], higher = more consistent with peers
        """
        n = len(texts)
        if n <= 1:
            return torch.ones(n, device=self.device)

        # Compute pairwise normalized edit distance
        similarities = torch.zeros(n, n, device=self.device)
        for i in range(n):
            for j in range(i + 1, n):
                max_len = max(len(texts[i]), len(texts[j]), 1)
                edit_dist = compute_edit_distance(texts[i], texts[j])
                sim = 1.0 - edit_dist / max_len
                similarities[i, j] = sim
                similarities[j, i] = sim
            similarities[i, i] = 1.0  # self-similarity

        # Mean similarity to all OTHER candidates
        mask = 1.0 - torch.eye(n, device=self.device)
        rewards = (similarities * mask).sum(dim=1) / mask.sum(dim=1)

        return rewards

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.compute(texts)
