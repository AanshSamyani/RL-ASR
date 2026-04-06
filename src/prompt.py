"""Learnable decoder prompt for Whisper test-time adaptation."""

from __future__ import annotations

import torch
from torch import nn


class LearnablePrompt(nn.Module):
    """Soft prompt prepended to Whisper decoder input embeddings.

    Following ASR-TRA, this acts as a causal intervention do(P) on the
    decoder's generation process.
    """

    def __init__(self, length: int = 4, embed_dim: int = 384) -> None:
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        # Zero-init so the prompt is a no-op before adaptation
        self.prompt = nn.Parameter(torch.zeros(1, length, embed_dim))

    def forward(self) -> torch.Tensor:
        """Return prompt embeddings [1, L, D]."""
        return self.prompt

    def reset(self) -> None:
        """Re-initialize prompt to zeros."""
        nn.init.zeros_(self.prompt)

    def clone_state(self) -> torch.Tensor:
        """Snapshot current prompt for later restoration."""
        return self.prompt.data.clone()

    def restore_state(self, state: torch.Tensor) -> None:
        """Restore prompt from a previous snapshot."""
        self.prompt.data.copy_(state)
