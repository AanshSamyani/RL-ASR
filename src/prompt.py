"""Learnable decoder prompt for Whisper test-time adaptation."""

import torch
import torch.nn as nn


class LearnablePrompt(nn.Module):
    """Soft prompt prepended to Whisper decoder input embeddings.

    Following ASR-TRA, this acts as a causal intervention do(P) on the
    decoder's generation process.
    """

    def __init__(self, length: int = 4, embed_dim: int = 384):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.randn(1, length, embed_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        """Return prompt embeddings [1, L, D]."""
        return self.prompt

    def reset(self):
        """Re-initialize prompt to random values."""
        nn.init.normal_(self.prompt, std=0.02)

    def clone_state(self) -> torch.Tensor:
        """Snapshot current prompt for later restoration."""
        return self.prompt.data.clone()

    def restore_state(self, state: torch.Tensor):
        """Restore prompt from a previous snapshot."""
        self.prompt.data.copy_(state)
