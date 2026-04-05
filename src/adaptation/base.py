"""Base adapter interface for test-time adaptation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.whisper_wrapper import WhisperWithPrompt


class BaseAdapter(ABC):
    """Abstract base class for TTA adapters."""

    def __init__(
        self,
        model: WhisperWithPrompt,
        reward_fn: object,
        rl_optimizer: object,
        n_candidates: int = 4,
        temp_range: tuple[float, float] = (0.4, 0.6),
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.rl_optimizer = rl_optimizer
        self.n_candidates = n_candidates
        self.temp_range = temp_range
        self.device = device

    @abstractmethod
    def adapt_and_decode(
        self, mel: torch.Tensor, audio: torch.Tensor
    ) -> dict:
        """Run test-time adaptation on a single sample.

        Args:
            mel: mel spectrogram [80, T]
            audio: raw waveform [T] (for reward computation)

        Returns:
            dict with text, baseline_text, rewards, info
        """
        ...
