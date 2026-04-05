"""CLAP-based audio-text semantic similarity reward."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import laion_clap


class CLAPReward:
    """Computes cosine similarity between audio and text using CLAP.

    This is the primary reward signal from the ASR-TRA paper.
    Spearman rho = -0.431 with ground-truth WER.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def compute(self, audio: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """Compute CLAP similarity between audio and each text candidate.

        Args:
            audio: waveform tensor [T] at 16kHz
            texts: list of N transcription candidates

        Returns:
            rewards: tensor [N] of cosine similarities in [-1, 1]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio_embed = self.model.get_audio_embedding_from_data(
            audio.to(self.device), use_tensor=True
        )
        text_embed = self.model.get_text_embedding(texts, use_tensor=True)

        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        return (text_embed @ audio_embed.T).squeeze(-1)

    def __call__(self, audio: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.compute(audio, texts)
