"""CLAP-based audio-text semantic similarity reward."""

import torch
import torch.nn.functional as F
import laion_clap


class CLAPReward:
    """Computes cosine similarity between audio and text using CLAP.

    This is the primary reward signal from the ASR-TRA paper.
    Spearman rho = -0.431 with ground-truth WER (higher similarity = lower WER).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # loads default checkpoint
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
        # CLAP expects audio at 48kHz, so resample
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Get audio embedding
        audio_embed = self.model.get_audio_embedding_from_data(
            audio.to(self.device), use_tensor=True
        )  # [1, D]

        # Get text embeddings
        text_embed = self.model.get_text_embedding(
            texts, use_tensor=True
        )  # [N, D]

        # Cosine similarity
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        similarities = (text_embed @ audio_embed.T).squeeze(-1)  # [N]

        return similarities

    def __call__(self, audio: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.compute(audio, texts)
