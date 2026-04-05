"""Language model perplexity reward using GPT-2 small.

GPT-2 small (124M params) provides a fast linguistic fluency signal (~10ms/candidate).
Unlike the full LLM reward in ASR-TRA (7-9x latency), this adds negligible overhead.
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class LMPerplexityReward:
    """Computes normalized negative perplexity as a fluency reward.

    Lower perplexity = more fluent = higher reward.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda") -> None:
        self.device = device
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def compute(self, texts: list[str]) -> torch.Tensor:
        """Compute negative log-perplexity for each text.

        Args:
            texts: list of N transcription candidates

        Returns:
            rewards: tensor [N] -- higher = more fluent
        """
        rewards = []
        for text in texts:
            if not text.strip():
                rewards.append(torch.tensor(-100.0, device=self.device))
                continue

            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            input_ids = inputs["input_ids"]

            outputs = self.model(**inputs, labels=input_ids)
            neg_log_ppl = -outputs.loss
            rewards.append(neg_log_ppl)

        return torch.stack(rewards)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.compute(texts)
