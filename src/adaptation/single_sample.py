"""Single-sample test-time adaptation (ASR-TRA baseline reproduction)."""

from __future__ import annotations

import torch

from src.adaptation.base import BaseAdapter


class SingleSampleAdapter(BaseAdapter):
    """Per-sample TTA: adapt, decode, then restore all parameters.

    Reproduces the original ASR-TRA approach where each utterance
    is treated independently with full parameter restoration.
    """

    def adapt_and_decode(
        self, mel: torch.Tensor, audio: torch.Tensor
    ) -> dict:
        """Adapt on a single sample, then restore parameters."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0).to(self.device)
        else:
            mel = mel.to(self.device)

        self.model.save_state()

        # Baseline decode (greedy, no prompt)
        audio_features = self.model.encode(mel)
        baseline_text = self.model.decode_greedy(audio_features)

        # Generate candidates with prompt + stochastic decoding
        candidates = self.model.generate_candidates(
            audio_features,
            n_candidates=self.n_candidates,
            temp_range=self.temp_range,
        )
        candidate_texts = [c["text"] for c in candidates]

        # Compute rewards
        reward_result = self.reward_fn(audio, candidate_texts)
        rewards = reward_result["reward"] if isinstance(reward_result, dict) else reward_result

        # RL update
        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)

        info = self.rl_optimizer.step(candidates, rewards)

        # Final decode with updated parameters + prompt (Algorithm 1 step 9)
        final_text = self.model.decode_greedy_with_prompt(audio_features)

        # Restore parameters
        self.model.restore_state()

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }
