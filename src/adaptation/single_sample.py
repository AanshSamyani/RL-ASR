"""Single-sample test-time adaptation (ASR-TRA baseline reproduction)."""

import torch

from .base import BaseAdapter


class SingleSampleAdapter(BaseAdapter):
    """Per-sample TTA: adapt, decode, then restore all parameters.

    This reproduces the original ASR-TRA approach where each utterance
    is treated independently with full parameter restoration after.
    """

    def adapt_and_decode(
        self, mel: torch.Tensor, audio: torch.Tensor
    ) -> dict:
        """Adapt on a single sample, then restore parameters."""
        mel = mel.unsqueeze(0).to(self.device) if mel.dim() == 2 else mel.to(self.device)

        # Save state for restoration
        self.model.save_state()

        # Step 1: Baseline decode (greedy, no prompt)
        audio_features = self.model.encode(mel)
        baseline_text = self.model.decode_greedy(audio_features)

        # Step 2: Generate candidates with prompt + stochastic decoding
        candidates = self.model.generate_candidates(
            audio_features,
            n_candidates=self.n_candidates,
            temp_range=self.temp_range,
        )
        candidate_texts = [c["text"] for c in candidates]

        # Step 3: Compute rewards
        reward_result = self.reward_fn(audio, candidate_texts)
        if isinstance(reward_result, dict):
            rewards = reward_result["reward"]
        else:
            rewards = reward_result

        # Step 4: RL update
        # For GRPO, store reference log probs before update
        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)

        info = self.rl_optimizer.step(candidates, rewards)

        # Step 5: Final decode with updated parameters
        final_text = self.model.decode_greedy(audio_features)

        # Step 6: Restore parameters
        self.model.restore_state()

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }
