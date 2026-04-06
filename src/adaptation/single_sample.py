"""Single-sample test-time adaptation (ASR-TRA baseline reproduction)."""

from __future__ import annotations

import torch

from src.adaptation.base import BaseAdapter


class SingleSampleAdapter(BaseAdapter):
    """Per-sample TTA: adapt, decode, then restore all parameters.

    Safety: defaults to baseline. Only uses adapted output if it has
    higher reward than baseline (conservative selection).
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

        # Compute rewards for candidates
        reward_result = self.reward_fn(audio, candidate_texts)
        rewards = reward_result["reward"] if isinstance(reward_result, dict) else reward_result

        # RL update
        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)

        info = self.rl_optimizer.step(candidates, rewards)

        # Final decode with adapted prompt + decoder
        adapted_text = self.model.decode_greedy_with_prompt(audio_features)

        # Score baseline vs adapted
        comparison = self.reward_fn(audio, [baseline_text, adapted_text])
        comp_rewards = (
            comparison["reward"] if isinstance(comparison, dict) else comparison
        )
        baseline_reward = comp_rewards[0].item()
        adapted_reward = comp_rewards[1].item()

        # Conservative: keep baseline unless adapted is strictly better
        if adapted_reward > baseline_reward:
            final_text = adapted_text
            info["selected"] = "adapted"
        else:
            final_text = baseline_text
            info["selected"] = "baseline"

        info["baseline_reward"] = baseline_reward
        info["adapted_reward"] = adapted_reward

        # Restore parameters
        self.model.restore_state()

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "adapted_text": adapted_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }
