"""Single-sample test-time adaptation (ASR-TRA baseline reproduction)."""

from __future__ import annotations

import torch

from src.adaptation.base import BaseAdapter


class SingleSampleAdapter(BaseAdapter):
    """Per-sample TTA: adapt, decode, then restore all parameters.

    Flow:
    1. Baseline: whisper.decode() with original weights
    2. Candidates: stochastic decode with prompt (for RL gradients)
    3. RL step: update decoder weights + prompt
    4. Final: whisper.decode() with UPDATED weights
    5. Keep whichever (baseline vs adapted) has higher reward
    6. Restore all parameters
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

        # Step 1: Baseline with original weights
        audio_features = self.model.encode(mel)
        baseline_text = self.model.decode_greedy(audio_features)

        # Step 2: Generate candidates (prompt + temperature, for RL)
        candidates = self.model.generate_candidates(
            audio_features,
            n_candidates=self.n_candidates,
            temp_range=self.temp_range,
        )
        candidate_texts = [c["text"] for c in candidates]

        # Step 3: Compute rewards + RL update
        reward_result = self.reward_fn(audio, candidate_texts)
        rewards = reward_result["reward"] if isinstance(reward_result, dict) else reward_result

        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)
        info = self.rl_optimizer.step(candidates, rewards)

        # Step 4: Final decode with UPDATED decoder weights (via whisper.decode)
        adapted_text = self.model.decode_greedy(audio_features)

        # Step 5: Keep whichever is better by reward
        comparison = self.reward_fn(audio, [baseline_text, adapted_text])
        comp_rewards = (
            comparison["reward"] if isinstance(comparison, dict) else comparison
        )
        baseline_reward = comp_rewards[0].item()
        adapted_reward = comp_rewards[1].item()

        if adapted_reward > baseline_reward:
            final_text = adapted_text
            info["selected"] = "adapted"
        else:
            final_text = baseline_text
            info["selected"] = "baseline"

        info["baseline_reward"] = baseline_reward
        info["adapted_reward"] = adapted_reward

        # Step 6: Restore
        self.model.restore_state()

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "adapted_text": adapted_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }
