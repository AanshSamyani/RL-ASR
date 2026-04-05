"""OPPA: Online Persistent Prompt Adaptation.

NOVELTY: Instead of resetting the prompt after each sample, we accumulate
knowledge via EMA across correlated samples. Real audio streams have
temporal coherence (same speaker, same noise environment) — this exploits
that structure for better domain-specific adaptation.

The model decoder weights are still restored per-sample (to prevent drift),
but the prompt retains an EMA of successful adaptations.
"""

import torch

from .base import BaseAdapter


class PersistentPromptAdapter(BaseAdapter):
    """Online persistent prompt adaptation with EMA accumulation.

    Key design choices:
    - Prompt persists across samples via EMA (exponential moving average)
    - Model decoder weights are still reset per-sample (stability)
    - Domain grouping: when metadata (noise type, speaker) is available,
      maintain separate prompts per domain
    - Warmup: first few samples use higher EMA momentum for fast adaptation
    """

    def __init__(
        self,
        *args,
        ema_decay: float = 0.9,
        warmup_samples: int = 5,
        warmup_decay: float = 0.5,
        use_domain_grouping: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.warmup_samples = warmup_samples
        self.warmup_decay = warmup_decay
        self.use_domain_grouping = use_domain_grouping

        # Running EMA prompt state
        self._ema_prompt = self.model.prompt.clone_state()
        self._sample_count = 0

        # Domain-specific prompts (optional)
        self._domain_prompts: dict[str, torch.Tensor] = {}
        self._domain_counts: dict[str, int] = {}

    def _get_current_decay(self, domain: str | None = None) -> float:
        """Adaptive decay: lower during warmup for faster initial adaptation."""
        count = (
            self._domain_counts.get(domain, 0)
            if domain and self.use_domain_grouping
            else self._sample_count
        )
        if count < self.warmup_samples:
            return self.warmup_decay
        return self.ema_decay

    def _get_prompt_for_domain(self, domain: str | None) -> torch.Tensor:
        """Get the appropriate prompt for the given domain."""
        if domain and self.use_domain_grouping:
            if domain not in self._domain_prompts:
                self._domain_prompts[domain] = self._ema_prompt.clone()
                self._domain_counts[domain] = 0
            return self._domain_prompts[domain]
        return self._ema_prompt

    def _update_ema(
        self,
        adapted_prompt: torch.Tensor,
        reward_improvement: float,
        domain: str | None = None,
    ):
        """Update EMA prompt, weighted by reward improvement.

        Only accumulate if adaptation actually helped (positive improvement).
        """
        if reward_improvement <= 0:
            return  # Skip negative adaptations

        decay = self._get_current_decay(domain)

        if domain and self.use_domain_grouping:
            old = self._domain_prompts[domain]
            self._domain_prompts[domain] = (
                decay * old + (1 - decay) * adapted_prompt
            )
            self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1
        else:
            self._ema_prompt = decay * self._ema_prompt + (1 - decay) * adapted_prompt

        self._sample_count += 1

    def adapt_and_decode(
        self,
        mel: torch.Tensor,
        audio: torch.Tensor,
        domain: str | None = None,
    ) -> dict:
        """Adapt with persistent prompt, then selectively update EMA."""
        mel = mel.unsqueeze(0).to(self.device) if mel.dim() == 2 else mel.to(self.device)

        # Save decoder state (will restore), but set prompt to EMA
        self.model.save_state()

        # Initialize prompt from persistent EMA
        ema_prompt = self._get_prompt_for_domain(domain)
        self.model.prompt.restore_state(ema_prompt)

        # Baseline decode
        audio_features = self.model.encode(mel)
        baseline_text = self.model.decode_greedy(audio_features)

        # Compute baseline reward for improvement tracking
        baseline_reward_result = self.reward_fn(audio, [baseline_text])
        if isinstance(baseline_reward_result, dict):
            baseline_reward = baseline_reward_result["reward"][0].item()
        else:
            baseline_reward = baseline_reward_result[0].item()

        # Generate candidates
        candidates = self.model.generate_candidates(
            audio_features,
            n_candidates=self.n_candidates,
            temp_range=self.temp_range,
        )
        candidate_texts = [c["text"] for c in candidates]

        # Compute rewards
        reward_result = self.reward_fn(audio, candidate_texts)
        if isinstance(reward_result, dict):
            rewards = reward_result["reward"]
        else:
            rewards = reward_result

        # RL update
        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)

        info = self.rl_optimizer.step(candidates, rewards)

        # Snapshot adapted prompt BEFORE restoration
        adapted_prompt = self.model.prompt.clone_state()

        # Final decode with updated parameters
        final_text = self.model.decode_greedy(audio_features)

        # Compute final reward to measure improvement
        final_reward_result = self.reward_fn(audio, [final_text])
        if isinstance(final_reward_result, dict):
            final_reward = final_reward_result["reward"][0].item()
        else:
            final_reward = final_reward_result[0].item()

        reward_improvement = final_reward - baseline_reward

        # Update EMA prompt (only if adaptation helped)
        self._update_ema(adapted_prompt, reward_improvement, domain=domain)

        # Restore decoder weights (but prompt will be reset to EMA on next call)
        self.model.restore_state()

        info["reward_improvement"] = reward_improvement
        info["ema_updated"] = reward_improvement > 0
        info["sample_count"] = self._sample_count

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }

    def reset(self):
        """Reset all persistent state."""
        self._ema_prompt = self.model.prompt.clone_state()
        self._sample_count = 0
        self._domain_prompts.clear()
        self._domain_counts.clear()
