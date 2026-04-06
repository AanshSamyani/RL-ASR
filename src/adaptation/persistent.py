"""OPPA: Online Persistent Prompt Adaptation.

NOVELTY: Instead of resetting the prompt after each sample, accumulate
knowledge via EMA across correlated samples. Real audio streams have
temporal coherence (same speaker, same noise) -- this exploits that.

The model decoder weights are still restored per-sample (stability),
but the prompt retains an EMA of successful adaptations.
"""

from __future__ import annotations

import torch

from src.adaptation.base import BaseAdapter


class PersistentPromptAdapter(BaseAdapter):
    """Online persistent prompt adaptation with EMA accumulation.

    Key design:
    - Prompt persists across samples via EMA
    - Model decoder weights still reset per-sample
    - Selective accumulation: only update when reward improved
    - Warmup: faster decay initially, slower later
    - Optional domain grouping (separate prompts per noise/speaker)
    """

    def __init__(
        self,
        *args,
        ema_decay: float = 0.9,
        warmup_samples: int = 5,
        warmup_decay: float = 0.5,
        use_domain_grouping: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.warmup_samples = warmup_samples
        self.warmup_decay = warmup_decay
        self.use_domain_grouping = use_domain_grouping

        self._ema_prompt = self.model.prompt.clone_state()
        self._sample_count = 0
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
    ) -> None:
        """Update EMA prompt, only if adaptation helped."""
        if reward_improvement <= 0:
            return

        decay = self._get_current_decay(domain)

        if domain and self.use_domain_grouping:
            old = self._domain_prompts[domain]
            self._domain_prompts[domain] = decay * old + (1 - decay) * adapted_prompt
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
        if mel.dim() == 2:
            mel = mel.unsqueeze(0).to(self.device)
        else:
            mel = mel.to(self.device)

        self.model.save_state()

        # Initialize prompt from persistent EMA
        ema_prompt = self._get_prompt_for_domain(domain)
        self.model.prompt.restore_state(ema_prompt)

        # Baseline decode
        audio_features = self.model.encode(mel)
        baseline_text = self.model.decode_greedy(audio_features)

        # Baseline reward
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
        rewards = reward_result["reward"] if isinstance(reward_result, dict) else reward_result

        # RL update
        if hasattr(self.rl_optimizer, "store_reference_log_probs"):
            self.rl_optimizer.store_reference_log_probs(candidates)
        info = self.rl_optimizer.step(candidates, rewards)

        # Snapshot adapted prompt before restoration
        adapted_prompt = self.model.prompt.clone_state()

        # Final decode with updated parameters + prompt
        adapted_text = self.model.decode_greedy_with_prompt(audio_features)

        # Score baseline vs adapted
        comparison = self.reward_fn(audio, [baseline_text, adapted_text])
        comp_rewards = (
            comparison["reward"] if isinstance(comparison, dict) else comparison
        )
        adapted_reward = comp_rewards[1].item()
        reward_improvement = adapted_reward - baseline_reward

        # Conservative: keep baseline unless adapted is better
        if adapted_reward > baseline_reward:
            final_text = adapted_text
            info["selected"] = "adapted"
        else:
            final_text = baseline_text
            info["selected"] = "baseline"

        self._update_ema(adapted_prompt, reward_improvement, domain=domain)

        # Restore decoder weights
        self.model.restore_state()

        info["reward_improvement"] = reward_improvement
        info["ema_updated"] = reward_improvement > 0
        info["sample_count"] = self._sample_count

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "adapted_text": adapted_text,
            "candidate_texts": candidate_texts,
            "rewards": rewards,
            "info": info,
        }

    def reset(self) -> None:
        """Reset all persistent state."""
        self._ema_prompt = self.model.prompt.clone_state()
        self._sample_count = 0
        self._domain_prompts.clear()
        self._domain_counts.clear()
