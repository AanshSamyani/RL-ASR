"""Vanilla REINFORCE with mean baseline — reproduction of ASR-TRA."""

import torch


class REINFORCE:
    """REINFORCE policy gradient optimizer as used in ASR-TRA.

    Uses mean reward as baseline: adv_i = r_i - mean(r)
    Loss = -sum(adv_i * log P(y_i | s, p))
    """

    def __init__(
        self,
        base_lr: float = 1e-5,
        prompt_lr_scale: float = 100.0,
    ):
        self.base_lr = base_lr
        self.prompt_lr_scale = prompt_lr_scale
        self.optimizer = None

    def setup_optimizer(self, param_groups: list[dict]) -> None:
        """Initialize Adam optimizer with per-group learning rates."""
        opt_groups = []
        for group in param_groups:
            lr = self.base_lr * group.get("lr_scale", 1.0)
            opt_groups.append({"params": group["params"], "lr": lr})
        self.optimizer = torch.optim.Adam(opt_groups)

    def compute_loss(
        self,
        candidates: list[dict],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REINFORCE loss with mean baseline.

        Args:
            candidates: list of N dicts, each with 'total_log_prob'
            rewards: tensor [N] of scalar rewards

        Returns:
            loss: scalar tensor
        """
        baseline = rewards.mean()
        advantages = rewards - baseline

        loss = torch.tensor(0.0, device=rewards.device, requires_grad=True)
        for i, cand in enumerate(candidates):
            # total_log_prob is sum of per-token log probs
            loss = loss + (-advantages[i] * cand["total_log_prob"])

        return loss / len(candidates)

    def step(
        self,
        candidates: list[dict],
        rewards: torch.Tensor,
    ) -> dict:
        """One REINFORCE update step.

        Returns:
            dict with loss value and advantage stats.
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(candidates, rewards)
        loss.backward()
        self.optimizer.step()

        baseline = rewards.mean()
        return {
            "loss": loss.item(),
            "mean_reward": baseline.item(),
            "reward_std": rewards.std().item(),
        }
