# Experiment 1: GRPO -- Group Relative Policy Optimization

## Problem

ASR-TRA uses vanilla REINFORCE with a mean baseline to optimize the decoder prompt at test time. This has two known failure modes:

1. **High gradient variance** -- with only 4 candidates, the mean baseline is a noisy estimator, leading to unstable updates that sometimes *increase* WER (visible in the paper's own results: SUTA degrades on 3/8 noise types).
2. **No collapse prevention** -- a single bad gradient step can push the prompt into a degenerate state from which recovery is impossible within the one-step TTA budget.

## Hypothesis

Replacing REINFORCE with GRPO (Group Relative Policy Optimization, adapted from DeepSeek-R1) will yield more stable and effective TTA by:

- **Normalized advantages**: Zero-mean, unit-variance within the candidate group (vs. simple mean subtraction). This makes gradient magnitudes independent of reward scale.
- **Gradient clipping**: Caps the max update norm to 1.0, preventing catastrophic single-step drift.
- **KL regularization**: Penalizes divergence from the pre-adaptation policy, keeping the model in a safe region.
- **Token-level advantages**: Distributes the sequence reward across individual token positions, giving denser gradient signal.

## Method

The GRPO loss for a group of N candidates is:

```
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
L_pg = -(1/N) * sum(advantages[i] * log_prob(y_i))
L_kl = (1/N) * sum(KL(pi_current || pi_ref))
L    = L_pg + beta * L_kl
```

Where `beta` (kl_coeff) controls the regularization strength. The token-level variant applies the advantage to each token's log-prob individually rather than the sequence sum.

## Ablations

| Variant | Token-level? | KL coeff | Description |
|---------|:---:|---:|---|
| `reinforce` | N/A | N/A | Baseline reproduction |
| `grpo_sequence` | No | 0.01 | GRPO with sequence-level advantage |
| `grpo_token` | Yes | 0.01 | GRPO with per-token advantage |
| `grpo_high_kl` | Yes | 0.10 | More conservative (stronger regularization) |

## How to Run

```bash
# Quick test (50 samples, ~15 min on GPU)
uv run python experiments/exp1_grpo.py --max-samples 50

# Full run (200 samples per noise type)
uv run python experiments/exp1_grpo.py --config configs/exp1_grpo.yaml

# CPU mode (slow)
uv run python experiments/exp1_grpo.py --max-samples 20 --device cpu
```

## What to Look For

1. **Mean WER**: GRPO variants should have equal or lower WER than REINFORCE.
2. **Variance**: GRPO should have lower WER variance across noise types (more consistent improvement).
3. **KL loss**: Should stay small (< 0.1). If it's large, the adaptation is drifting too far.
4. **Token-level vs sequence**: Token-level should help more on longer utterances.

## Expected Results

- `grpo_token` should improve WER by 0.5-1.5% absolute over `reinforce`
- `grpo_high_kl` may underperform `grpo_token` (too conservative) but should never degrade from baseline
- All GRPO variants should have lower WER standard deviation across samples

## Output

Results are saved to `results/exp1_grpo/` as JSON files, one per method-noise combination.
