# Experiment 3: OPPA -- Online Persistent Prompt Adaptation

## Problem

ASR-TRA resets all model parameters (including the learned prompt) after every single utterance. Each sample starts from scratch with a randomly initialized prompt. This wastes information: if the model successfully adapted to white noise on utterance #1, it throws that knowledge away before processing utterance #2 -- which may also have white noise.

In real-world deployment, audio comes in streams: same speaker, same room, same microphone, same background noise. There's strong temporal correlation that ASR-TRA ignores.

## Hypothesis

By persisting the soft prompt across samples via EMA (exponential moving average), later utterances benefit from accumulated domain knowledge, reducing the "cold start" cost of per-sample adaptation.

Key design choices:
- **Selective accumulation**: Only update the EMA when the adaptation actually improved the reward (prevents error propagation from bad adaptations).
- **Warmup schedule**: Use lower EMA decay initially (faster learning), higher decay later (stability).
- **Domain grouping**: Optionally maintain separate prompts per noise type or speaker, so different domains don't interfere.
- **Decoder reset**: Only the prompt persists. Model weights are still restored per-sample to prevent catastrophic drift.

## Method

After each utterance, the prompt update rule is:

```
if reward_improvement > 0:
    if sample_count < warmup_samples:
        decay = warmup_decay    # e.g., 0.5 (fast learning)
    else:
        decay = ema_decay       # e.g., 0.9 (stable)

    prompt_ema = decay * prompt_ema + (1 - decay) * prompt_adapted
```

With domain grouping, each noise type / speaker gets its own `prompt_ema`.

## Ablations

| Strategy | EMA Decay | Warmup | Domain Grouping | Description |
|----------|:---------:|:------:|:---:|---|
| `single_sample` | N/A | N/A | N/A | Baseline (full reset) |
| `persistent_ema09` | 0.9 | 5 / 0.5 | No | Conservative persistence |
| `persistent_ema07` | 0.7 | 5 / 0.3 | No | Aggressive persistence |
| `persistent_domain` | 0.9 | 3 / 0.5 | Yes | Per-noise-type prompts |

## How to Run

```bash
# Quick test (50 samples per noise type)
uv run python experiments/exp3_oppa.py --max-samples 50

# Full run
uv run python experiments/exp3_oppa.py --config configs/exp3_oppa.yaml

# CPU mode
uv run python experiments/exp3_oppa.py --max-samples 20 --device cpu
```

## What to Look For

1. **Temporal trend**: The key signal. For persistent adapters, the 2nd half of samples should have lower WER than the 1st half (the prompt has "warmed up"). For single-sample, there should be no trend.
2. **Mean WER**: Persistent should beat single-sample overall, especially for the domain-grouped variant.
3. **EMA update rate**: What fraction of samples trigger an EMA update? If too low (< 30%), the prompt barely changes. If too high (> 90%), there may not be enough filtering.
4. **Domain grouping**: Compare `persistent_domain` vs `persistent_ema09`. Domain grouping should help when noise types differ significantly.

## Expected Results

- `persistent_ema09` should improve WER by 0.5-1% over `single_sample`
- The temporal trend should be clearly visible: 2nd half WER improves by 0.5-2% vs 1st half
- `persistent_domain` should be the best variant when data comes from distinct noise domains
- `persistent_ema07` may be too aggressive and show instability on domain boundaries

## Output

Results are saved to `results/exp3_oppa/` as JSON files. Each entry includes `ema_updated` (bool) and `reward_improvement` (float) for persistent variants. The summary includes the temporal (1st half vs 2nd half) analysis.
