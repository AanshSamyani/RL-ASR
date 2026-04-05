# Experiment 2: PARE -- Phoneme-Aware Reward Ensemble

## Problem

ASR-TRA's primary reward signal is CLAP audio-text cosine similarity, which has a weak Spearman correlation of rho = -0.431 with ground-truth WER. The paper shows that adding a full LLM reward improves WER substantially (33.42% -> 30.24% on Whisper-Base) but at 7-9x latency cost.

The weak reward means the RL optimization often follows noise rather than signal, limiting adaptation quality.

## Hypothesis

We can build a composite reward with near-LLM quality at near-CLAP speed by combining three lightweight signals:

| Signal | What it captures | Latency | Model |
|--------|-----------------|---------|-------|
| **CLAP** | Audio-text semantic alignment | ~20ms | CLAP (pretrained) |
| **GPT-2 perplexity** | Linguistic fluency / grammar | ~10ms | GPT-2 small (124M) |
| **Self-consistency** | Candidate agreement (edit distance) | <1ms | None (free) |

Each signal captures an orthogonal aspect of transcription quality:
- CLAP catches semantic drift (wrong content)
- GPT-2 catches disfluent outputs (grammatical errors, repetition)
- Consistency catches outliers (if 3/4 candidates agree and 1 differs, the outlier is likely wrong)

## Method

Each signal is computed independently and min-max normalized within the candidate group to [0, 1]. The combined reward is:

```
R = w_clap * norm(CLAP) + w_lm * norm(GPT2_PPL) + w_cons * norm(CONSISTENCY)
```

The normalization ensures signals are commensurable regardless of their raw scales.

## Ablations

| Config | CLAP | LM | Consistency | Weights |
|--------|:---:|:---:|:---:|---|
| `clap_only` | Yes | No | No | Baseline |
| `clap_lm` | Yes | Yes | No | 0.6 / 0.4 |
| `clap_consistency` | Yes | No | Yes | 0.7 / 0.3 |
| `pare_full` | Yes | Yes | Yes | 0.5 / 0.3 / 0.2 |
| `pare_lm_heavy` | Yes | Yes | Yes | 0.3 / 0.5 / 0.2 |

## How to Run

```bash
# Quick test (50 samples, ~20 min -- GPT-2 loads once, cached thereafter)
uv run python experiments/exp2_pare.py --max-samples 50

# Full run
uv run python experiments/exp2_pare.py --config configs/exp2_pare.yaml

# CPU mode
uv run python experiments/exp2_pare.py --max-samples 20 --device cpu
```

## What to Look For

1. **Reward-WER correlation** (rho): The primary metric. CLAP alone is ~-0.43. PARE should achieve ~-0.55 to -0.65.
2. **Mean WER**: Better correlation should translate to lower WER.
3. **Latency**: PARE full should add ~10ms over CLAP-only (GPT-2 inference). Much less than the 4+ seconds for LLM reward.
4. **Self-consistency as free lunch**: `clap_consistency` should improve over `clap_only` with zero latency cost.

## Expected Results

- `pare_full` should improve WER by 1-2% absolute over `clap_only`
- `clap_consistency` should give ~0.5% improvement for free
- `pare_lm_heavy` may outperform `pare_full` if the LM signal is particularly strong
- All PARE variants should have significantly better reward-WER correlation than CLAP alone

## Output

Results are saved to `results/exp2_pare/` as JSON files, one per config-noise combination. The summary table includes Spearman rho for each configuration.
