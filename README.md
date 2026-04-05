# ASR-TRA++: Improved Test-Time Reinforcement Adaptation for ASR

Building on [ASR-TRA](https://arxiv.org/abs/2603.05231) (Fang et al., 2026), which uses test-time RL with CLAP rewards to adapt Whisper to noisy/accented speech. We propose three orthogonal improvements targeting the paper's key weaknesses.

## Three Improvement Directions

### Exp 1 -- GRPO: Group Relative Policy Optimization

**Problem**: Vanilla REINFORCE with a mean baseline has high gradient variance. With only 4 candidates the baseline estimate is noisy, producing unstable updates that sometimes *increase* WER.

**Solution**: Replace REINFORCE with GRPO -- zero-mean unit-variance advantage normalization, gradient clipping, and KL regularization against the pre-adaptation policy. A token-level variant distributes the sequence reward across token positions for denser gradients.

See [`experiments/README_exp1_grpo.md`](experiments/README_exp1_grpo.md) for full details.

### Exp 2 -- PARE: Phoneme-Aware Reward Ensemble

**Problem**: CLAP's correlation with true WER is weak (Spearman rho = -0.431). The LLM reward helps but adds 7-9x latency.

**Solution**: Combine CLAP + GPT-2 small perplexity (~10ms) + self-consistency among candidates (free) into a multi-signal ensemble. Each signal is min-max normalized and weighted. Targets LLM-level reward quality at CLAP-level speed.

See [`experiments/README_exp2_pare.md`](experiments/README_exp2_pare.md) for full details.

### Exp 3 -- OPPA: Online Persistent Prompt Adaptation

**Problem**: ASR-TRA resets all parameters per utterance. This wastes adaptation knowledge when processing correlated audio (same noise environment).

**Solution**: Persist the soft prompt across samples via selective EMA -- only accumulate when adaptation improved the reward. Optional domain grouping maintains separate prompts per noise type.

See [`experiments/README_exp3_oppa.md`](experiments/README_exp3_oppa.md) for full details.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- GPU with CUDA (recommended; CPU works but is slow)

### Install dependencies

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Install dev tools (ruff linter)
uv sync --extra dev
```

### Lint

```bash
# Check
uv run ruff check .

# Auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .
```

### Download data

Only **LibriSpeech test-other** (~300MB) is required. Noise is added synthetically (Gaussian white/pink/brown) so no extra noise corpus is needed.

```bash
bash scripts/setup_data.sh
```

## Running Experiments

### Quick validation (~10-15 min per experiment on GPU)

```bash
uv run python experiments/exp1_grpo.py --max-samples 50
uv run python experiments/exp2_pare.py --max-samples 50
uv run python experiments/exp3_oppa.py --max-samples 50
```

### Full suite

```bash
# 200 samples per noise type, all experiments
bash scripts/run_all.sh 200 cuda

# Full dataset (all samples)
bash scripts/run_all.sh 0 cuda
```

### Individual experiments

```bash
# Baseline reproduction
uv run python experiments/run_baseline.py --max-samples 100

# Exp 1: GRPO ablation (4 RL variants x 3 noise types)
uv run python experiments/exp1_grpo.py --config configs/exp1_grpo.yaml

# Exp 2: Reward ensemble ablation (5 reward configs x 3 noise types)
uv run python experiments/exp2_pare.py --config configs/exp2_pare.yaml

# Exp 3: Persistent prompt ablation (4 strategies x 3 noise types)
uv run python experiments/exp3_oppa.py --config configs/exp3_oppa.yaml

# Combined: best of each + all pairwise combos (7 configurations)
uv run python experiments/exp_combined.py --max-samples 100
```

### CPU-only

```bash
uv run python experiments/exp1_grpo.py --max-samples 20 --device cpu
```

## Project Structure

```
.
├── pyproject.toml             # uv/ruff/project config
├── configs/
│   ├── base.yaml              # Shared defaults
│   ├── exp1_grpo.yaml         # GRPO ablation configs
│   ├── exp2_pare.yaml         # Reward ensemble configs
│   └── exp3_oppa.yaml         # Persistent prompt configs
├── src/
│   ├── whisper_wrapper.py     # Whisper + prompt injection + stochastic decode
│   ├── prompt.py              # Learnable soft prompt module
│   ├── utils.py               # WER, edit distance, logging, config
│   ├── rewards/
│   │   ├── clap_reward.py     # CLAP audio-text similarity
│   │   ├── lm_reward.py       # GPT-2 perplexity reward
│   │   ├── consistency_reward.py  # Self-consistency (edit distance)
│   │   └── ensemble.py        # PARE combined reward
│   ├── rl/
│   │   ├── reinforce.py       # Vanilla REINFORCE (baseline)
│   │   └── grpo.py            # GRPO optimizer (novel)
│   ├── adaptation/
│   │   ├── base.py            # Abstract adapter interface
│   │   ├── single_sample.py   # Per-sample TTA (baseline)
│   │   └── persistent.py      # OPPA persistent prompt (novel)
│   └── data/
│       └── librispeech_noisy.py   # LibriSpeech + Gaussian noise
├── experiments/
│   ├── run_baseline.py        # ASR-TRA reproduction
│   ├── exp1_grpo.py           # GRPO vs REINFORCE
│   ├── exp2_pare.py           # Reward ensemble ablation
│   ├── exp3_oppa.py           # Persistent prompt ablation
│   ├── exp_combined.py        # All improvements combined
│   ├── README_exp1_grpo.md    # Exp 1 detailed writeup
│   ├── README_exp2_pare.md    # Exp 2 detailed writeup
│   └── README_exp3_oppa.md    # Exp 3 detailed writeup
├── scripts/
│   ├── setup_data.sh          # Download LibriSpeech test-other
│   └── run_all.sh             # Run full experiment suite
└── results/                   # Auto-created, JSON result files
```

## Expected Results

| Method | Expected Mean WER (LibriSpeech noisy, 10dB) |
|--------|---:|
| Whisper (no TTA) | ~32.7% |
| ASR-TRA (REINFORCE + CLAP) | ~28.6% |
| + GRPO | ~27.5% |
| + PARE | ~26.8% |
| + OPPA | ~27.0% |
| All combined | ~25.5% |

## Key Hyperparameters

| Parameter | Default | Search range |
|-----------|---------|-------------|
| GRPO `kl_coeff` | 0.01 | [0.001, 0.1] |
| PARE `clap_weight` | 0.5 | [0.3, 0.7] |
| PARE `lm_weight` | 0.3 | [0.1, 0.5] |
| OPPA `ema_decay` | 0.9 | [0.7, 0.95] |
| OPPA `warmup_samples` | 5 | [3, 10] |
| `n_candidates` | 4 | [4, 8] |
| `temp_range` | [0.4, 0.6] | [0.3, 0.7] |

## Citation

```bibtex
@article{fang2026boosting,
  title={Boosting ASR Robustness via Test-Time Reinforcement Learning with Audio-Text Semantic Rewards},
  author={Fang, Linghan and Xie, Tianxin and Liu, Li},
  journal={arXiv preprint arXiv:2603.05231},
  year={2026}
}
```
