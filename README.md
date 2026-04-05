# ASR-TRA++: Improved Test-Time Reinforcement Adaptation for ASR

Building on [ASR-TRA](https://arxiv.org/abs/2603.05231) (Fang et al., 2026), which uses test-time RL with CLAP rewards to adapt Whisper to noisy/accented speech. We propose three orthogonal improvements targeting the paper's key weaknesses.

## Three Improvement Directions

### Experiment 1 — GRPO: Group Relative Policy Optimization

**Problem**: ASR-TRA uses vanilla REINFORCE with a mean baseline, which has high gradient variance and can produce unstable TTA updates.

**Solution**: Replace REINFORCE with GRPO (Group Relative Policy Optimization), adapted from DeepSeek-R1 for the ASR-TTA setting:
- **Normalized advantages**: Zero-mean, unit-variance within the candidate group (vs. simple mean subtraction)
- **Gradient clipping**: Prevents catastrophic single-step updates
- **KL regularization**: Keeps adapted model close to the original, preventing prompt collapse
- **Token-level advantages**: Denser gradient signal by distributing the reward across individual tokens

**Expected outcome**: More stable adaptation that consistently improves WER (ASR-TRA's REINFORCE sometimes degrades, e.g., SUTA's WER *increases* on several noise types).

### Experiment 2 — PARE: Phoneme-Aware Reward Ensemble

**Problem**: CLAP's correlation with true WER is weak (Spearman ρ = −0.431). The LLM reward helps but adds 7–9× latency overhead.

**Solution**: Combine multiple lightweight reward signals:
| Signal | What it measures | Latency overhead |
|--------|-----------------|------------------|
| CLAP similarity | Audio-text semantic alignment | ~20ms (baseline) |
| GPT-2 small perplexity | Linguistic fluency | ~10ms |
| Self-consistency | Agreement among candidates | ~1ms (free) |

Each signal is min-max normalized within the candidate group, then combined with calibrated weights. The ensemble should achieve LLM-level reward quality at CLAP-level speed.

**Expected outcome**: Higher reward–WER correlation → better gradient signal → lower WER, without latency regression.

### Experiment 3 — OPPA: Online Persistent Prompt Adaptation

**Problem**: ASR-TRA resets all parameters after each utterance. This wastes adaptation knowledge when processing correlated audio (same speaker, same noise environment).

**Solution**: Maintain a running EMA (exponential moving average) of the soft prompt across samples:
- **Selective accumulation**: Only update EMA when adaptation improved the reward (skip harmful updates)
- **Warmup schedule**: Lower EMA decay initially for fast adaptation, higher decay later for stability
- **Domain grouping** (optional): Maintain separate prompt per noise type or speaker accent
- **Decoder reset**: Model weights still reset per-sample (only the prompt persists)

**Expected outcome**: Later samples in a correlated stream benefit from accumulated domain knowledge, with WER improving over time within each domain.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Tested with**: Python 3.10+, PyTorch 2.1+, CUDA 12.x, single NVIDIA GPU (RTX 6000 Ada or equivalent).

### Data Preparation

```bash
bash scripts/setup_data.sh
```

This downloads:
1. **LibriSpeech test-other** (~300MB) — clean speech, we add noise programmatically
2. **MS-SNSD** (~50MB) — noise corpus (8 noise types at 10dB SNR)
3. **L2-Arctic** (manual download) — accented English from 6 L1 backgrounds

L2-Arctic requires manual download from [https://psi.engr.tamu.edu/l2-arctic-corpus/](https://psi.engr.tamu.edu/l2-arctic-corpus/) — extract to `data/l2arctic_release_v5/`. You can run LibriSpeech experiments without it.

## Running Experiments

### Quick validation (small sample, ~10 min each)

```bash
# Individual experiments
python experiments/exp1_grpo.py --max-samples 50 --device cuda
python experiments/exp2_pare.py --max-samples 50 --device cuda
python experiments/exp3_oppa.py --max-samples 50 --device cuda
```

### Full experiment suite

```bash
# Run all experiments with 200 samples per dataset
bash scripts/run_all.sh 200 cuda

# Or run everything with full datasets
bash scripts/run_all.sh 0 cuda
```

### Individual experiments with custom configs

```bash
# Baseline reproduction (ASR-TRA as described in paper)
python experiments/run_baseline.py --config configs/base.yaml --device cuda

# Exp 1: GRPO ablation
python experiments/exp1_grpo.py --config configs/exp1_grpo.yaml --device cuda

# Exp 2: Reward ensemble ablation
python experiments/exp2_pare.py --config configs/exp2_pare.yaml --device cuda

# Exp 3: Persistent prompt ablation
python experiments/exp3_oppa.py --config configs/exp3_oppa.yaml --device cuda

# Combined: best of all three + all pairwise combos
python experiments/exp_combined.py --max-samples 100 --device cuda
```

### CPU-only mode (slow but works)

```bash
python experiments/exp1_grpo.py --max-samples 20 --device cpu
```

## Project Structure

```
├── configs/
│   ├── base.yaml              # Shared base configuration
│   ├── exp1_grpo.yaml         # GRPO ablation configs
│   ├── exp2_pare.yaml         # Reward ensemble configs
│   └── exp3_oppa.yaml         # Persistent prompt configs
├── src/
│   ├── whisper_wrapper.py     # Whisper + prompt injection + stochastic decode
│   ├── prompt.py              # Learnable soft prompt module
│   ├── utils.py               # WER computation, logging, config loading
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
│       ├── librispeech_noisy.py   # LibriSpeech + MS-SNSD noise
│       └── l2arctic.py            # L2-Arctic accented English
├── experiments/
│   ├── run_baseline.py        # ASR-TRA reproduction
│   ├── exp1_grpo.py           # GRPO vs REINFORCE
│   ├── exp2_pare.py           # Reward ensemble ablation
│   ├── exp3_oppa.py           # Persistent prompt ablation
│   └── exp_combined.py        # All improvements combined
├── scripts/
│   ├── setup_data.sh          # Data download script
│   └── run_all.sh             # Run full experiment suite
├── results/                   # Auto-created, JSON result files
├── requirements.txt
└── README.md
```

## Expected Results

Based on analysis of ASR-TRA's reported numbers and the nature of our improvements:

| Method | Expected WER (LibriSpeech noisy) | Expected WER (L2-Arctic) |
|--------|----------------------------------|--------------------------|
| Whisper (no TTA) | ~32.7% | ~32.1% |
| ASR-TRA (baseline) | ~28.6% | ~28.2% |
| + GRPO | ~27.5% (more stable) | ~27.5% |
| + PARE | ~26.8% (better reward) | ~26.5% |
| + OPPA | ~27.0% (temporal gains) | ~26.0% |
| All combined | ~25.5% | ~24.5% |

## Key Hyperparameters to Tune

If results are underwhelming, try adjusting:

| Parameter | Default | Range to search |
|-----------|---------|-----------------|
| GRPO `kl_coeff` | 0.01 | [0.001, 0.1] |
| GRPO `clip_eps` | 0.2 | [0.1, 0.3] |
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
