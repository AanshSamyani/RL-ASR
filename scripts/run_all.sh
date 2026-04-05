#!/bin/bash
# Run all experiments sequentially.
# Usage: bash scripts/run_all.sh [--max-samples N] [--device DEVICE]

set -e

MAX_SAMPLES="${1:-100}"
DEVICE="${2:-cuda}"

echo "============================================"
echo "ASR-TRA++ Full Experiment Suite"
echo "Max samples per dataset: $MAX_SAMPLES"
echo "Device: $DEVICE"
echo "============================================"

# Quick sanity check
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import whisper; print(f'Whisper loaded')"

echo ""
echo ">>> [0/4] Running baseline reproduction..."
python experiments/run_baseline.py \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee results/baseline.log

echo ""
echo ">>> [1/4] Experiment 1: GRPO vs REINFORCE..."
python experiments/exp1_grpo.py \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee results/exp1_grpo.log

echo ""
echo ">>> [2/4] Experiment 2: PARE Reward Ensemble..."
python experiments/exp2_pare.py \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee results/exp2_pare.log

echo ""
echo ">>> [3/4] Experiment 3: OPPA Persistent Prompt..."
python experiments/exp3_oppa.py \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee results/exp3_oppa.log

echo ""
echo ">>> [4/4] Combined experiment..."
python experiments/exp_combined.py \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    2>&1 | tee results/exp_combined.log

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results saved to results/"
echo "============================================"
