#!/bin/bash
# Run all experiments sequentially.
# Usage: bash scripts/run_all.sh [MAX_SAMPLES] [GPU_ID]
#
# Examples:
#   bash scripts/run_all.sh 100 0       # 100 samples on GPU 0
#   bash scripts/run_all.sh 200 2       # 200 samples on GPU 2
#   bash scripts/run_all.sh 0 1         # full dataset on GPU 1

set -e

MAX_SAMPLES="${1:-100}"
GPU_ID="${2:-0}"

GPU_FLAG="--gpu $GPU_ID"

mkdir -p results

echo "============================================"
echo "ASR-TRA++ Full Experiment Suite"
echo "Max samples per noise type: $MAX_SAMPLES"
echo "GPU: $GPU_ID"
echo "============================================"

# Sanity check
CUDA_VISIBLE_DEVICES=$GPU_ID uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
uv run python -c "import whisper; print('Whisper OK')"

echo ""
echo ">>> [0/4] Baseline reproduction..."
uv run python experiments/run_baseline.py \
    --max-samples "$MAX_SAMPLES" \
    $GPU_FLAG \
    2>&1 | tee results/baseline.log

echo ""
echo ">>> [1/4] Experiment 1: GRPO vs REINFORCE..."
uv run python experiments/exp1_grpo.py \
    --max-samples "$MAX_SAMPLES" \
    $GPU_FLAG \
    2>&1 | tee results/exp1_grpo.log

echo ""
echo ">>> [2/4] Experiment 2: PARE Reward Ensemble..."
uv run python experiments/exp2_pare.py \
    --max-samples "$MAX_SAMPLES" \
    $GPU_FLAG \
    2>&1 | tee results/exp2_pare.log

echo ""
echo ">>> [3/4] Experiment 3: OPPA Persistent Prompt..."
uv run python experiments/exp3_oppa.py \
    --max-samples "$MAX_SAMPLES" \
    $GPU_FLAG \
    2>&1 | tee results/exp3_oppa.log

echo ""
echo ">>> [4/4] Combined experiment..."
uv run python experiments/exp_combined.py \
    --max-samples "$MAX_SAMPLES" \
    $GPU_FLAG \
    2>&1 | tee results/exp_combined.log

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results saved to results/"
echo "============================================"
