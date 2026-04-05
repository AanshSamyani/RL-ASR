#!/bin/bash
# Download and prepare datasets for ASR-TRA++ experiments.
# Run from project root: bash scripts/setup_data.sh

set -e

DATA_ROOT="data"
mkdir -p "$DATA_ROOT"

echo "============================================"
echo "ASR-TRA++ Data Setup"
echo "============================================"

# ---------------------------------------------------
# 1. LibriSpeech test-other
# ---------------------------------------------------
echo ""
echo "[1/3] Downloading LibriSpeech test-other..."
LIBRI_DIR="$DATA_ROOT/LibriSpeech"
if [ -d "$LIBRI_DIR/test-other" ]; then
    echo "  Already exists, skipping."
else
    cd "$DATA_ROOT"
    wget -q --show-progress https://www.openslr.org/resources/12/test-other.tar.gz
    tar xzf test-other.tar.gz
    rm test-other.tar.gz
    cd ..
    echo "  Done."
fi

# ---------------------------------------------------
# 2. MS-SNSD noise corpus
# ---------------------------------------------------
echo ""
echo "[2/3] Downloading MS-SNSD noise corpus..."
NOISE_DIR="$DATA_ROOT/MS-SNSD"
if [ -d "$NOISE_DIR" ]; then
    echo "  Already exists, skipping."
else
    cd "$DATA_ROOT"
    git clone https://github.com/microsoft/MS-SNSD.git
    cd ..
    echo "  Done."
fi

# ---------------------------------------------------
# 3. L2-Arctic
# ---------------------------------------------------
echo ""
echo "[3/3] Downloading L2-Arctic corpus..."
L2_DIR="$DATA_ROOT/l2arctic_release_v5"
if [ -d "$L2_DIR" ]; then
    echo "  Already exists, skipping."
else
    echo "  L2-Arctic requires manual download from:"
    echo "  https://psi.engr.tamu.edu/l2-arctic-corpus/"
    echo ""
    echo "  After downloading, extract to: $DATA_ROOT/l2arctic_release_v5/"
    echo "  Expected structure:"
    echo "    $L2_DIR/<SPEAKER_ID>/annotation/*.txt"
    echo "    $L2_DIR/<SPEAKER_ID>/wav/*.wav"
    echo ""
    echo "  Alternatively, you can run experiments on LibriSpeech only."
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Directory structure:"
echo "  $DATA_ROOT/"
ls -la "$DATA_ROOT/" 2>/dev/null || echo "  (no files yet)"
echo "============================================"
