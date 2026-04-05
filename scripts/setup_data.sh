#!/bin/bash
# Download LibriSpeech test-other for ASR-TRA++ experiments.
# Run from project root: bash scripts/setup_data.sh
#
# Only LibriSpeech test-other is required. Noise is added
# programmatically (Gaussian) so no noise corpus is needed.

set -e

DATA_ROOT="${1:-data}"
mkdir -p "$DATA_ROOT"

echo "============================================"
echo "ASR-TRA++ Data Setup"
echo "============================================"

# ---------------------------------------------------
# LibriSpeech test-other (~300MB)
# ---------------------------------------------------
echo ""
echo "Downloading LibriSpeech test-other..."
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

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Contents:"
ls -la "$DATA_ROOT/" 2>/dev/null || echo "  (empty)"
echo ""
echo "Noise is generated synthetically (Gaussian)."
echo "No additional downloads needed."
echo "============================================"
