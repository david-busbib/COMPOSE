#!/bin/bash
# Download COMPOSE model checkpoint from HuggingFace
# Usage: bash scripts/download_checkpoint.sh [target_dir]

TARGET="${1:-checkpoints}"
mkdir -p "$TARGET"

python3 -c "import huggingface_hub" 2>/dev/null || pip install huggingface-hub

huggingface-cli download TheNisso/compose-checkpoint \
    --repo-type model \
    --local-dir "$TARGET"

echo "Checkpoint downloaded to $TARGET"
echo "Set COMPOSE_CKPT_DIR=$TARGET or export in your environment."
