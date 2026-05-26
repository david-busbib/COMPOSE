#!/bin/bash
# Download COMPOSE Mathlib corpus from HuggingFace (~3 GB)
# Required for full enc1+enc2 inference. Without this, only enc1 runs.
# Usage: bash scripts/download_mathlib.sh [target_dir]

TARGET="${1:-data/LeanDojo/leandojo_benchmark_4}"
mkdir -p "$TARGET/embeddings_corpus" "$TARGET/embeddings_deepseek" "$TARGET/processed"

HF_REPO="https://huggingface.co/datasets/TheNisso/compose-mathlib-corpus/resolve/main"

echo "Downloading Mathlib corpus to $TARGET (~3 GB)..."

for FILE in \
    "embeddings_corpus/embeddings.npy" \
    "embeddings_corpus/idx_to_name.json" \
    "embeddings_corpus/idx_to_statement.json" \
    "embeddings_deepseek/embeddings.npy" \
    "embeddings_deepseek/name_to_idx.json" \
    "processed/graph.json"
do
    OUT="$TARGET/$FILE"
    if [ -f "$OUT" ]; then
        echo "  Already exists: $FILE"
        continue
    fi
    echo "  Downloading $FILE..."
    curl -L --progress-bar "$HF_REPO/$FILE" -o "$OUT"
done

echo "Done. Set COMPOSE_DATA_DIR=$(dirname $(dirname $TARGET)) before running run_on_paper.py"
