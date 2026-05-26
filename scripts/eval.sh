#!/bin/bash
#SBATCH --job-name=compose_eval
#SBATCH --output=logs/compose_eval_%j.log
#SBATCH --partition=YOUR_PARTITION  # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Submit from the repo root: sbatch scripts/eval.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$REPO_ROOT/code"
CKPT_ROOT="$REPO_ROOT/checkpoints"

mkdir -p "$REPO_ROOT/logs"

[ -f .venv/bin/activate ] && source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CKPT="$CKPT_ROOT/best_model.pt"
CANON="$CODE_DIR/baselines/metrics/canonical_200_subset.json"

cd "$CODE_DIR/baselines/metrics"

echo "=== COMPOSE full-graph eval on canonical 200-sample subset ==="
python3 eval_future.py \
    --model full_graph \
    --checkpoint "$CKPT" \
    --subgraph_ids_file "$CANON" \
    --use_root_title