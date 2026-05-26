#!/bin/bash
#SBATCH --job-name=compose_train
#SBATCH --output=logs/compose_train_%j.log
#SBATCH --error=logs/compose_train_err_%j.log
#SBATCH --partition=YOUR_PARTITION  # adjust to your cluster
#SBATCH --gres=gpu:1        # 80GB VRAM required (H100/H200/A100)
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00

# Submit from the repo root: sbatch scripts/train.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$REPO_ROOT/code"
DATA_DIR="$REPO_ROOT/data"
CKPT_ROOT="$REPO_ROOT/checkpoints"

mkdir -p "$REPO_ROOT/logs"

[ -f .venv/bin/activate ] && source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make data/checkpoints visible to train_dual.py path defaults
export COMPOSE_DATA_DIR="$DATA_DIR"
export COMPOSE_CKPT_DIR="$CKPT_ROOT"

RESUME="$CKPT_ROOT/best_model.pt"
TRAIN_CKPT_DIR="$CKPT_ROOT/training_output_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$TRAIN_CKPT_DIR"

cd "$CODE_DIR"
python3 train_dual.py \
    --data "$DATA_DIR/dual_training_samples_v7_clean.jsonl" \
    --batch_size 6 \
    --grad_accum 1 \
    --epochs 51 \
    --lr 2e-5 \
    --checkpoint_dir "$TRAIN_CKPT_DIR" \
    --freeze_decoder_epochs 0 \
    --decoder_lr 5e-7 \
    --resume "$RESUME" \
    --thm_ft_gen_weight 1.0
