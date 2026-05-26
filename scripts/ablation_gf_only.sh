#!/bin/bash
#SBATCH --partition=YOUR_PARTITION  # adjust to your cluster
#SBATCH --gres=gpu:1        # 80GB VRAM required (H100/H200/A100)
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --job-name=ablation_gf_only
#SBATCH --output=logs/ablation_gf_only_%j.log
#SBATCH --error=logs/ablation_gf_only_err_%j.log

# Submit from the repo root: sbatch scripts/ablation_gf_only.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$REPO_ROOT/code"
DATA_DIR="$REPO_ROOT/data"
CKPT_ROOT="$REPO_ROOT/checkpoints"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

mkdir -p "$REPO_ROOT/logs"

[ -f .venv/bin/activate ] && source .venv/bin/activate

export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COMPOSE_DATA_DIR="$DATA_DIR"
export COMPOSE_CKPT_DIR="$CKPT_ROOT"

# Ablation: Gf encoder only — use only the formal theorem graph encoder (enc2),
# disabling the paper graph encoder (enc1). Tests contribution of the paper-graph modality.
RESUME="$CKPT_ROOT/encoder_only.pt"
TRAIN_CKPT_DIR="$CKPT_ROOT/ablation_gf_only_${TIMESTAMP}"
mkdir -p "$TRAIN_CKPT_DIR"

cd "$CODE_DIR"
python3 train_dual.py \
    --data "$DATA_DIR/dual_training_samples_v7_clean.jsonl" \
    --batch_size 8 \
    --epochs 30 \
    --lr 2e-5 \
    --checkpoint_dir "$TRAIN_CKPT_DIR" \
    --resume "$RESUME" \
    --enc2_only_resume \
    --no_graph \
    --freeze_decoder_epochs 0 \
    --decoder_lr 5e-7