#!/bin/bash
#SBATCH --job-name=compose_prep
#SBATCH --output=logs/compose_prep_%j.log
#SBATCH --partition=YOUR_PARTITION  # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00

# Submit from the repo root: sbatch scripts/prepare_data.sh
# Builds data/train_target_thmft_embs.npz (theorem-finetuned target embeddings for training).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
METRICS_DIR="$REPO_ROOT/code/baselines/metrics"

mkdir -p "$REPO_ROOT/logs"

[ -f .venv/bin/activate ] && source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

python3 - << PYEOF
"""
Build train_target_thmft_embs.npz:
  For each unique target_arxiv_id in training data, embed target_text
  with the thm_ft model. Saved keyed by arxiv_id so train_dual.py can
  look up the embedding for each training sample's target paper.
"""
import json, os, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from collections import defaultdict

DATA_DIR   = '$DATA_DIR'
METRICS_DIR= '$METRICS_DIR'

V7         = os.path.join(DATA_DIR, 'dual_training_samples_v7_clean.jsonl')
THM_FT     = os.path.join(METRICS_DIR, 'theorem_emb_model')
BASE_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
OUT        = os.path.join(DATA_DIR, 'train_target_thmft_embs.npz')

if os.path.exists(OUT):
    d = np.load(OUT)
    print(f"Already exists: {len(d['arxiv_ids'])} papers")
    exit(0)

# ── Collect target texts per arxiv_id ──────────────────────────────
print("Loading training data...", flush=True)
pid_to_texts = defaultdict(list)
n_samples = 0
with open(V7) as f:
    for line in f:
        d = json.loads(line)
        pid = d.get('target_arxiv_id', '').split('v')[0]
        txt = d.get('target_text', '')
        if pid and txt and len(txt.strip()) > 20:
            if len(pid_to_texts[pid]) < 8:  # keep up to 8 texts per paper
                pid_to_texts[pid].append(txt.strip()[:512])
        n_samples += 1

print(f"  {n_samples} samples, {len(pid_to_texts)} unique target papers", flush=True)

# ── Load thm_ft model ───────────────────────────────────────────────
print("Loading thm_ft model...", flush=True)
device = 'cuda'
tok = AutoTokenizer.from_pretrained(THM_FT)
tok.pad_token = tok.eos_token
base = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, THM_FT).to(device).eval()
print("  Loaded.", flush=True)

@torch.no_grad()
def embed_texts(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        seq_len = enc['attention_mask'].sum(dim=1) - 1
        emb = h[torch.arange(len(batch), device=device), seq_len]
        out.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)

# ── Embed all target papers ─────────────────────────────────────────
print(f"Embedding {len(pid_to_texts)} papers...", flush=True)
arxiv_ids = sorted(pid_to_texts.keys())
paper_embs = []   # one embedding per paper (max-pool over its target texts)

for i, pid in enumerate(arxiv_ids):
    if i % 200 == 0:
        print(f"  {i}/{len(arxiv_ids)}", flush=True)
    texts = pid_to_texts[pid]
    embs = embed_texts(texts)          # (K, 4096)
    # max-pool across texts (same as pool construction)
    paper_emb = embs.max(axis=0)
    norm = np.linalg.norm(paper_emb)
    if norm > 0:
        paper_emb /= norm
    paper_embs.append(paper_emb)

paper_embs = np.stack(paper_embs, axis=0).astype(np.float32)
print(f"  Done. Shape: {paper_embs.shape}", flush=True)

np.savez(OUT, arxiv_ids=np.array(arxiv_ids), embeddings=paper_embs)
print(f"Saved -> {OUT}", flush=True)
PYEOF