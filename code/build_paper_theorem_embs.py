"""
Build E5 embeddings for paper theorem statements and create lookup files.

Reads:
  paper_theorems_filtered.jsonl              -- 732K theorem statements from arXiv papers
  dual_training_samples_rich.jsonl           -- 52K training samples (for filtering)
  paper_theorems_mathlib_emb_matches_1.jsonl -- 712K paper_stmt → mathlib theorem matches

Writes (to s2orc/):
  paper_theorem_embs.npy            [N_stmts, 1024] float32, L2-normalized
  paper_theorem_stmt_to_idx.json    {stmt_id: row_index}
  paper_theorem_by_paper.json       {paper_id: [stmt_id, ...]}
  paper_thm_to_mathlib.json         {stmt_id: mathlib_theorem_name}  (top-1 match)

Usage:
  python3 build_paper_theorem_embs.py [--batch_size 128]
"""

import json
import argparse
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get('COMPOSE_DATA_DIR', _REPO_ROOT / 'data'))
TRAINING_DATA      = BASE / "dual_training_samples_v3.jsonl"
PAPER_THMS         = BASE / "paper_theorems_filtered.jsonl"
MATHLIB_MATCHES    = BASE / "paper_theorems_mathlib_emb_matches_1.jsonl"
MODEL_NAME         = "intfloat/e5-large-v2"

OUT_EMBS           = BASE / "paper_theorem_embs.npy"
OUT_STMT_TO_IDX    = BASE / "paper_theorem_stmt_to_idx.json"
OUT_BY_PAPER       = BASE / "paper_theorem_by_paper.json"
OUT_TO_MATHLIB     = BASE / "paper_thm_to_mathlib.json"


def average_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_embeddings(texts, tokenizer, model, batch_size, device, max_length=512):
    all_embs = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        embs = average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embs = F.normalize(embs, p=2, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Step 1: Collect target_arxiv_ids from training data ───────────────────
    logger.info("Loading training data to get target_arxiv_ids...")
    training_paper_ids = set()
    with open(TRAINING_DATA) as f:
        for line in f:
            sample = json.loads(line)
            arxiv_id = sample.get("target_arxiv_id", "")
            if arxiv_id:
                training_paper_ids.add(arxiv_id)
    logger.info(f"  Unique target papers in training data: {len(training_paper_ids):,}")

    # ── Step 2: Load paper theorems filtered to training papers ───────────────
    logger.info("Loading paper_theorems_filtered.jsonl...")
    stmt_to_text = {}      # stmt_id -> text
    by_paper = {}          # paper_id -> [stmt_id, ...]
    total_seen = 0
    matched_papers = 0

    with open(PAPER_THMS) as f:
        for line in f:
            total_seen += 1
            rec = json.loads(line)
            paper_id = rec.get("paper_id", "")
            if paper_id not in training_paper_ids:
                continue
            stmt_id = rec["stmt_id"]
            text = rec.get("text", "").strip()
            if not text:
                continue
            if paper_id not in by_paper:
                by_paper[paper_id] = []
                matched_papers += 1
            by_paper[paper_id].append(stmt_id)
            stmt_to_text[stmt_id] = text

    logger.info(f"  Total records seen: {total_seen:,}")
    logger.info(f"  Matched papers: {matched_papers:,} / {len(training_paper_ids):,} ({100*matched_papers/len(training_paper_ids):.1f}%)")
    logger.info(f"  Unique theorem statements: {len(stmt_to_text):,}")

    # ── Step 3: Build ordered index ───────────────────────────────────────────
    all_stmt_ids = sorted(stmt_to_text.keys())
    stmt_to_idx = {sid: i for i, sid in enumerate(all_stmt_ids)}
    texts = [f"passage: {stmt_to_text[sid]}" for sid in all_stmt_ids]  # E5 prefix
    logger.info(f"  Total texts to embed: {len(texts):,}")

    # ── Step 4: Build paper_thm_to_mathlib from matches file ─────────────────
    logger.info("Loading paper_theorems_mathlib_emb_matches_1.jsonl for top-1 matches...")
    thm_to_mathlib = {}
    total_match_records = 0
    covered = 0

    with open(MATHLIB_MATCHES) as f:
        for line in f:
            total_match_records += 1
            rec = json.loads(line)
            stmt_id = rec.get("stmt_id", "")
            if stmt_id not in stmt_to_idx:
                continue  # not in our filtered set
            matches = rec.get("matches", [])
            if not matches:
                continue
            # Sort by rank ascending, take top-1
            top = sorted(matches, key=lambda m: m.get("rank", 999))[0]
            name = top.get("name", "")
            if name:
                thm_to_mathlib[stmt_id] = name
                covered += 1

    logger.info(f"  Match records processed: {total_match_records:,}")
    logger.info(f"  Stmt_ids with mathlib match: {covered:,} / {len(stmt_to_idx):,} ({100*covered/max(len(stmt_to_idx),1):.1f}%)")

    # ── Step 5: Load E5 and compute embeddings ─────────────────────────────────
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    logger.info("Model loaded. Computing embeddings...")

    embs = compute_embeddings(texts, tokenizer, model, args.batch_size, device, args.max_length)
    logger.info(f"  Embeddings shape: {embs.shape}")

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    np.save(OUT_EMBS, embs)
    logger.info(f"  Saved: {OUT_EMBS}  ({embs.nbytes / 1e6:.1f} MB)")

    with open(OUT_STMT_TO_IDX, "w") as f:
        json.dump(stmt_to_idx, f)
    logger.info(f"  Saved: {OUT_STMT_TO_IDX}")

    with open(OUT_BY_PAPER, "w") as f:
        json.dump(by_paper, f)
    logger.info(f"  Saved: {OUT_BY_PAPER}")

    with open(OUT_TO_MATHLIB, "w") as f:
        json.dump(thm_to_mathlib, f)
    logger.info(f"  Saved: {OUT_TO_MATHLIB}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
