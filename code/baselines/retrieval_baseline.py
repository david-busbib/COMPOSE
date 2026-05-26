"""
Retrieval Baseline
==================
No model training. Pure embedding similarity.

Steps:
  1. Build index: embed all training sample `phrase` fields with E5-large-v2
  2. At inference: embed query `phrase` → cosine similarity → return top-1 target_text

Usage:
  # Build index (run once)
  python3 retrieval_baseline.py --build --data /path/to/v7.jsonl --index /path/to/index.npz

  # Run inference + compare
  python3 retrieval_baseline.py --infer --index /path/to/index.npz --data /path/to/v7.jsonl --n 30 --lean_only
"""

import os, sys, json, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA  = os.path.join(BASE, 'dual_training_samples_v7.jsonl')
DEFAULT_INDEX = os.path.join(BASE, 'retrieval_index.npz')
E5_MODEL      = 'intfloat/e5-large-v2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--build',      action='store_true', help='Build the embedding index')
    p.add_argument('--infer',      action='store_true', help='Run retrieval inference')
    p.add_argument('--data',       default=DEFAULT_DATA)
    p.add_argument('--index',      default=DEFAULT_INDEX)
    p.add_argument('--n',          type=int, default=30, help='Number of test samples')
    p.add_argument('--skip',       type=int, default=0,  help='Skip first N samples')
    p.add_argument('--lean_only',  action='store_true',  help='Only lean_only / informal_with_lean targets')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--output',     default=None, help='Save results to JSON')
    return p.parse_args()


def load_e5():
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading E5 model: {E5_MODEL}")
    tok   = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    logger.info(f"E5 loaded on {device}")
    return tok, model, device


@torch.no_grad()
def embed_texts(texts, tok, model, device, batch_size=256, prefix='passage'):
    """Embed list of texts → numpy [N, 1024], L2-normalized."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {t}" for t in texts[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


def build_index(data_path, index_path, batch_size):
    logger.info(f"Reading dataset: {data_path}")
    phrases, targets, arxiv_ids, sources = [], [], [], []
    with open(data_path) as f:
        for line in f:
            s = json.loads(line)
            phrases.append(s.get('phrase', '') or s.get('target_title', ''))
            targets.append(s.get('target_text', ''))
            arxiv_ids.append(s.get('target_arxiv_id', ''))
            sources.append(s.get('target_text_source', ''))
    logger.info(f"Loaded {len(phrases)} samples")

    tok, model, device = load_e5()
    embs = embed_texts(phrases, tok, model, device, batch_size)

    np.savez(
        index_path,
        embeddings = embs,
        targets    = np.array(targets,    dtype=object),
        arxiv_ids  = np.array(arxiv_ids,  dtype=object),
        sources    = np.array(sources,    dtype=object),
        phrases    = np.array(phrases,    dtype=object),
    )
    logger.info(f"Index saved: {index_path}  shape={embs.shape}")


def run_inference(data_path, index_path, n, skip, lean_only, output):
    logger.info(f"Loading index: {index_path}")
    idx = np.load(index_path, allow_pickle=True)
    index_embs    = idx['embeddings']   # [N, 1024]
    index_targets = idx['targets']
    index_ids     = idx['arxiv_ids']
    index_phrases = idx['phrases']
    logger.info(f"Index loaded: {index_embs.shape[0]} samples")

    # Load test samples
    test_samples = []
    with open(data_path) as f:
        skipped = 0
        for line in f:
            s = json.loads(line)
            if lean_only and s.get('target_text_source') not in ('lean_only', 'informal_with_lean'):
                continue
            if skipped < skip:
                skipped += 1
                continue
            test_samples.append(s)
            if len(test_samples) >= n:
                break
    logger.info(f"Test samples: {len(test_samples)} (lean_only={lean_only})")

    tok, model, device = load_e5()

    # Embed test phrases
    test_phrases = [s.get('phrase', '') or s.get('target_title', '') for s in test_samples]
    test_embs = embed_texts(test_phrases, tok, model, device, batch_size=64)  # [n, 1024]

    # Cosine similarity (embeddings already L2-normalized → dot product = cosine)
    index_t = torch.tensor(index_embs, dtype=torch.float32)
    query_t = torch.tensor(test_embs,  dtype=torch.float32)
    scores  = query_t @ index_t.T  # [n, N]

    results = []
    print("\n" + "="*70)
    print("RETRIEVAL BASELINE RESULTS")
    print("="*70)

    for i, sample in enumerate(test_samples):
        gt         = sample.get('target_text', '')
        arxiv_id   = sample.get('target_arxiv_id', '')
        src        = sample.get('target_text_source', '')

        top_idx    = scores[i].argmax().item()
        top_score  = scores[i][top_idx].item()
        retrieved  = index_targets[top_idx]
        ret_id     = index_ids[top_idx]

        # Avoid returning the exact same sample
        if ret_id == arxiv_id:
            # get second best
            scores_copy = scores[i].clone()
            scores_copy[top_idx] = -1
            top_idx   = scores_copy.argmax().item()
            top_score = scores[i][top_idx].item()
            retrieved = index_targets[top_idx]
            ret_id    = index_ids[top_idx]

        print(f"\n[{i+1}] arxiv={arxiv_id}  src={src}  sim={top_score:.3f}")
        print(f"  GT        : {gt[:250]}")
        print(f"  RETRIEVED : {retrieved[:250]}")
        print(f"  (from arxiv={ret_id})")

        results.append({
            'arxiv_id':      arxiv_id,
            'source':        src,
            'ground_truth':  gt,
            'retrieved':     str(retrieved),
            'retrieved_from': str(ret_id),
            'similarity':    float(top_score),
        })

    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output}")


def main():
    args = parse_args()
    if args.build:
        build_index(args.data, args.index, args.batch_size)
    if args.infer:
        run_inference(args.data, args.index, args.n, args.skip, args.lean_only, args.output)
    if not args.build and not args.infer:
        print("Specify --build and/or --infer")


if __name__ == '__main__':
    main()
