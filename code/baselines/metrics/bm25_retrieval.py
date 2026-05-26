"""
BM25 retrieval baseline on canonical 200 subset.
Query = anchor paper title + abstract (from subgraph root node)
Pool = 14K future papers title + abstract
"""
import json, numpy as np, re
from rank_bm25 import BM25Okapi

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DATA      = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
M          = os.path.dirname(os.path.abspath(__file__))
SUBGRAPHS  = os.path.join(_DATA, 'new_subgraphs.jsonl')
SOURCE     = os.path.join(_DATA, 'arxiv_theorems_final.jsonl')
POOL_FILE  = M + '/future_paper_theorem_embs_finetuned.npz'
SG_MAP     = M + '/future_paper_embs_sg_map.json'
CANON      = M + '/canonical_200_subset.json'
EVAL_JSON  = M + '/eval_goai_2026-04-29_21-24-13.json'

# Load 14K pool paper ids + their text
print("Loading 14K pool + paper metadata...")
pool_meta = {}
with open(SOURCE) as f:
    for line in f:
        r = json.loads(line)
        aid = r['arxiv_id'].split('v')[0]
        if aid not in pool_meta:
            pool_meta[aid] = (r.get('title','') or '') + ' ' + (r.get('abstract','') or '')

pd_data = np.load(POOL_FILE)
pool_pids = list(pd_data['paper_ids'])
p2i = {p: i for i, p in enumerate(pool_pids)}
print(f"  {len(pool_pids)} papers in pool")

# Build BM25 index over pool
def tokenize(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()

print("Building BM25 index...")
corpus = [tokenize(pool_meta.get(pid, '')) for pid in pool_pids]
bm25 = BM25Okapi(corpus)
print("  Done.")

# Load anchor text (gt_text = anchor theorem in context) per subgraph
print("Loading anchor texts (gt_text from eval)...")
sg_map    = json.load(open(SG_MAP))
canon_sgs = set(json.load(open(CANON))['subgraph_ids'])
eval_data = json.load(open(EVAL_JSON))
sg2text   = {r['subgraph_id']: r['gt_text'] for r in eval_data['results']
             if r['subgraph_id'] in canon_sgs and r.get('gt_text','').strip()}
print(f"  Found anchor text for {len(sg2text)}/{len(canon_sgs)} canonical subgraphs")

# Score
ks = [1, 5, 10, 20, 50, 100]
hits = {k: [] for k in ks}
tgt_sims, neg_sims, exp_sims = [], [], []

np.random.seed(42)
print("Scoring on canonical 200 subset...")
for sg in canon_sgs:
    tidxs = [p2i[a] for a in sg_map.get(sg, []) if a in p2i]
    if not tidxs:
        continue
    query_text = sg2text.get(sg, '')
    if not query_text.strip():
        continue
    scores = bm25.get_scores(tokenize(query_text))
    # normalize scores to [0,1] range for comparability
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = scores - s_min

    ranked = np.argsort(-scores)
    t = set(tidxs)
    for k in ks:
        hits[k].append(float(bool(t & set(ranked[:k].tolist()))))

    neg_idxs = [i for i in range(len(pool_pids)) if i not in t]
    tgt_sims.append(float(scores_norm[tidxs].mean()))
    neg_sims.append(float(scores_norm[neg_idxs].mean()) if neg_idxs else 0.0)
    exp_sims.append(float(scores_norm.mean()))

n = len(tgt_sims)
tgt  = np.mean(tgt_sims)
neg  = np.mean(neg_sims)
exp  = np.mean(exp_sims)
gap  = tgt - neg

print(f"\n{'='*65}")
print(f"BM25 RETRIEVAL — canonical 200 subset, 14K pool  (n={n})")
print(f"{'='*65}")
print('  '.join(f'H@{k}' for k in ks))
print('  '.join('%.3f'%np.mean(hits[k]) for k in ks))
print(f"\nTgt-Sim: {tgt:.4f}  Exp-Sim: {exp:.4f}  Neg-Sim: {neg:.4f}  Gap: {gap:.4f}")
