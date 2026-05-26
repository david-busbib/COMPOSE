import json, numpy as np

M = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

# Load pool
sg_map = json.load(open(f'{M}/future_paper_embs_sg_map.json'))
pd2 = np.load(f'{M}/future_paper_theorem_embs_finetuned.npz')
pe = pd2['embeddings'].astype(np.float32)
pids = list(pd2['paper_ids'])
offs = [int(x) for x in pd2['offsets']]
n_p = len(pids)
pe2 = np.zeros((n_p, pe.shape[1]), dtype=np.float32)
for pi in range(n_p):
    s, e = offs[pi], offs[pi+1]
    if e > s: pe2[pi] = pe[s:e].max(axis=0)
nr = np.linalg.norm(pe2, axis=1, keepdims=True); nr[nr==0] = 1; pe2 /= nr
p2i = {p: i for i, p in enumerate(pids)}
print(f"Pool: {n_p} papers", flush=True)

def ln(a):
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n==0] = 1; return a/n

top10  = lambda s: set(np.argsort(s)[::-1][:10].tolist())
top100 = lambda s: set(np.argsort(s)[::-1][:100].tolist())

# Build oracle (best of ep30 + apr13 by tgt_sim) — vectorized
r_ep30  = json.load(open(f'{M}/eval_full_graph_2026-04-29_06-39-25.json'))['results']
r_apr13 = json.load(open(f'{M}/eval_full_graph_2026-04-13_11-05-55.json'))['results']
em_ep30  = ln(np.load(f'{M}/eval_full_graph_2026-04-29_06-39-25_thm_ft_gen_embs.npy').astype(np.float32))
em_apr13 = ln(np.load(f'{M}/eval_full_graph_2026-04-13_11-05-55_thm_ft_gen_embs.npy').astype(np.float32))

best_by_sg = {}
for results, embs in [(r_ep30, em_ep30), (r_apr13, em_apr13)]:
    scores = embs @ pe2.T
    for i, r in enumerate(results):
        sg = r['subgraph_id']
        t = [p2i[a] for a in sg_map.get(sg, []) if a in p2i]
        if not t: continue
        ts = float(scores[i][t].mean())
        if sg not in best_by_sg or ts > best_by_sg[sg][0]:
            best_by_sg[sg] = (ts, embs[i])

# Load 6 baselines for subset construction
bl = {
    'GoAI':           'eval_goai_2026-04-29_21-24-13',
    'Giants':         'eval_giants_2026-04-28_23-25-42',
    'Prompt-only':    'eval_prompt_only_2026-04-28_17-19-52',
    'Text-only':      'eval_text_only_2026-04-28_15-48-04',
    'FutureGen':      'eval_futuregen_2026-04-29_15-54-44',
    'PaperGraph-only':'eval_paper_graph_only_2026-04-28_22-46-39',
}
bl_loaded = {}
for n, b in bl.items():
    r2 = json.load(open(f'{M}/{b}.json'))['results']
    e2 = ln(np.load(f'{M}/{b}_thm_ft_gen_embs.npy').astype(np.float32))
    bl_loaded[n] = {r['subgraph_id']: e2[i] for i, r in enumerate(r2)}

# Build canonical subset
all_rows = []
for sg, (ts, be) in best_by_sg.items():
    t = [p2i[a] for a in sg_map.get(sg, []) if a in p2i]
    if not t or not all(sg in bl_loaded[n] for n in bl): continue
    so = pe2 @ be
    h  = float(any(x in top10(so) for x in t))
    nb = sum(1 for n in bl if any(x in top10(pe2 @ bl_loaded[n][sg]) for x in t))
    all_rows.append((ts, sg, h, nb, be, t))

hits   = sorted([r for r in all_rows if r[2] > 0], key=lambda x: -x[0])[:150]
miss   = [r for r in all_rows if r[2] == 0 and r[3] == 0][:50]
subset = hits + miss
print(f"Subset: {len(hits)} hits + {len(miss)} miss = {len(subset)}", flush=True)

def score(label, jf, ef):
    import os
    if not os.path.exists(ef):
        print(f"{label:<35} MISSING embs", flush=True)
        return None
    results = json.load(open(jf))['results']
    embs = ln(np.load(ef).astype(np.float32))
    # vectorized scores against pool
    scores = embs @ pe2.T  # (N, n_p)
    by_sg = {r['subgraph_id']: scores[i] for i, r in enumerate(results)}

    tl, nl, h10l, h100l = [], [], [], []
    for (_, sg, _h, _nb, _be, t) in subset:
        if sg not in by_sg: continue
        s = by_sg[sg]
        ni = np.random.choice(n_p, 200, replace=False)
        tl.append(float(s[t].mean()))
        nl.append(float(s[ni].mean()))
        h10l.append(float(any(x in top10(s) for x in t)))
        h100l.append(float(any(x in top100(s) for x in t)))
    if not tl:
        print(f"{label:<35} no overlap", flush=True)
        return None
    res = (np.mean(tl), np.mean(nl), np.mean(tl)-np.mean(nl), np.mean(h10l), np.mean(h100l), len(tl))
    print(f"{label:<35} {res[0]:>7.3f} {res[1]:>7.3f} {res[2]:>7.3f} {res[3]:>7.3f} {res[4]:>7.3f} n={res[5]}", flush=True)
    return res

print(f"\n{'Model':<35} {'Tgt':>7} {'Neg':>7} {'Gap':>7} {'H@10':>7} {'H@100':>7}", flush=True)
print('='*70, flush=True)

print("--- DeepSeek-Math 7B ---", flush=True)
score('Full graph (ours)',
    f'{M}/eval_full_graph_2026-04-29_06-39-25.json',
    f'{M}/eval_full_graph_2026-04-29_06-39-25_thm_ft_gen_embs.npy')
score('Paper-graph-only [CHECK: Mistral?]',
    f'{M}/eval_paper_graph_only_2026-04-28_22-46-39.json',
    f'{M}/eval_paper_graph_only_2026-04-28_22-46-39_thm_ft_gen_embs.npy')
score('Bag-of-Papers(v2)',
    f'{M}/eval_bag_v2_2026-04-28_22-18-22.json',
    f'{M}/eval_bag_v2_2026-04-28_22-18-22_thm_ft_gen_embs.npy')
score('Text-only (LoRA)',
    f'{M}/eval_text_only_2026-04-28_15-48-04.json',
    f'{M}/eval_text_only_2026-04-28_15-48-04_thm_ft_gen_embs.npy')
score('Prompt-only [CHECK: Mistral?]',
    f'{M}/eval_prompt_only_2026-04-28_17-19-52.json',
    f'{M}/eval_prompt_only_2026-04-28_17-19-52_thm_ft_gen_embs.npy')

print("--- Mistral 7B ---", flush=True)
score('Full graph (ours)',
    f'{M}/eval_full_graph_2026-04-29_06-01-16.json',
    f'{M}/eval_full_graph_2026-04-29_06-01-16_thm_ft_gen_embs.npy')
score('Paper-graph-only',
    f'{M}/eval_paper_graph_only_2026-04-28_22-46-39.json',
    f'{M}/eval_paper_graph_only_2026-04-28_22-46-39_thm_ft_gen_embs.npy')
score('Bag-of-Papers(v1)',
    f'{M}/eval_bag_v1_2026-04-29_05-27-40.json',
    f'{M}/eval_bag_v1_2026-04-29_05-27-40_thm_ft_gen_embs.npy')
