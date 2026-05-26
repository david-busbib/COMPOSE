"""
Metric helpers for eval_future.py
=================================

Metrics:
  Future paper similarity (vs linked future papers of this subgraph):
  - max_sim:     highest cosine similarity to any linked future paper
  - mean_sim:    average cosine similarity across all linked future papers
  - std_sim:     standard deviation of similarities

  Retrieval rank (correct future papers ranked against ALL 14K future papers):
  - hits_at_10:   1 if any correct future paper ranks in top-10
  - hits_at_100:  1 if any correct future paper ranks in top-100
  - mrr:          1/rank of the best-ranked correct future paper
  - median_rank:  median rank of all correct future papers

  Generation quality:
  - math_token_ratio: fraction of tokens that are math keywords/symbols
  - has_structure:    1 if text contains Theorem/Lemma/Proposition/Corollary/Proof
  - gen_length:       number of whitespace-split tokens in generated text
  - distinct_1:       corpus-level unique unigram ratio (higher = more diverse)
  - distinct_2:       corpus-level unique bigram ratio  (higher = more diverse)
"""

import re
import numpy as np


# =============================================================================
# Text generation quality vs ground truth
# =============================================================================

def rouge_l(gen: str, ref: str) -> float:
    """ROUGE-L F1 between generated text and reference (GT theorem)."""
    if not gen or not ref:
        return 0.0
    gen_tokens = gen.lower().split()
    ref_tokens = ref.lower().split()
    m, n = len(gen_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gen_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


_bertscore_model = {}

def bertscore_f1(gen: str, ref: str) -> float:
    """BERTScore F1 between generated text and reference using a small BERT model."""
    if not gen or not ref:
        return 0.0
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        if 'model' not in _bertscore_model:
            tok = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device).eval()
            _bertscore_model.update(tok=tok, model=model, device=device)

        tok   = _bertscore_model['tok']
        model = _bertscore_model['model']
        device = _bertscore_model['device']

        with torch.no_grad():
            def encode(text):
                enc = tok(text, return_tensors='pt', truncation=True,
                          max_length=512, padding=True).to(device)
                out = model(**enc).last_hidden_state  # [1, T, 768]
                # mean pool (exclude special tokens)
                mask = enc['attention_mask'].unsqueeze(-1).float()
                return (out * mask).sum(1) / mask.sum(1)  # [1, 768]

            g_emb = encode(gen)   # [1, 768]
            r_emb = encode(ref)   # [1, 768]

            # token-level cosine sim — approximate via sentence vectors
            g_norm = g_emb / g_emb.norm(dim=-1, keepdim=True)
            r_norm = r_emb / r_emb.norm(dim=-1, keepdim=True)
            score = (g_norm * r_norm).sum().item()
        return float(score)
    except Exception:
        return 0.0


LEAN_SYNTAX_RE = re.compile(
    r'\b(theorem|lemma|proposition|corollary|def|noncomputable)\b'
    r'.*?'
    r'(\:=|\bby\b)',
    re.DOTALL
)
LEAN_QUANTIFIER_RE = re.compile(r'[∀∃]|\\forall|\\exists')
LEAN_TYPE_RE = re.compile(r'\b(Nat|Int|Real|Complex|Finset|Set|List|Matrix|MvPolynomial|RingHom)\b')


def lean_syntax_score(gen_text: str) -> float:
    """
    Heuristic Lean syntax validity score in [0, 1].
    Checks for: theorem/lemma declaration, := or by, quantifiers, Lean type names.
    """
    score = 0.0
    if STRUCTURE_RE.search(gen_text):
        score += 0.25
    if LEAN_SYNTAX_RE.search(gen_text):
        score += 0.35
    if LEAN_QUANTIFIER_RE.search(gen_text):
        score += 0.20
    if LEAN_TYPE_RE.search(gen_text):
        score += 0.20
    return min(score, 1.0)


def gt_generation_metrics(gen_text: str, gt_text: str) -> dict:
    """
    Compute all GT-grounded generation metrics.
    Args:
        gen_text: model generated string
        gt_text:  ground truth target_text from v7
    Returns:
        dict with rouge_l, bertscore_f1, lean_syntax_score
    """
    return {
        'rouge_l':          rouge_l(gen_text, gt_text),
        'bertscore_f1':     bertscore_f1(gen_text, gt_text),
        'lean_syntax_score': lean_syntax_score(gen_text),
    }


def input_novelty_metrics(gen_text: str, input_texts: list, future_rouge: float) -> dict:
    """
    Measure whether the generated text is novel vs. copying input papers.

    Computes ROUGE-L between generated text and each input paper abstract,
    takes the max (worst case = most copying). Then computes the ratio
    future_rouge / input_rouge — ratio > 1 means the model is predicting
    forward, ratio < 1 means it's rephrasing known work.

    Args:
        gen_text:      generated text
        input_texts:   list of input paper abstracts/summaries
        future_rouge:  ROUGE-L against ground truth future paper (already computed)

    Returns:
        dict with input_rouge_l, future_input_ratio
    """
    if not gen_text or not input_texts:
        return {'input_rouge_l': 0.0, 'future_input_ratio': 0.0}

    input_rouges = [rouge_l(gen_text, t) for t in input_texts if t]
    input_rouge_max = max(input_rouges) if input_rouges else 0.0

    if input_rouge_max > 0:
        ratio = future_rouge / input_rouge_max
    else:
        ratio = float(future_rouge > 0)

    return {
        'input_rouge_l':      float(input_rouge_max),
        'future_input_ratio': float(ratio),
    }


# =============================================================================
# Future paper similarity
# =============================================================================

def future_paper_metrics(gen_emb: np.ndarray, paper_embs_list: list) -> dict:
    """
    Compare generated text embedding against this subgraph's linked future papers.

    Each future paper has MULTIPLE embeddings (abstract + theorems).
    Take max similarity across all embeddings of a paper → one score per paper.

    Args:
        gen_emb:         [1024] L2-normalized embedding of generated text
        paper_embs_list: list of [K_i, 1024] arrays, one per linked future paper

    Returns:
        dict with max_sim, mean_sim, std_sim, n_future
    """
    if not paper_embs_list:
        return {'max_sim': 0.0, 'mean_sim': 0.0, 'std_sim': 0.0, 'n_future': 0}

    per_paper_sims = []
    for paper_embs in paper_embs_list:
        sims = paper_embs @ gen_emb          # [K_i]
        per_paper_sims.append(float(np.max(sims)))

    per_paper_sims = np.array(per_paper_sims)
    return {
        'max_sim':  float(np.max(per_paper_sims)),
        'mean_sim': float(np.mean(per_paper_sims)),
        'std_sim':  float(np.std(per_paper_sims)),
        'n_future': len(per_paper_sims),
    }


# =============================================================================
# Retrieval rank metrics — correct pool
# =============================================================================

def retrieval_rank_metrics(gen_emb: np.ndarray,
                           future_arxiv_ids: list,
                           pool_embs: np.ndarray,
                           pool_paper_ids: np.ndarray) -> dict:
    """
    Rank correct future papers against ALL 14K future papers in the pool.

    The pool contains embeddings for every unique future paper across all subgraphs.
    For each subgraph, the correct papers (~24) compete against all other 14K papers.

    Args:
        gen_emb:          [1024] L2-normalized embedding of generated text
        future_arxiv_ids: list of str — arxiv_ids of correct future papers for this subgraph
        pool_embs:        [N, 1024] — one embedding per unique future paper (best embedding)
        pool_paper_ids:   [N] str  — arxiv_id for each row in pool_embs

    Returns:
        dict with hits_at_10, hits_at_100, mrr, median_rank
    """
    if not future_arxiv_ids:
        return {'hits_at_10': 0.0, 'hits_at_100': 0.0, 'mrr': 0.0, 'median_rank': 0}

    scores = pool_embs @ gen_emb          # [N]
    order = np.argsort(-scores)           # descending, rank 1 = best
    ranked_ids = pool_paper_ids[order]

    future_set = set(future_arxiv_ids)
    ranks = []
    for i, pid in enumerate(ranked_ids):
        if pid in future_set:
            ranks.append(i + 1)           # 1-indexed
            future_set.discard(pid)
            if not future_set:
                break

    if not ranks:
        return {'hits_at_10': 0.0, 'hits_at_100': 0.0, 'mrr': 0.0, 'median_rank': 0}

    best_rank = min(ranks)
    return {
        'hits_at_10':  float(best_rank <= 10),
        'hits_at_100': float(best_rank <= 100),
        'mrr':         float(1.0 / best_rank),
        'median_rank': int(np.median(ranks)),
    }


def retrieval_rank_metrics_multi(gen_emb: np.ndarray,
                                 future_arxiv_ids: list,
                                 paper_id_to_embs: dict,
                                 top_ks=(10, 100)) -> dict:
    """
    Rank correct future papers against all future papers using ALL embeddings
    available per paper.

    For each paper, its score is max(sim(generation, paper_target_i)) across
    abstract/theorem embeddings. This avoids the lossy single-representative
    pool used by retrieval_rank_metrics.

    Returns rank metrics plus score-separation diagnostics.
    """
    base = {
        'hits_at_10': 0.0,
        'hits_at_100': 0.0,
        'mrr': 0.0,
        'median_rank': 0,
        'best_rank': 0,
        'rank_percentile': 0.0,
        'recall_at_10': 0.0,
        'recall_at_100': 0.0,
        'average_precision': 0.0,
        'ndcg_at_100': 0.0,
        'best_pos_score': 0.0,
        'best_neg_score': 0.0,
        'score_gap': 0.0,
        'pos_mean_score': 0.0,
        'neg_mean_score': 0.0,
        'score_std': 0.0,
    }
    if not future_arxiv_ids or not paper_id_to_embs:
        return base

    paper_ids, scores = [], []
    for pid, embs in paper_id_to_embs.items():
        if embs is None or len(embs) == 0:
            continue
        paper_ids.append(pid)
        scores.append(float(np.max(embs @ gen_emb)))

    if not paper_ids:
        return base

    paper_ids = np.array(paper_ids, dtype=str)
    scores = np.array(scores, dtype=np.float32)
    future_set = set(future_arxiv_ids)
    is_pos = np.array([pid in future_set for pid in paper_ids], dtype=bool)
    n_pos = int(is_pos.sum())
    if n_pos == 0:
        return base

    order = np.argsort(-scores)
    ranked_pos = is_pos[order]
    ranks = np.flatnonzero(ranked_pos) + 1  # 1-indexed

    best_rank = int(ranks[0])
    precisions_at_pos = np.arange(1, len(ranks) + 1, dtype=np.float32) / ranks

    k = 100
    rel = ranked_pos[:k].astype(np.float32)
    dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
    ideal_len = min(n_pos, k)
    idcg = float(np.sum(np.ones(ideal_len, dtype=np.float32) /
                        np.log2(np.arange(2, ideal_len + 2)))) if ideal_len else 0.0

    pos_scores = scores[is_pos]
    neg_scores = scores[~is_pos]
    best_neg = float(np.max(neg_scores)) if len(neg_scores) else 0.0
    neg_mean = float(np.mean(neg_scores)) if len(neg_scores) else 0.0

    out = dict(base)
    out.update({
        'hits_at_10': float(best_rank <= 10),
        'hits_at_100': float(best_rank <= 100),
        'mrr': float(1.0 / best_rank),
        'median_rank': int(np.median(ranks)),
        'best_rank': best_rank,
        'rank_percentile': float(1.0 - ((best_rank - 1) / len(scores))),
        'recall_at_10': float(np.sum(ranked_pos[:10]) / n_pos),
        'recall_at_100': float(np.sum(ranked_pos[:100]) / n_pos),
        'average_precision': float(np.mean(precisions_at_pos)),
        'ndcg_at_100': float(dcg / idcg) if idcg else 0.0,
        'best_pos_score': float(np.max(pos_scores)),
        'best_neg_score': best_neg,
        'score_gap': float(np.max(pos_scores) - best_neg),
        'pos_mean_score': float(np.mean(pos_scores)),
        'neg_mean_score': neg_mean,
        'score_std': float(np.std(scores)),
    })
    return out


def build_pool_from_future_embs(paper_id_to_embs: dict) -> tuple:
    """
    Build pool arrays from future paper embeddings.
    Uses the best (highest-norm) embedding per paper as its single representative.

    Args:
        paper_id_to_embs: dict arxiv_id -> [K, 1024]

    Returns:
        pool_embs:      [N, 1024] float32
        pool_paper_ids: [N] str array
    """
    ids, embs = [], []
    for pid, e in paper_id_to_embs.items():
        # Use the embedding with highest L2 norm as representative
        norms = np.linalg.norm(e, axis=1)
        best = int(np.argmax(norms))
        ids.append(pid)
        embs.append(e[best])
    pool_embs = np.stack(embs, axis=0).astype(np.float32)   # [N, 1024]
    pool_paper_ids = np.array(ids, dtype=str)                # [N]
    return pool_embs, pool_paper_ids


# =============================================================================
# Generation quality metrics
# =============================================================================

MATH_KEYWORDS = {
    'theorem', 'lemma', 'proposition', 'corollary', 'proof', 'conjecture',
    'definition', 'remark', 'claim', 'hypothesis', 'axiom',
    'equation', 'inequality', 'integral', 'derivative', 'gradient',
    'convergence', 'divergence', 'continuous', 'differentiable', 'bounded',
    'compact', 'manifold', 'algebra', 'topology', 'metric', 'norm',
    'matrix', 'vector', 'eigenvalue', 'determinant', 'polynomial',
    'prime', 'integer', 'rational', 'real', 'complex', 'field', 'ring', 'group',
}

MATH_SYMBOLS_RE = re.compile(
    r'[∀∃∈∉⊂⊃⊆⊇∪∩→←↔⇒⇔≤≥≠≈∑∏∫∂∇√∞±×÷αβγδεζηθλμνπρστφψω]'
)
STRUCTURE_RE = re.compile(
    r'\b(theorem|lemma|proposition|corollary|proof|claim)\b', re.IGNORECASE
)


def generation_quality_metrics(gen_text: str) -> dict:
    """
    Per-sample generation quality metrics.

    Args:
        gen_text: full raw generated string (not truncated)

    Returns:
        dict with math_token_ratio, has_structure, gen_length
    """
    tokens = gen_text.split()
    n = len(tokens)

    if n == 0:
        return {'math_token_ratio': 0.0, 'has_structure': 0.0, 'gen_length': 0}

    math_count = sum(
        1 for t in tokens
        if t.lower().strip('.,;:()[]{}') in MATH_KEYWORDS
        or MATH_SYMBOLS_RE.search(t)
    )

    return {
        'math_token_ratio': float(math_count / n),
        'has_structure':    float(bool(STRUCTURE_RE.search(gen_text))),
        'gen_length':       n,
    }


# =============================================================================
# Citation relevance metric — 47K 2024-2025 papers index
# =============================================================================

_cit_index = {}   # lazy-loaded cache

def load_citation_index():
    """Load the 47K citation FAISS index (lazy, cached)."""
    if _cit_index:
        return _cit_index

    import os
    path = os.path.join(
        os.path.dirname(__file__),
        "../../../baselines/arxiv_2425_citation_embs.npz"
    )
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Citation index not found: {path}. Run embed_citation_papers.py first.")

    data = np.load(path, allow_pickle=True)
    embeddings      = data["embeddings"].astype(np.float32)   # [T, 1024]
    paper_ids       = data["paper_ids"]                        # [P]
    offsets         = data["offsets"].astype(int)              # [P+1]
    citation_counts = data["citation_counts"].astype(int)      # [P]
    pub_months      = data["pub_months"]                       # [P]

    # Build one representative embedding per paper (max-norm across its texts)
    pool_embs = []
    for i in range(len(paper_ids)):
        embs = embeddings[offsets[i]: offsets[i + 1]]
        if len(embs) == 0:
            pool_embs.append(np.zeros(1024, dtype=np.float32))
        else:
            norms = np.linalg.norm(embs, axis=1)
            pool_embs.append(embs[int(np.argmax(norms))])
    pool_embs = np.stack(pool_embs, axis=0)   # [P, 1024]

    # Precompute percentile ranks within each publication month
    month_to_counts = {}
    for cit, month in zip(citation_counts, pub_months):
        month_to_counts.setdefault(str(month), []).append(int(cit))

    # For each paper: percentile = fraction of papers in same month with <= citations
    percentiles = np.zeros(len(paper_ids), dtype=np.float32)
    for i, (cit, month) in enumerate(zip(citation_counts, pub_months)):
        bucket = month_to_counts.get(str(month), [int(cit)])
        percentiles[i] = float(np.mean(np.array(bucket) <= int(cit)))

    _cit_index.update(
        pool_embs=pool_embs,
        citation_counts=citation_counts,
        percentiles=percentiles,
        paper_ids=paper_ids,
        pub_months=pub_months,
    )
    return _cit_index


def citation_relevance_metrics(gen_emb: np.ndarray, top_k: int = 10) -> dict:
    """
    Embed generated text → find top-K nearest 2024-2025 papers →
    return mean raw citation count and mean percentile (normalized within pub month).

    Args:
        gen_emb: [1024] L2-normalized embedding of generated text
        top_k:   number of nearest neighbors to retrieve

    Returns:
        dict with citation_mean_raw, citation_percentile
    """
    try:
        idx = load_citation_index()
    except FileNotFoundError:
        return {'citation_mean_raw': -1.0, 'citation_percentile': -1.0}

    scores = idx["pool_embs"] @ gen_emb          # [P]
    top_k_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]

    cit_raw  = float(np.mean(idx["citation_counts"][top_k_idx]))
    cit_pct  = float(np.mean(idx["percentiles"][top_k_idx]))

    return {
        'citation_mean_raw':  cit_raw,
        'citation_percentile': cit_pct,
    }


def distinct_metrics(all_gen_texts: list) -> dict:
    """
    Corpus-level diversity across ALL generated texts.
    Call once after all generations are collected.

    distinct_1 = unique unigrams / total unigrams
    distinct_2 = unique bigrams  / total bigrams

    Args:
        all_gen_texts: list of full (untruncated) generated strings

    Returns:
        dict with distinct_1, distinct_2
    """
    all_unigrams, all_bigrams = [], []
    for text in all_gen_texts:
        tokens = text.split()
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens, tokens[1:]))

    d1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    d2 = len(set(all_bigrams))  / max(len(all_bigrams),  1)
    return {'distinct_1': float(d1), 'distinct_2': float(d2)}
