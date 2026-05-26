"""
Run COMPOSE on a single arXiv paper.

Fetches the paper and its references from Semantic Scholar, builds both the
citation graph (enc1) and the Mathlib theorem subgraph (enc2) automatically,
and generates predicted future theorems with the full dual-encoder model.

Usage:
    python3 run_on_paper.py --arxiv_id 2301.07041
    python3 run_on_paper.py --arxiv_id 2301.07041 --n 3 --max_new_tokens 250

    # With retrieval: rank predictions against the 14,677-paper pool
    python3 run_on_paper.py --arxiv_id 2301.07041 \
        --retrieval_index data/retrieval_index.npz \
        --target_arxiv_id 2309.05516

Requirements:
    pip install requests
    Checkpoint in $COMPOSE_CKPT_DIR/best_model.pt (or pass --checkpoint)
    Mathlib data in $COMPOSE_DATA_DIR/LeanDojo/ (for full dual-encoder)
"""

import os, sys, json, argparse, logging, tempfile, shutil, warnings, contextlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Suppress noisy third-party warnings before any imports trigger them
warnings.filterwarnings('ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TQDM_DISABLE', '1')           # hide checkpoint shard progress bars
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_CKPT_ROOT = os.environ.get('COMPOSE_CKPT_DIR', os.path.join(_REPO_ROOT, 'checkpoints'))
_DATA_ROOT = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
_LEAN_DIR  = os.path.join(_DATA_ROOT, 'LeanDojo', 'leandojo_benchmark_4')

DEFAULT_CKPT   = os.path.join(_CKPT_ROOT, 'best_model.pt')
DEFAULT_INDEX  = os.path.join(_DATA_ROOT, 'retrieval_index.npz')

# Mathlib corpus: E5-1024 embeddings of all Mathlib theorem statements
CORPUS_EMB     = os.path.join(_LEAN_DIR, 'embeddings_corpus', 'embeddings.npy')
CORPUS_NAMES   = os.path.join(_LEAN_DIR, 'embeddings_corpus', 'idx_to_name.json')
CORPUS_STMTS   = os.path.join(_LEAN_DIR, 'embeddings_corpus', 'idx_to_statement.json')
GRAPH_FILE     = os.path.join(_LEAN_DIR, 'processed', 'graph.json')
# DeepSeek name index — enc2 looks up embeddings by name from this set
DEEPSEEK_IDX   = os.path.join(_LEAN_DIR, 'embeddings_deepseek', 'name_to_idx.json')

E5_MODEL = "intfloat/e5-large-v2"
S2_API   = "https://api.semanticscholar.org/graph/v1"
S2_CACHE = os.environ.get(
    "COMPOSE_S2_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "compose_s2_cache")
)
MAX_REFS = 30

# Mathlib subgraph build params
TOP_K_SEEDS  = 10   # top-K theorems by cosine sim to paper abstract
BFS_HOPS     = 4    # dependency expansion hops
MAX_PER_HOP  = 6    # max new nodes per hop

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose internal logs — show only errors from third-party libs
for _noisy in ('model_clean', 'model_clean_theorem', 'train_dual',
               'transformers', 'peft', 'torch', 'accelerate'):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def _fetch_arxiv_fallback(arxiv_id):
    """Fetch title+abstract from arXiv OAI API when S2 is rate-limited."""
    import requests, xml.etree.ElementTree as ET
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(r.text)
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        title = (entry.findtext('atom:title', '', ns) or '').replace('\n', ' ').strip()
        abstract = (entry.findtext('atom:summary', '', ns) or '').replace('\n', ' ').strip()
        if not title:
            return None
        return {'paperId': None, 'title': title, 'abstract': abstract, 'references': []}
    except Exception as e:
        logger.warning(f"  arXiv fallback failed: {e}")
        return None


def fetch_paper(arxiv_id):
    import requests, time
    os.makedirs(S2_CACHE, exist_ok=True)
    cache_path = os.path.join(S2_CACHE, f"{arxiv_id.replace('/', '_')}.json")
    if os.path.exists(cache_path):
        logger.info(f"  Using cached S2 response for arXiv:{arxiv_id}")
        with open(cache_path) as f:
            return json.load(f)

    fields = "paperId,title,abstract,year,references.paperId,references.title,references.abstract"
    waits = [15, 30, 60]
    for attempt in range(3):
        try:
            r = requests.get(f"{S2_API}/paper/arXiv:{arxiv_id}",
                             params={"fields": fields}, timeout=30)
            if r.status_code == 429:
                wait = waits[min(attempt, len(waits) - 1)]
                logger.info(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if data is None or not data.get("paperId"):
                raise ValueError(f"Paper not found on Semantic Scholar: arXiv:{arxiv_id}")
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return data
        except ValueError:
            raise
        except Exception:
            if attempt == 2: raise
            time.sleep(10)

    logger.warning("  Semantic Scholar rate-limited after 3 attempts — falling back to arXiv API "
                   "(citation graph will be empty; enc2 Mathlib path still active)")
    data = _fetch_arxiv_fallback(arxiv_id)
    if data is None:
        logger.warning("  arXiv fallback also unavailable — running with no citation graph (enc2/Mathlib path only)")
        data = {'paperId': None, 'title': f'arXiv:{arxiv_id}', 'abstract': '', 'references': []}
    return data


# ── E5 embedding ──────────────────────────────────────────────────────────────

_e5 = {}

def get_e5(device):
    if not _e5:
        logger.info(f"Loading E5: {E5_MODEL}")
        tok = AutoTokenizer.from_pretrained(E5_MODEL)
        mdl = AutoModel.from_pretrained(E5_MODEL).to(device).eval()
        _e5.update(tok=tok, mdl=mdl)
    return _e5['tok'], _e5['mdl']


def embed(texts, device, prefix="passage"):
    tok, mdl = get_e5(device)
    inputs = [f"{prefix}: {t}" for t in texts]
    enc = tok(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        out = mdl(**enc)
    emb = out.last_hidden_state[:, 0]
    return F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)


# ── Mathlib subgraph builder ──────────────────────────────────────────────────

_mathlib_corpus = {}

def load_mathlib_corpus():
    if _mathlib_corpus:
        return _mathlib_corpus
    if not os.path.exists(CORPUS_EMB) or not os.path.exists(DEEPSEEK_IDX):
        return {}
    logger.info("Loading Mathlib corpus embeddings...")
    embs = np.load(CORPUS_EMB, mmap_mode='r')          # [180907, 1024] E5
    with open(CORPUS_NAMES) as f:
        idx_to_name = json.load(f)
    with open(CORPUS_STMTS) as f:
        idx_to_stmt = json.load(f)
    with open(DEEPSEEK_IDX) as f:
        deepseek_names = set(json.load(f).keys())      # 104K names enc2 can look up

    # Only keep corpus entries that enc2 can actually embed (have DeepSeek embeddings).
    # 78K corpus theorems have no DeepSeek embedding → they'd be zero vectors in enc2.
    valid_mask = np.array([n in deepseek_names for n in idx_to_name], dtype=bool)
    valid_indices = np.where(valid_mask)[0]             # [~102K]

    logger.info(f"  {len(idx_to_name):,} corpus theorems, "
                f"{valid_mask.sum():,} valid for enc2 (have DeepSeek embedding)")

    _mathlib_corpus.update(
        embs=embs,
        idx_to_name=idx_to_name,
        idx_to_stmt=idx_to_stmt,
        name_to_idx={n: i for i, n in enumerate(idx_to_name)},
        valid_indices=valid_indices,               # used for constrained top-K search
        deepseek_names=deepseek_names,
    )
    return _mathlib_corpus


_mathlib_graph = {}

def load_mathlib_graph():
    if _mathlib_graph:
        return _mathlib_graph
    if not os.path.exists(GRAPH_FILE):
        return {}
    logger.info("Loading Mathlib dependency graph...")
    with open(GRAPH_FILE) as f:
        g = json.load(f)
    _mathlib_graph.update(
        theorem_to_premises=g.get('theorem_to_premises', {}),
        premise_to_theorems=g.get('premise_to_theorems', {}),
    )
    logger.info(f"  {len(_mathlib_graph['theorem_to_premises']):,} theorems in graph")
    return _mathlib_graph


def build_mathlib_subgraph(paper_abstract, device):
    """
    Build a Mathlib theorem subgraph for any paper by:
      1. Embedding the paper abstract with E5
      2. Finding top-K nearest Mathlib theorems by cosine similarity
      3. BFS-expanding their proof dependencies

    Returns a dict with 'subgraph_theorems' and 'subgraph_edges' ready for enc2,
    or None if Mathlib corpus is not available.
    """
    corpus = load_mathlib_corpus()
    graph  = load_mathlib_graph()
    if not corpus or not graph:
        return None

    # Embed paper abstract with E5 → find nearest Mathlib theorems.
    # Search only among theorems that have DeepSeek embeddings (valid for enc2).
    paper_emb   = embed([paper_abstract], device, prefix="query")[0]  # [1024]
    corpus_embs = corpus['embs']                                        # [N, 1024] mmap
    valid_idx   = corpus['valid_indices']                               # [~102K]

    chunk = 10000
    valid_scores = np.empty(len(valid_idx), dtype=np.float32)
    for i in range(0, len(valid_idx), chunk):
        rows = valid_idx[i:i+chunk]
        valid_scores[i:i+chunk] = corpus_embs[rows] @ paper_emb

    top_pos     = np.argsort(-valid_scores)[:TOP_K_SEEDS]
    top_idx     = valid_idx[top_pos]
    seed_names  = [corpus['idx_to_name'][i] for i in top_idx]
    logger.info(f"  Top Mathlib seeds: {seed_names[:3]}...")

    # BFS expand via proof dependencies — only add nodes enc2 can embed
    t2p          = graph['theorem_to_premises']
    deepseek_ok  = corpus['deepseek_names']
    all_nodes    = list(seed_names)
    node_set     = set(seed_names)
    frontier     = set(seed_names)

    for _ in range(BFS_HOPS):
        next_frontier = set()
        added = 0
        for name in frontier:
            for dep in t2p.get(name, []):
                if dep not in node_set and dep in deepseek_ok:
                    node_set.add(dep)
                    all_nodes.append(dep)
                    next_frontier.add(dep)
                    added += 1
                    if added >= MAX_PER_HOP:
                        break
            if added >= MAX_PER_HOP:
                break
        frontier = next_frontier
        if not frontier:
            break

    node_idx = {name: i for i, name in enumerate(all_nodes)}

    # Build edges [src_idx, dst_idx]
    edges = []
    for name in all_nodes:
        for dep in t2p.get(name, []):
            if dep in node_idx:
                edges.append([node_idx[dep], node_idx[name]])

    # Build theorem entries (name + statement)
    # All nodes are guaranteed to be in deepseek_names, so enc2 will find their embeddings.
    # Statement is optional (enc2 only uses the name for embedding lookup).
    theorems = []
    for name in all_nodes:
        idx  = corpus['name_to_idx'].get(name)
        stmt = corpus['idx_to_stmt'][idx] if idx is not None else ""
        theorems.append({"name": name, "statement": stmt, "file": ""})

    logger.info(f"  Mathlib subgraph: {len(theorems)} nodes, {len(edges)} edges")
    return {"subgraph_theorems": theorems, "subgraph_edges": edges}


# ── Build citation subgraph sample ───────────────────────────────────────────

def build_sample(data, device, tmpdir, mathlib_subgraph=None):
    anchor_id    = data.get("paperId", "anchor")
    anchor_title = data.get("title", "") or ""
    anchor_abs   = data.get("abstract", "") or ""

    refs = [r for r in (data.get("references") or [])
            if r.get("title") and r.get("abstract")][:MAX_REFS]

    raw_nodes = [{"id": anchor_id, "title": anchor_title, "abstract": anchor_abs}]
    for r in refs:
        raw_nodes.append({
            "id": r["paperId"] or f"ref_{len(raw_nodes)}",
            "title": r["title"],
            "abstract": r["abstract"] or "",
        })

    logger.info(f"Embedding {len(raw_nodes)} citation nodes with E5...")
    title_embs = embed([n["title"] for n in raw_nodes], device)
    abs_embs   = embed([n["abstract"] for n in raw_nodes], device)

    nodes = []
    for i, n in enumerate(raw_nodes):
        t_path = os.path.join(tmpdir, f"title_{i}.npy")
        s_path = os.path.join(tmpdir, f"summ_{i}.npy")
        np.save(t_path, title_embs[i])
        np.save(s_path, abs_embs[i])
        nodes.append({
            "paper_id":       n["id"],
            "title_text":     n["title"],
            "summery_text":   n["abstract"],
            "title_emb_e5":   t_path,
            "summery_emb_e5": s_path,
            "identifiers":    [],
            "year":           None,
        })

    edges = [{"source": anchor_id, "target": n["id"]} for n in raw_nodes[1:]]

    return {
        "paper_subgraph":     {"nodes": nodes, "edges": edges},
        "mathlib_subgraph":   mathlib_subgraph or {},
        "phrase":             anchor_title,
        "target_arxiv_id":    "demo",
        "target_text":        "",
        "target_text_source": "demo",
    }


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device, use_enc2=False):
    from train_dual import DualEncoderModel
    logger.info("Building DualEncoderModel...")
    # Redirect both stdout/stderr during model init to suppress tqdm shard bars and peft prints
    with open(os.devnull, 'w') as _dn, \
         contextlib.redirect_stdout(_dn), contextlib.redirect_stderr(_dn):
        model = DualEncoderModel(device=device, no_enc2=not use_enc2)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.bridge.load_state_dict(ckpt['bridge_state_dict'])
    model.fusion.load_state_dict(ckpt['fusion_state_dict'])
    model.type_embed.load_state_dict(ckpt['type_embed_state_dict'])
    model.fused_proj.load_state_dict(ckpt['fused_proj_state_dict'])
    model.decoder.load_state_dict(ckpt['decoder_state_dict'])
    model.enc1.gnn.load_state_dict(ckpt['enc1_gnn_state_dict'])
    if use_enc2:
        result = model.enc2.gnn.load_state_dict(ckpt['enc2_gnn_adapt_state_dict'], strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.debug(f"  enc2 GNN partial load — missing: {result.missing_keys[:3]}, "
                         f"unexpected: {result.unexpected_keys[:3]}")
    logger.info(f"  epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model.float().eval().to(device)


# ── Retrieval pool ────────────────────────────────────────────────────────────

def load_retrieval_pool(npz_path):
    logger.info(f"Loading retrieval pool: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embs      = data['embeddings'].astype(np.float32)
    arxiv_ids = data['arxiv_ids'].astype(str)

    id_to_rows = defaultdict(list)
    for i, pid in enumerate(arxiv_ids):
        id_to_rows[pid].append(i)

    pool_ids, pool_embs = [], []
    for pid, rows in id_to_rows.items():
        e = embs[rows]
        pool_embs.append(e[np.argmax(np.linalg.norm(e, axis=1))])
        pool_ids.append(pid)

    pool_embs = np.stack(pool_embs, axis=0)
    pool_ids  = np.array(pool_ids, dtype=str)
    logger.info(f"  Pool: {len(pool_ids):,} unique papers")
    return pool_embs, pool_ids


def retrieve(gen_text, device, pool_embs, pool_ids, target_arxiv_id=None):
    gen_emb = embed([gen_text], device, prefix="query")[0]
    order   = np.argsort(-(pool_embs @ gen_emb))
    rank = None
    if target_arxiv_id is not None:
        hits = np.where(pool_ids[order] == target_arxiv_id)[0]
        rank = int(hits[0]) + 1 if len(hits) else None
    return rank, order


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--arxiv_id',  help='Single arXiv ID, e.g. 2301.07041')
    grp.add_argument('--arxiv_ids', help='Comma-separated arXiv IDs for batch mode (model loads once)')
    p.add_argument('--checkpoint',      default=DEFAULT_CKPT)
    p.add_argument('--n',               type=int, default=3, help='Number of predictions')
    p.add_argument('--max_new_tokens',  type=int, default=150)
    p.add_argument('--temperature',     type=float, default=0.5,
                   help='Sampling temperature. Set to 0 for greedy.')
    p.add_argument('--device',          default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--retrieval_index', default=None,
                   help='Path to retrieval_index.npz. Ranks predictions against the 14,677-paper pool.')
    p.add_argument('--target_arxiv_id', default=None,
                   help='arXiv ID of a known future paper. Reports its rank in the retrieval pool.')
    return p.parse_args()


def main():
    args = parse_args()

    arxiv_ids = [args.arxiv_id] if args.arxiv_id else [i.strip() for i in args.arxiv_ids.split(',') if i.strip()]

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Download with:  bash scripts/download_checkpoint.sh")
        sys.exit(1)

    retrieval_path = args.retrieval_index or (DEFAULT_INDEX if os.path.exists(DEFAULT_INDEX) else None)

    # In batch mode, load Mathlib corpus + model once before looping
    _mathlib_corpus_preload(args.device)
    model = None

    for arxiv_id in arxiv_ids:
        args.arxiv_id = arxiv_id
        _run_one(args, model, retrieval_path)
        if model is None:
            # grab the model that was built on the first pass so subsequent
            # papers skip the 3-minute checkpoint load
            model = _last_model[0] if _last_model else None


def _mathlib_corpus_preload(device):
    """Warm up Mathlib embeddings and dependency graph (shared across papers)."""
    build_mathlib_subgraph("", device)  # empty string → loads corpus, returns None subgraph


_last_model = []


def _run_one(args, preloaded_model, retrieval_path):
    logger.info(f"Fetching arXiv:{args.arxiv_id} from Semantic Scholar...")
    data = fetch_paper(args.arxiv_id)
    logger.info(f"  Title: {data.get('title')}")
    logger.info(f"  References with abstracts: "
                f"{sum(1 for r in (data.get('references') or []) if r.get('abstract'))}")

    # Build Mathlib theorem subgraph for enc2 (works for any paper)
    abstract = data.get("abstract") or data.get("title") or ""
    mathlib_sg = build_mathlib_subgraph(abstract, args.device)
    use_enc2 = mathlib_sg is not None and len(mathlib_sg.get('subgraph_theorems', [])) > 0

    if use_enc2:
        logger.info("  Mode: full COMPOSE (enc1 citation graph + enc2 Mathlib theorem graph)")
    else:
        logger.info("  Mode: citation-graph only (enc1) — Mathlib corpus not found, "
                    "set COMPOSE_DATA_DIR to enable enc2")

    pool_embs = pool_ids = None
    if retrieval_path:
        pool_embs, pool_ids = load_retrieval_pool(retrieval_path)

    tmpdir = tempfile.mkdtemp(prefix="compose_")
    try:
        sample = build_sample(data, args.device, tmpdir, mathlib_subgraph=mathlib_sg)
        if preloaded_model is not None:
            model = preloaded_model
        else:
            model = load_model(args.checkpoint, args.device, use_enc2=use_enc2)
            _last_model.clear()
            _last_model.append(model)

        do_sample  = args.temperature > 0
        gen_kwargs = {"do_sample": do_sample, "temperature": args.temperature} if do_sample else {}

        def _clean(text):
            """Truncate at the last complete sentence; strip garbled endings."""
            import re
            text = text.strip()
            # Remove non-ASCII garbage (Cyrillic, etc.) that the model occasionally hallucinates
            text = ''.join(c for c in text if ord(c) < 128 or c in '–—…')
            text = text.strip()
            if not text:
                return ''
            # Try stopping markers in priority order.
            # '; ' included as fallback for long single-sentence outputs.
            found = False
            for end_marker in ['. ', '.\n', '.\t', '; ']:
                idx = text.rfind(end_marker)
                if idx > len(text) // 3:
                    text = text[:idx + 1].strip()
                    found = True
                    break
            if not found:
                for marker in ['),', ').', '],', '].']:
                    idx = text.rfind(marker)
                    if idx > len(text) // 2:
                        text = text[:idx + 1].strip()
                        break
            # If text has an odd number of $ signs, the last formula is unclosed — strip it
            dollar_count = text.count('$')
            if dollar_count % 2 == 1:
                last_dollar = text.rfind('$')
                # Strip back to the last word boundary before the unclosed $
                text = re.sub(r'[\s,;]+$', '', text[:last_dollar]).strip()
            # Strip dangling LaTeX openers at the very end
            text = re.sub(r'[\\\{,;\s:=]+$', '', text).strip()
            return text

        def _generate_one(temperature=None):
            kw = dict(gen_kwargs)
            if temperature is not None:
                kw['temperature'] = temperature
                kw['do_sample'] = True
            with torch.no_grad():
                _, out, _ = model(
                    [sample],
                    target_texts=None,
                    phrases=[sample['phrase']],
                    max_new_tokens=args.max_new_tokens,
                    **kw,
                )
            return _clean(out[0] if out else '')

        generated = []
        for _ in range(args.n):
            text = _generate_one()
            # Retry up to 3 times at same temperature, then once at higher temperature
            for attempt in range(4):
                if text:
                    break
                bump = 0.7 if attempt == 3 else None
                text = _generate_one(temperature=bump)
            generated.append(text or '(no output)')

        mode_tag = "enc1 + enc2" if use_enc2 else "enc1 only"
        print("\n" + "="*70)
        print(f"COMPOSE predictions for arXiv:{args.arxiv_id}  [{mode_tag}]")
        print(f"  {data.get('title', '')}")
        print("="*70)

        for i, text in enumerate(generated):
            print(f"\n[{i+1}]\n{text.strip()}")

            if pool_embs is not None:
                rank, order = retrieve(
                    text, args.device, pool_embs, pool_ids, args.target_arxiv_id
                )
                n_pool = len(pool_ids)
                print(f"\n  Retrieval (pool={n_pool:,} papers):")
                if args.target_arxiv_id:
                    if rank is not None:
                        print(f"    Target arXiv:{args.target_arxiv_id}  "
                              f"rank={rank:,}  H@10={int(rank<=10)}  H@100={int(rank<=100)}")
                    else:
                        print(f"    Target arXiv:{args.target_arxiv_id} not found in pool")
                else:
                    print(f"    Top-5 retrieved: {', '.join(pool_ids[order[:5]])}")

        print("="*70 + "\n")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
