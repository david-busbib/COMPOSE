"""
Evaluate all 4 models against linked 2024-2025 future papers.
==============================================================

For each v7 val-split sample that has linked future papers:
  1. Generate text with the selected model
  2. Embed generated text with E5-large-v2
  3. Compare against pre-embedded future paper targets

Models:
  --model full_graph          (dual encoder, enc1+enc2+fusion)
  --model paper_graph_only    (dual encoder, enc2 zeroed)
  --model text_only           (DeepSeek-Math + LoRA, flat text prompt)
  --model prompt_only         (base DeepSeek-Math, same flat text prompt, no fine-tune)
  --model retrieval           (E5 nearest-neighbor from v7 index)

Metrics per sample:
  max_sim, mean_sim, hits@0.7, std_sim  (vs linked future papers)
  v7_sim                                (vs v7 ground-truth target_text)

Usage:
  # Step 1: Pre-embed future paper targets (run once)
  python3 eval_future.py --embed_targets

  # Step 2: Run eval for each model
  python3 eval_future.py --model full_graph     --checkpoint /path/to/best_model.pt
  python3 eval_future.py --model paper_graph_only --checkpoint /path/to/best_model.pt
  python3 eval_future.py --model text_only      --checkpoint /path/to/checkpoint.pt
  python3 eval_future.py --model prompt_only
  python3 eval_future.py --model retrieval

  # Options
  --n 100          number of val samples to evaluate (default: all with future papers)
  --skip 0         skip first N eligible samples
  --max_new_tokens 300
"""

import os
import sys
import json
import argparse
import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# CODE_BASE: compose-arxiv/code/  (walking up from code/baselines/metrics/eval_future.py)
CODE_BASE     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT    = os.path.dirname(CODE_BASE)

# Data root: override with COMPOSE_DATA_DIR env var, otherwise default to repo data/
BASE          = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
_CKPT_ROOT    = os.environ.get('COMPOSE_CKPT_DIR', os.path.join(_REPO_ROOT, 'checkpoints'))
LEANDOJO_BASE = os.path.join(BASE, 'LeanDojo/leandojo_benchmark_4')

# Eval-specific data dir (future paper embeddings, theorem LoRA).
# For arXiv users: put eval data files here or set COMPOSE_EVAL_DIR.
# Defaults to the same directory as this script.
_EVAL_DIR     = os.environ.get('COMPOSE_EVAL_DIR', os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, CODE_BASE)
sys.path.insert(0, os.path.join(CODE_BASE, 'baselines'))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Paths
V7_DATA        = os.path.join(BASE, 'dual_training_samples_v7_clean.jsonl')
FUTURE_INDEX   = os.path.join(BASE, 'subgraph_to_2425_papers.jsonl')
FUTURE_EMBS    = os.path.join(_EVAL_DIR, 'future_paper_embs_mistral.npz')
RETRIEVAL_IDX  = os.environ.get('COMPOSE_RETRIEVAL_IDX', os.path.join(BASE, 'retrieval_index.npz'))
E5_MODEL       = 'intfloat/e5-large-v2'
MISTRAL_EMB_MODEL = 'intfloat/e5-mistral-7b-instruct'

# Default checkpoints (override with --checkpoint)
DEFAULT_CKPTS = {
    'full_graph':       os.path.join(_CKPT_ROOT, 'best_model.pt'),
    'paper_graph_only': os.path.join(_CKPT_ROOT, 'best_model.pt'),
    'text_only':        None,  # set via --checkpoint
}


# =============================================================================
# E5 embedding
# =============================================================================

_e5_cache = {}

def load_e5():
    if 'model' in _e5_cache:
        return _e5_cache['tok'], _e5_cache['model'], _e5_cache['device']
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading E5: {E5_MODEL}")
    tok   = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    _e5_cache.update(tok=tok, model=model, device=device)
    logger.info(f"E5 loaded on {device}")
    return tok, model, device


@torch.no_grad()
def embed_texts(texts, batch_size=128, prefix='passage'):
    """Embed texts → numpy [N, 1024], L2-normalized."""
    tok, model, device = load_e5()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {t}" for t in texts[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


_mistral_emb_cache = {}

def load_mistral_emb():
    if 'model' in _mistral_emb_cache:
        return _mistral_emb_cache['tok'], _mistral_emb_cache['model'], _mistral_emb_cache['device']
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading Mistral embedder: {MISTRAL_EMB_MODEL}")
    tok   = AutoTokenizer.from_pretrained(MISTRAL_EMB_MODEL)
    model = AutoModel.from_pretrained(MISTRAL_EMB_MODEL, torch_dtype=torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    _mistral_emb_cache.update(tok=tok, model=model, device=device)
    logger.info(f"Mistral embedder loaded on {device}")
    return tok, model, device


_thm_lora_emb_cache = {}
THM_LORA_MODEL_DIR = os.path.join(_EVAL_DIR, 'theorem_emb_lora_model', 'epoch_2')
FUTURE_THM_LORA_POOL = os.path.join(_EVAL_DIR, 'future_paper_theorem_embs_finetuned.npz')

def load_thm_lora_emb():
    """Load DeepSeek-Math-7B with finetuned theorem LoRA adapter."""
    if 'model' in _thm_lora_emb_cache:
        return _thm_lora_emb_cache['tok'], _thm_lora_emb_cache['model'], _thm_lora_emb_cache['device']
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    # Base model must match what the LoRA was trained on (adapter_config says deepseek-math-7b-instruct)
    _THM_BASE = 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading theorem LoRA embedder from {THM_LORA_MODEL_DIR} (base: {_THM_BASE})")
    tok   = AutoTokenizer.from_pretrained(THM_LORA_MODEL_DIR)
    tok.pad_token = tok.eos_token
    base  = AutoModel.from_pretrained(_THM_BASE, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, THM_LORA_MODEL_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    _thm_lora_emb_cache.update(tok=tok, model=model, device=device)
    logger.info(f"Theorem LoRA embedder loaded on {device}")
    return tok, model, device


@torch.no_grad()
def embed_texts_thm_lora(texts, batch_size=8, instruction=None):
    """Embed texts with finetuned theorem LoRA on E5-Mistral → numpy [N, 4096], L2-normalized."""
    tok, model, device = load_thm_lora_emb()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if instruction:
            batch = [f"Instruct: {instruction}\nQuery: {t}" for t in batch]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        seq_len = enc['attention_mask'].sum(dim=1) - 1
        emb = h[torch.arange(len(batch)), seq_len]
        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  ThmLoRA embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def embed_texts_mistral(texts, batch_size=16, instruction=None):
    """Embed texts with e5-mistral-7b-instruct → numpy [N, 4096], L2-normalized.
    Uses last-token pooling as per e5-mistral specification.
    """
    tok, model, device = load_mistral_emb()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if instruction:
            batch = [f"Instruct: {instruction}\nQuery: {t}" for t in batch]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        seq_len = enc['attention_mask'].sum(dim=1) - 1
        emb = h[torch.arange(len(batch)), seq_len]
        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Mistral embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


# =============================================================================
# Step 0: Pre-embed all future paper targets
# =============================================================================

def get_future_target_texts(paper):
    """Return ALL meaningful target texts for a future paper as a list.

    Each paper contributes multiple targets:
      - abstract (always included if non-empty)
      - each individual theorem/proposition (up to MAX_THMS)

    This lets us take max similarity across all variants,
    matching the fact that the model may generate in different styles.
    """
    MAX_THMS = 5
    targets = []

    abstract = paper.get('abstract', '').strip()
    if abstract:
        targets.append(abstract)

    if not paper.get('is_abstract_fallback', False):
        theorems = paper.get('theorems', [])
        types    = paper.get('theorem_types', [])
        # Add each theorem individually (prefer theorem/proposition first)
        preferred, other = [], []
        for thm, typ in zip(theorems, types):
            thm = thm.strip()
            if not thm:
                continue
            if typ in ('theorem', 'proposition'):
                preferred.append(thm)
            else:
                other.append(thm)
        for thm in (preferred + other)[:MAX_THMS]:
            targets.append(thm)

    return targets if targets else ['']


def embed_future_targets():
    """Pre-embed all future paper targets (multiple per paper). Save to .npz.

    Storage layout:
      embeddings: [T, 1024]  — all target embeddings flat
      arxiv_ids:  [T]        — which paper each embedding belongs to
      offsets:    [P+1]      — paper p has embeddings[offsets[p]:offsets[p+1]]
      paper_ids:  [P]        — sorted unique arxiv_ids (index = paper index)
    """
    logger.info(f"Loading future paper index: {FUTURE_INDEX}")

    paper_targets = {}   # arxiv_id -> [text, text, ...]
    sg_to_arxiv   = {}   # subgraph_id -> [arxiv_ids]

    with open(FUTURE_INDEX) as f:
        for line in f:
            r = json.loads(line)
            sg_id = r['subgraph_id']
            arxiv_ids = []
            for p in r['papers']:
                aid = p['arxiv_id']
                arxiv_ids.append(aid)
                if aid not in paper_targets:
                    paper_targets[aid] = get_future_target_texts(p)
            sg_to_arxiv[sg_id] = arxiv_ids

    logger.info(f"Unique future papers: {len(paper_targets)}")
    logger.info(f"Subgraphs with future papers: {len(sg_to_arxiv)}")
    total_targets = sum(len(v) for v in paper_targets.values())
    logger.info(f"Total targets to embed: {total_targets}")

    # Flatten: build parallel lists of (arxiv_id, text)
    paper_ids_sorted = sorted(paper_targets.keys())
    flat_aids  = []
    flat_texts = []
    offsets    = [0]
    for aid in paper_ids_sorted:
        texts = paper_targets[aid]
        flat_aids.extend([aid] * len(texts))
        flat_texts.extend(texts)
        offsets.append(len(flat_texts))

    logger.info("Embedding all targets...")
    embs = embed_texts(flat_texts, batch_size=128)

    # Save
    os.makedirs(os.path.dirname(FUTURE_EMBS), exist_ok=True)
    np.savez(
        FUTURE_EMBS,
        embeddings = embs,                            # [T, 1024]
        flat_aids  = np.array(flat_aids),             # [T]
        paper_ids  = np.array(paper_ids_sorted),      # [P]
        offsets    = np.array(offsets, dtype=np.int64), # [P+1]
    )
    sg_map_path = os.path.join(os.path.dirname(FUTURE_EMBS), 'future_paper_embs_sg_map.json')
    with open(sg_map_path, 'w') as f:
        json.dump(sg_to_arxiv, f)

    logger.info(f"Saved {len(flat_texts)} target embeddings ({len(paper_ids_sorted)} papers) → {FUTURE_EMBS}")
    logger.info(f"Saved subgraph map → {sg_map_path}")


# =============================================================================
# Load future paper embeddings
# =============================================================================

def load_future_embs():
    """Load pre-embedded future paper targets.

    Returns:
        paper_id_to_embs: dict arxiv_id -> [K, 1024] array (K targets for that paper)
        sg_to_arxids:     dict subgraph_id -> [arxiv_id, ...]
    """
    data = np.load(FUTURE_EMBS, allow_pickle=True)
    embs      = data['embeddings']   # [T, D] — D=4096 for mistral, 1024 for e5
    paper_ids = list(data['paper_ids'])
    offsets   = data['offsets']      # [P+1]

    # Build paper_id -> embeddings slice
    paper_id_to_embs = {}
    for i, pid in enumerate(paper_ids):
        start, end = int(offsets[i]), int(offsets[i+1])
        paper_id_to_embs[pid] = embs[start:end]  # [K, 1024]

    sg_map_path = os.path.join(os.path.dirname(FUTURE_EMBS), 'future_paper_embs_sg_map.json')
    with open(sg_map_path) as f:
        sg_to_arxids = json.load(f)

    total_targets = sum(v.shape[0] for v in paper_id_to_embs.values())
    logger.info(f"Loaded {len(paper_ids)} future papers, {total_targets} targets, {len(sg_to_arxids)} subgraphs")
    return paper_id_to_embs, sg_to_arxids


# =============================================================================
# Load v7 val split + filter to samples with future papers
# =============================================================================

def load_val_samples(sg_to_arxids, n=None, skip=0, canonical_sg_ids=None):
    """Load v7 val split, keep only samples whose subgraph has linked future papers.

    If canonical_sg_ids is provided, restrict to exactly those subgraph IDs (ignoring n/skip).
    """
    logger.info(f"Loading v7 val split from {V7_DATA}")
    all_samples = []
    with open(V7_DATA) as f:
        for line in f:
            all_samples.append(json.loads(line))

    # Val split = last 10%
    split = int(0.9 * len(all_samples))
    val_samples = all_samples[split:]
    logger.info(f"Val split: {len(val_samples)} samples (from {split} to {len(all_samples)})")

    if canonical_sg_ids is not None:
        # Canonical mode: return exactly the specified subgraphs (any that appear in val split)
        seen_sg = set()
        eligible = []
        for s in val_samples:
            sg_id = s['paper_subgraph']['subgraph_id']
            if sg_id in canonical_sg_ids and sg_id not in seen_sg:
                seen_sg.add(sg_id)
                eligible.append(s)
        logger.info(f"Canonical subset matched: {len(eligible)}/{len(canonical_sg_ids)} subgraphs in val split")
        if n is not None and n > 0:
            eligible = eligible[:n]
        logger.info(f"Selected for eval: {len(eligible)}")
        return eligible

    # Filter to samples with linked future papers (min 11), one per subgraph
    seen_sg = set()
    eligible = []
    for s in val_samples:
        sg_id = s['paper_subgraph']['subgraph_id']
        if sg_id in sg_to_arxids and len(sg_to_arxids[sg_id]) >= 11 and sg_id not in seen_sg:
            seen_sg.add(sg_id)
            eligible.append(s)

    logger.info(f"Eligible (have future papers): {len(eligible)}")

    if skip > 0:
        eligible = eligible[skip:]
    if n is not None and n > 0:
        eligible = eligible[:n]

    logger.info(f"Selected for eval: {len(eligible)}")
    return eligible


# =============================================================================
# Model loaders & generators
# =============================================================================

def load_dual_model(ckpt_path, no_graph=False, no_enc2=False, no_fusion=False, no_enrichment=False, decoder_model=None):
    """Load DualEncoderModel from checkpoint using the same format as DualTrainer.load_checkpoint."""
    from train_dual import DualEncoderModel, DEFAULT_DECODER
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dec = decoder_model or DEFAULT_DECODER
    logger.info(f"Loading DualEncoderModel (no_graph={no_graph}, no_enc2={no_enc2}, no_fusion={no_fusion}, no_enrichment={no_enrichment}, decoder={dec})")
    model = DualEncoderModel(device=device, no_graph=no_graph, no_enc2=no_enc2, no_fusion=no_fusion, no_enrichment=no_enrichment, decoder_model=dec)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Load each component separately (matches DualTrainer.save_checkpoint format)
    model.bridge.load_state_dict(ckpt['bridge_state_dict'])
    model.fusion.load_state_dict(ckpt['fusion_state_dict'])
    model.type_embed.load_state_dict(ckpt['type_embed_state_dict'])
    model.fused_proj.load_state_dict(ckpt['fused_proj_state_dict'])
    if 'cross_proj_paper_state_dict' in ckpt:
        model.cross_proj_paper.load_state_dict(ckpt['cross_proj_paper_state_dict'])
        model.cross_proj_mathlib.load_state_dict(ckpt['cross_proj_mathlib_state_dict'])
    # mem_bank/mem_ptr are only used during training; resize to match checkpoint if needed
    if ckpt['mem_bank'].shape[0] != model.mem_bank.shape[0]:
        model.mem_bank = torch.zeros_like(ckpt['mem_bank'])
        model.mem_ptr = torch.zeros_like(ckpt['mem_ptr'])
    model.mem_bank.copy_(ckpt['mem_bank'])
    model.mem_ptr.copy_(ckpt['mem_ptr'])
    if 'mem_full' in ckpt:
        model.mem_full.copy_(ckpt['mem_full'])
    model.decoder.load_state_dict(ckpt['decoder_state_dict'])
    model.enc1.gnn.load_state_dict(ckpt['enc1_gnn_state_dict'])
    model.enc2.gnn.W_struct.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['W_struct'])
    model.enc2.gnn.layer_norms.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['layer_norms'])

    # Cast everything to float32 to avoid half/float dtype mismatch on H200
    model = model.float()
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model


@torch.no_grad()
def _get_root_title(sample):
    """Return root node title + abstract (anchor paper) from the subgraph."""
    from collections import Counter
    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
    edges = sample.get('paper_subgraph', {}).get('edges', [])
    if not edges:
        return sample.get('phrase', '')
    src_counts = Counter(e['source'] for e in edges)
    root_pid = src_counts.most_common(1)[0][0]
    root_node = next((n for n in nodes if str(n.get('paper_id', '')) == root_pid), None)
    if root_node is None:
        root_node = next((n for n in nodes if root_pid[:30] in str(n.get('paper_id', ''))), None)
    title_raw = (root_node.get('title_text', '') or '') if root_node else ''
    title = ' '.join(title_raw) if isinstance(title_raw, list) else title_raw
    title = title.strip() or sample.get('phrase', '')
    if root_node:
        abst_raw = root_node.get('summery_text', '') or root_node.get('abstract', '') or ''
        abst = ' '.join(abst_raw) if isinstance(abst_raw, list) else abst_raw
        abst = abst.strip()[:400]
        if abst:
            return f"{title}\nAbstract: {abst}"
    return title


def generate_dual(model, sample, max_new_tokens=300, no_phrase=False, use_title=False, use_graph_context=False, use_root_title=False, use_goai_prompt=False):
    """Generate text from a dual encoder model."""
    if no_phrase:
        phrases = None
    elif use_goai_prompt:
        phrase = build_goai_hybrid_prompt(sample)[:800]  # truncate to fit V100 44GB
        phrases = [phrase] if phrase else None
    elif use_root_title:
        phrase = _get_root_title(sample)
        phrases = [phrase] if phrase else None
    elif use_graph_context:
        # Build phrase from graph node titles — gives the decoder rich topical context
        nodes = sample.get('paper_subgraph', {}).get('nodes', [])
        titles = []
        for n in nodes[:6]:
            t = n.get('title_text', '') or n.get('title', '')
            if t and not t.endswith('_title_'):  # skip file path stubs
                titles.append(t.strip()[:80])
        base_phrase = sample.get('phrase', '') or sample.get('target_title', '')
        if titles:
            phrase = base_phrase + '. Related work: ' + '; '.join(titles[:4])
        else:
            phrase = base_phrase
        phrases = [phrase] if phrase else None
    elif use_title:
        phrase = sample.get('target_title', '') or sample.get('phrase', '')
        phrases = [phrase] if phrase else None
    else:
        phrase = sample.get('phrase', '') or sample.get('target_title', '')
        phrases = [phrase] if phrase else None
    _, generated, _ = model([sample], target_texts=None, max_new_tokens=max_new_tokens, phrases=phrases)
    return generated[0] if generated else ''


def load_prompt_only_model(decoder_model=None):
    """Load base model with no LoRA weights."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    DECODER_MODEL = decoder_model or 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading prompt-only base model: {DECODER_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('Prompt-only base model loaded')
    return model, tokenizer


def load_text_only_model(ckpt_path):
    """Load text-only DeepSeek-Math + LoRA from checkpoint."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    DECODER_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading text-only model: {DECODER_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Text-only missing keys ({len(missing)}): {missing[:5]}...")
    model.eval()
    logger.info(f"Text-only model loaded from {ckpt_path}")
    return model, tokenizer


def load_bag_model(ckpt_path, version='v1'):
    """Load bag_v1 or bag_v2 model from checkpoint."""
    import json as _json
    if version == 'v1':
        from train_bag_v1 import BagV1Model, V4_DATA
        model = BagV1Model(device='cuda:0')
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.proj.load_state_dict(ckpt['proj_state_dict'])
        model.decoder.load_state_dict(ckpt['decoder_state_dict'])
    else:
        from train_bag_v2 import BagV2Model, V4_DATA
        model = BagV2Model(device='cuda:0')
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.proj_e5.load_state_dict(ckpt['proj_e5_state_dict'])
        model.proj_formal.load_state_dict(ckpt['proj_formal_state_dict'])
        model.decoder.load_state_dict(ckpt['decoder_state_dict'])

    informal_pairs_map = {}
    with open(V4_DATA) as f:
        for line in f:
            s = _json.loads(line)
            pairs = s.get('mathlib_subgraph', {}).get('informal_pairs', [])
            gid   = s.get('graph_id')
            if gid and pairs:
                informal_pairs_map[gid] = pairs
    model.informal_pairs_map = informal_pairs_map

    model = model.float().eval()
    logger.info(f"Bag {version} model loaded from {ckpt_path} (epoch={ckpt.get('epoch')})")
    return model


@torch.no_grad()
def generate_bag(model, sample, max_new_tokens=300):
    """Generate text from bag_v1 or bag_v2 model."""
    phrase  = sample.get('phrase', '') or sample.get('target_title', '')
    phrases = [phrase] if phrase else None
    _, generated, _ = model(
        [sample], model.informal_pairs_map,
        target_texts=None, phrases=phrases,
        max_new_tokens=max_new_tokens,
    )
    return generated[0] if generated else ''


@torch.no_grad()
def generate_text_only(model, tokenizer, sample, max_new_tokens=300):
    """Generate text from text-only model."""
    from train_text_only import build_prompt
    prompt = build_prompt(sample)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def build_llemma_prompt(sample):
    """Build a plain few-shot completion prompt for Llemma (no [INST] tags)."""
    import sys
    from train_text_only import get_target_node, get_neighbor_nodes, first_two_sentences, ABSTRACT_SOURCES, MAX_NEIGHBORS, MAX_MATHLIB

    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    src    = sample.get('target_text_source', '')

    target_node  = get_target_node(sample) or {}
    abstract_raw = target_node.get('summery_text') or ''
    abstract     = (abstract_raw if isinstance(abstract_raw, str) else ' '.join(abstract_raw)).strip()

    neighbors = get_neighbor_nodes(sample)[:MAX_NEIGHBORS]
    neighbor_lines = []
    for n in neighbors:
        title_raw = n.get('title_text', '') or ''
        title = (title_raw if isinstance(title_raw, str) else ' '.join(title_raw)).strip()
        summ_raw = n.get('summery_text', '') or ''
        summ  = first_two_sentences(summ_raw if isinstance(summ_raw, str) else ' '.join(summ_raw))
        if title:
            neighbor_lines.append(f"- {title}: {summ}" if summ else f"- {title}")

    mathlib_thms  = sample.get('mathlib_subgraph', {}).get('subgraph_theorems', [])
    mathlib_lines = []
    for t in mathlib_thms[:MAX_MATHLIB]:
        name = t.get('name', '')
        stmt = str(t.get('statement', '')).strip()[:200]
        if name and stmt:
            mathlib_lines.append(f"- {name}: {stmt}")
        elif name:
            mathlib_lines.append(f"- {name}")

    neighbors_block = '\n'.join(neighbor_lines) if neighbor_lines else '(none)'
    mathlib_block   = '\n'.join(mathlib_lines)  if mathlib_lines  else '(none)'

    # Plain completion format — Llemma is a base model, not instruction-tuned
    if src in ABSTRACT_SOURCES:
        prompt = (
            f"Given a mathematics paper and related work, write the paper's abstract.\n\n"
            f"Paper: {phrase}\n\n"
            f"Related papers:\n{neighbors_block}\n\n"
            f"Related Mathlib theorems:\n{mathlib_block}\n\n"
            f"Abstract: "
        )
    else:
        prompt = (
            f"Given a mathematics paper and related work, predict the paper's main theorem.\n\n"
            f"Paper: {phrase}\n"
            f"Abstract: {abstract}\n\n"
            f"Related papers:\n{neighbors_block}\n\n"
            f"Related Mathlib theorems:\n{mathlib_block}\n\n"
            f"Main theorem: "
        )
    return prompt


@torch.no_grad()
def generate_llemma(model, tokenizer, sample, max_new_tokens=300, use_root_title=False):
    """Generate text with Llemma using plain completion prompt (no chat template)."""
    if use_root_title:
        title = _get_root_title(sample)
        if title:
            sample = dict(sample)
            sample['phrase'] = title
    prompt = build_llemma_prompt(sample)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
        no_repeat_ngram_size=6,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Llemma pretraining data bleeds in after '#' (Socratic Q&A format) — truncate there
    if '#' in text:
        text = text[:text.index('#')].strip()
    return text


def load_giants_model():
    """Load GIANTS-4B (Qwen3-4B fine-tuned on insight anticipation)."""
    import sys
    # Need transformers>=4.51 for Qwen3 support
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "giants2026/GIANTS-4B"
    logger.info(f"Loading GIANTS-4B from {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    logger.info("GIANTS-4B loaded.")
    return model, tok


@torch.no_grad()
def generate_giants(model, tokenizer, sample, max_new_tokens=300):
    """Generate using GIANTS-4B. Input: first 2 citation neighbor abstracts."""
    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
    abstracts = []
    for node in nodes:
        abst = (node.get('summery_text', '') or node.get('abstract', '') or node.get('text', ''))
        if abst and len(abst) > 50:
            abstracts.append(abst.strip())
        if len(abstracts) == 2:
            break

    if len(abstracts) < 2:
        # Pad with empty if fewer than 2 neighbors
        while len(abstracts) < 2:
            abstracts.append("No abstract available.")

    query = (
        f"Paper 1 summary: {abstracts[0]}\n\n"
        f"Paper 2 summary: {abstracts[1]}\n\n"
        f"Based on these two foundational papers, what is the key insight or main contribution of the downstream paper?"
    )
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=2048,   # must be large — Qwen3 thinks first then answers
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Strip Qwen3 thinking block — keep only the answer after </think>
    if '</think>' in decoded:
        decoded = decoded.split('</think>', 1)[1].strip()
    return decoded


def load_futuregen_model():
    """Load DeepSeek-Math-7B-Instruct for futuregen flat-RAG baseline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DECODER_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading futuregen base model: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('Futuregen model loaded')
    return model, tokenizer


_futuregen_pool = {}

def load_futuregen_pool():
    """Load v7 paper pool for futuregen RAG retrieval."""
    if _futuregen_pool:
        return _futuregen_pool
    logger.info(f"Loading retrieval index for futuregen RAG: {RETRIEVAL_IDX}")
    idx = np.load(RETRIEVAL_IDX, allow_pickle=True)
    _futuregen_pool['embeddings'] = idx['embeddings'].astype(np.float32)  # [N, 1024]
    _futuregen_pool['targets']    = idx['targets']     # Lean theorem text
    _futuregen_pool['arxiv_ids']  = idx['arxiv_ids'].astype(str)
    _futuregen_pool['phrases']    = idx['phrases']     # Topic phrase (used as title)
    logger.info(f"Futuregen pool: {_futuregen_pool['embeddings'].shape[0]} papers")
    return _futuregen_pool


def _get_anchor_paper(sample):
    """Find the anchor (target) paper node in the subgraph."""
    sg_id = sample['paper_subgraph']['subgraph_id']
    target_id = sg_id.split('_h2')[0] if '_h2' in sg_id else sg_id
    nodes = sample['paper_subgraph'].get('nodes') or []
    for n in nodes:
        if target_id in (n.get('paper_id') or ''):
            return n, target_id
    return (nodes[0] if nodes else {}), target_id


def build_futuregen_prompt(sample, retrieved_papers, anchor_node=None):
    """Build the FutureGen flat-RAG prompt.

    Matches the real FutureGen: model sees the anchor paper (title + abstract
    from graph node) plus retrieved prior papers. The ground-truth target_text
    is NOT included.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '') or ''

    anchor_title = ''
    anchor_abstract = ''
    if anchor_node:
        anchor_title = anchor_node.get('title_text', '') or ''
        anchor_abstract = (anchor_node.get('summery_text', '') or '')[:800]

    pieces = [
        'You are a scientific research assistant predicting future contributions.',
        'Given an anchor paper and retrieved prior work, generate one concise paragraph '
        '(under 100 words) predicting the most likely next key contribution '
        '— a new theorem, lemma, or technical result — that would follow from this line of work.',
        'Ground your prediction in the anchor paper and retrieved literature. Be specific and technical.',
        '',
        f'Research Topic: {phrase}',
        '',
        'Anchor Paper:',
        f'Title: {anchor_title}',
        f'Abstract: {anchor_abstract}',
        '',
        f'Retrieved Prior Papers ({len(retrieved_papers)}):',
    ]
    for i, item in enumerate(retrieved_papers, 1):
        pieces.extend([
            f"[{i}] Title: {item['title']}",
            f"[{i}] Abstract: {item['text'][:600]}",
            '',
        ])
    pieces.append('Based on the anchor paper and retrieved literature, predict the next key contribution. '
                   'Output only the prediction paragraph.')
    return '\n'.join(pieces)


@torch.no_grad()
def generate_futuregen(model, tokenizer, sample, pool=None, max_new_tokens=200):
    """Generate using futuregen flat-RAG: anchor paper + corpus-retrieved papers.

    Real FutureGen retrieves from a large external corpus, not just subgraph
    neighbors.  When *pool* is provided (from load_futuregen_pool()), we embed
    the anchor phrase with E5, retrieve the top-3 most similar papers from the
    full pool (excluding the anchor itself), and feed them into the prompt.
    Falls back to subgraph neighbors only when pool is None.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    arxiv_id = sample.get('target_arxiv_id', '')

    # Find anchor paper
    anchor_node, target_id = _get_anchor_paper(sample)

    if pool is not None and phrase:
        # --- corpus-level retrieval (real FutureGen) ---
        q_emb  = embed_texts([phrase], batch_size=1, prefix='query')   # [1, 1024]
        scores = (q_emb @ pool['embeddings'].T)[0]                    # [N]

        # Exclude the anchor paper itself
        pool_aids = pool['arxiv_ids']
        for idx in range(len(pool_aids)):
            if pool_aids[idx] == arxiv_id:
                scores[idx] = -1

        top_idx = np.argsort(scores)[::-1][:3]
        retrieved = []
        for idx in top_idx:
            retrieved.append({
                'title': str(pool['phrases'][idx]),
                'text':  str(pool['targets'][idx]),
            })
    else:
        # Fallback: subgraph neighbors (legacy behaviour)
        nodes = sample.get('paper_subgraph', {}).get('nodes', [])
        candidates = []
        for node in nodes:
            if target_id in (node.get('paper_id') or ''):
                continue
            title   = node.get('title_text', '') or ''
            summary = node.get('summery_text', '') or node.get('abstract', '') or ''
            if not title and not summary:
                continue
            candidates.append({'title': title, 'text': summary})
        if len(candidates) > 3 and phrase:
            node_texts = [f"{c['title']} {c['text'][:300]}" for c in candidates]
            q_emb   = embed_texts([phrase], batch_size=1, prefix='query')
            n_embs  = embed_texts(node_texts, batch_size=32, prefix='passage')
            scores  = (q_emb @ n_embs.T)[0]
            top_idx = np.argsort(scores)[::-1][:3]
            retrieved = [candidates[i] for i in top_idx]
        else:
            retrieved = candidates[:3]

    prompt = build_futuregen_prompt(sample, retrieved, anchor_node=anchor_node)

    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1536)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def load_goai_model():
    """Load DeepSeek-Math-7B-Instruct for GoAI-inspired graph-structured baseline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DECODER_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading GoAI base model: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('GoAI model loaded')
    return model, tokenizer


def build_goai_prompt(sample):
    """Build GoAI-inspired graph-structured prompt.

    Explicitly describes citation graph topology (nodes + edges) to the LLM,
    mirroring GoAI's approach of reasoning over typed citation relationships.
    Limits to 8 nodes and 12 edges to keep prompt within model context.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    nodes  = sample.get('paper_subgraph', {}).get('nodes', [])
    edges  = sample.get('paper_subgraph', {}).get('edges', [])

    MAX_NODES = 8
    MAX_EDGES = 12

    # Build paper_id -> short label + title map (limit nodes)
    pid_to_info = {}
    for idx, node in enumerate(nodes[:MAX_NODES]):
        pid   = node.get('paper_id', '')
        title = node.get('title_text', '') or ''
        summ  = (node.get('summery_text', '') or '')[:200]
        label = f"P{idx+1}"
        pid_to_info[pid] = {'label': label, 'title': title, 'summary': summ}

    # Paper descriptions
    paper_lines = []
    for pid, info in pid_to_info.items():
        paper_lines.append(
            f"{info['label']}: \"{info['title']}\"\n"
            f"   Summary: {info['summary']}"
        )

    # Citation edges as typed relationships (limit edges)
    edge_lines = []
    seen = set()
    for edge in edges:
        if len(edge_lines) >= MAX_EDGES:
            break
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        ref = edge.get('reference_text', '') or edge.get('match_value', '')
        src_info = pid_to_info.get(src)
        tgt_info = pid_to_info.get(tgt)
        if not src_info or not tgt_info:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        edge_lines.append(
            f"{src_info['label']} → {tgt_info['label']}  (cites: \"{ref[:60]}\")"
        )

    # Build graph context block
    graph_block = '=== Citation Graph — Paper Nodes ===\n'
    graph_block += '\n'.join(paper_lines)
    if edge_lines:
        graph_block += '\n\n=== Citation Graph — Edges (who cites whom) ===\n'
        graph_block += '\n'.join(edge_lines)

    # GoAI-style prompt: system instruction + graph context + generation task
    # Using chat format for DeepSeek-Math-Instruct
    prompt = (
        f"User: You are given a citation graph of mathematical research papers "
        f"and their relationships in the area of: {phrase}\n\n"
        f"{graph_block}\n\n"
        f"Based on the development trend visible in this citation graph, "
        f"predict one specific novel theorem or lemma that this line of work "
        f"will produce next. State only the mathematical claim — no explanation, "
        f"no references, no paper titles.\n\n"
        f"Assistant: The next theorem this research direction will produce is:\n"
    )
    return prompt


@torch.no_grad()
def generate_goai(model, tokenizer, sample, max_new_tokens=200):
    """Generate using GoAI-inspired graph-structured prompt over citation graph."""
    prompt = build_goai_prompt(sample)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1536)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# =============================================================================
# GoAI-Hybrid: GoAI prompt + theorem context from enc2
# =============================================================================

def build_goai_hybrid_prompt(sample, theorem_texts=None):
    """GoAI prompt augmented with theorems from enc2 knowledge base.

    Same citation-graph structure as GoAI, but appends the top-k
    Mathlib/arxiv theorems that enc2 retrieved for this subgraph,
    giving the LLM explicit formal math anchors.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    nodes  = sample.get('paper_subgraph', {}).get('nodes', [])
    edges  = sample.get('paper_subgraph', {}).get('edges', [])

    MAX_NODES = 8
    MAX_EDGES = 12

    pid_to_info = {}
    for idx, node in enumerate(nodes[:MAX_NODES]):
        pid   = node.get('paper_id', '')
        title = node.get('title_text', '') or ''
        summ  = (node.get('summery_text', '') or '')[:200]
        label = f"P{idx+1}"
        pid_to_info[pid] = {'label': label, 'title': title, 'summary': summ}

    paper_lines = []
    for pid, info in pid_to_info.items():
        paper_lines.append(
            f"{info['label']}: \"{info['title']}\"\n"
            f"   Summary: {info['summary']}"
        )

    edge_lines = []
    seen = set()
    for edge in edges:
        if len(edge_lines) >= MAX_EDGES:
            break
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        ref = edge.get('reference_text', '') or edge.get('match_value', '')
        src_info = pid_to_info.get(src)
        tgt_info = pid_to_info.get(tgt)
        if not src_info or not tgt_info:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        edge_lines.append(
            f"{src_info['label']} → {tgt_info['label']}  (cites: \"{ref[:60]}\")"
        )

    graph_block = '=== Citation Graph — Paper Nodes ===\n'
    graph_block += '\n'.join(paper_lines)
    if edge_lines:
        graph_block += '\n\n=== Citation Graph — Edges (who cites whom) ===\n'
        graph_block += '\n'.join(edge_lines)

    # Append theorem context if provided
    thm_block = ''
    if theorem_texts:
        thm_lines = '\n'.join(f"  - {t[:150]}" for t in theorem_texts[:4])
        thm_block = (
            f"\n\n=== Related Formal Theorems (from Mathlib / arXiv) ===\n"
            f"{thm_lines}"
        )

    prompt = (
        f"User: You are given a citation graph of mathematical research papers "
        f"and their relationships in the area of: {phrase}\n\n"
        f"{graph_block}"
        f"{thm_block}\n\n"
        f"Based on the development trend visible in this citation graph, "
        f"predict one specific novel theorem or lemma that this line of work "
        f"will produce next. State only the mathematical claim — no explanation, "
        f"no references, no paper titles.\n\n"
        f"Assistant: The next theorem this research direction will produce is:\n"
    )
    return prompt


@torch.no_grad()
def generate_goai_hybrid(goai_model, goai_tok, sample, max_new_tokens=200):
    """Generate using GoAI prompt augmented with Mathlib theorem context.

    Reads theorem statements directly from sample['mathlib_subgraph']['subgraph_theorems']
    (already present in the data — no extra model needed).
    Feeds top-k theorem statements into the GoAI prompt alongside the citation graph.
    """
    # Extract theorem texts from the mathlib subgraph in the sample
    theorem_texts = []
    mlib = sample.get('mathlib_subgraph', {})
    for t in mlib.get('subgraph_theorems', [])[:4]:
        stmt = t.get('statement', '') or ''
        name = t.get('name', '')
        if stmt:
            # Show name + first line of statement for clarity
            first_line = stmt.split('\n')[0][:120]
            theorem_texts.append(f"{name}: {first_line}" if name else first_line)

    prompt = build_goai_hybrid_prompt(sample, theorem_texts or None)
    enc = goai_tok(prompt, return_tensors='pt', truncation=True, max_length=1792)
    enc = {k: v.to(goai_model.device) for k, v in enc.items()}
    out = goai_model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=goai_tok.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return goai_tok.decode(gen_ids, skip_special_tokens=True)


# =============================================================================
# Chain of Ideas (CoI) baseline — Li et al. 2024, EMNLP 2025
# =============================================================================

def _build_coi_chains(sample, max_chains=3, max_len=5):
    """Extract ordered citation chains from the subgraph via DFS.

    Each chain is a list of paper dicts ordered by citation flow,
    representing a development trajectory through the literature.
    """
    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
    edges = sample.get('paper_subgraph', {}).get('edges', [])

    if not nodes:
        return []

    # Build adjacency list (source → targets)
    pid_to_node = {}
    for node in nodes:
        pid = node.get('paper_id', '')
        pid_to_node[pid] = node
    adj = defaultdict(list)
    for edge in edges:
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        if src in pid_to_node and tgt in pid_to_node:
            adj[src].append(tgt)

    # Find chains via DFS from nodes with no incoming edges (roots)
    has_incoming = set()
    for edge in edges:
        tgt = edge.get('target', '')
        if tgt in pid_to_node:
            has_incoming.add(tgt)
    roots = [pid for pid in pid_to_node if pid not in has_incoming]
    if not roots:
        roots = list(pid_to_node.keys())[:1]

    chains = []
    for root in roots:
        if len(chains) >= max_chains:
            break
        # DFS to build longest path from root
        stack = [(root, [root])]
        best_path = [root]
        while stack:
            node, path = stack.pop()
            if len(path) > len(best_path):
                best_path = path
            if len(path) >= max_len:
                continue
            for nb in adj.get(node, []):
                if nb not in path:
                    stack.append((nb, path + [nb]))
        chain = [pid_to_node[pid] for pid in best_path]
        chains.append(chain)

    # Fallback: if no chains found, just use first few nodes in order
    if not chains:
        chains = [nodes[:min(max_len, len(nodes))]]

    return chains


def _format_chain_for_prompt(chain, chain_idx):
    """Format a single citation chain as text for the prompt."""
    lines = [f"Chain {chain_idx}:"]
    for i, node in enumerate(chain):
        title = node.get('title_text', '') or 'Untitled'
        summ = (node.get('summery_text', '') or '')[:250]
        lines.append(f"  Paper {i+1}: \"{title}\"")
        if summ:
            lines.append(f"    Summary: {summ}")
    return '\n'.join(lines)


def build_coi_trend_prompt(sample, chains):
    """CoI Step 1: Summarize the research trend from citation chains.

    Aligned with CoI's get_deep_trend_idea_chains_prompt().
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')

    chains_text = '\n\n'.join(
        _format_chain_for_prompt(chain, i + 1) for i, chain in enumerate(chains)
    )

    prompt = (
        f"User: You are a research trend analyst in mathematics.\n"
        f"Research area: {phrase}\n\n"
        f"Below are citation chains showing how research has progressed in this area. "
        f"Each chain traces a development path through the literature.\n\n"
        f"{chains_text}\n\n"
        f"Examine the progression of ideas across these papers. "
        f"Detail how each paper transitions to the next, focusing on what "
        f"mathematical concepts, techniques, and results evolved. "
        f"Summarize the overall research trend in 2-3 sentences.\n\n"
        f"Assistant: Research trend: "
    )
    return prompt


def build_coi_idea_prompt(sample, trend_summary, chains):
    """CoI Step 2: Generate a specific theorem prediction using 4 thinking modes.

    Aligned with CoI's get_deep_generate_future_direciton_prompt() and
    get_deep_generate_idea_prompt().
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')

    # Brief chain summary (just titles for context)
    chain_titles = []
    for chain in chains:
        titles = [f"\"{n.get('title_text', '')[:80]}\"" for n in chain]
        chain_titles.append(' → '.join(titles))
    chains_brief = '\n'.join(chain_titles)

    prompt = (
        f"User: You are a mathematical research scientist predicting the next "
        f"contribution in an active research area.\n\n"
        f"Research area: {phrase}\n"
        f"Research trend: {trend_summary}\n"
        f"Paper chains: {chains_brief}\n\n"
        f"Apply four modes of reasoning to predict the next theorem or lemma:\n"
        f"1. Reflection: What gaps or open questions remain in this line of work?\n"
        f"2. Analogy: What parallel results from related areas could transfer here?\n"
        f"3. Deep Dive: What deeper mathematical structure is left unexplored?\n"
        f"4. Imitate: What techniques from the chain could be extended or generalized?\n\n"
        f"Based on these four perspectives, state ONE specific novel theorem or lemma "
        f"that this research direction will produce next. "
        f"Output only the mathematical claim — no explanation, no references.\n\n"
        f"Assistant: The next theorem this research will produce is:\n"
    )
    return prompt


def _run_coi_llm(model, tokenizer, prompt, max_new_tokens=150):
    """Run a single local LLM generation for CoI."""
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_coi(model, tokenizer, sample, max_new_tokens=200):
    """Generate using Chain of Ideas (CoI) baseline — Li et al. 2024.

    Pipeline: (1) extract citation chains, (2) summarize trend,
    (3) generate theorem using 4 thinking modes.
    """
    # Step 1: Extract citation chains from subgraph
    chains = _build_coi_chains(sample, max_chains=2, max_len=4)

    # Step 2: Trend analysis (first LLM call)
    trend_prompt = build_coi_trend_prompt(sample, chains)
    trend_summary = _run_coi_llm(model, tokenizer, trend_prompt, max_new_tokens=120)
    # Take first 2-3 sentences as trend
    trend_sentences = trend_summary.split('.')[:3]
    trend_summary = '.'.join(trend_sentences).strip()
    if trend_summary and not trend_summary.endswith('.'):
        trend_summary += '.'

    # Step 3: Idea generation with 4 thinking modes (second LLM call)
    idea_prompt = build_coi_idea_prompt(sample, trend_summary, chains)
    theorem = _run_coi_llm(model, tokenizer, idea_prompt, max_new_tokens=max_new_tokens)

    return theorem


def _coi_openai_call(client, prompt, max_tokens=200, model_name='gpt-4o-mini'):
    """Single OpenAI API call for CoI pipeline."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or '').strip()


def generate_coi_openai(sample, max_new_tokens=200, model_name='gpt-4o-mini'):
    """Generate using Chain of Ideas via OpenAI API."""
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # Step 1: Extract citation chains
    chains = _build_coi_chains(sample, max_chains=2, max_len=4)

    # Step 2: Trend analysis
    trend_prompt = build_coi_trend_prompt(sample, chains)
    # Remove User:/Assistant: format for OpenAI chat
    trend_prompt_clean = trend_prompt.replace('User: ', '').replace('Assistant: Research trend: ', '')
    trend_summary = _coi_openai_call(client, trend_prompt_clean, max_tokens=150, model_name=model_name)
    trend_sentences = trend_summary.split('.')[:3]
    trend_summary = '.'.join(trend_sentences).strip()
    if trend_summary and not trend_summary.endswith('.'):
        trend_summary += '.'

    # Step 3: Idea generation with 4 thinking modes
    idea_prompt = build_coi_idea_prompt(sample, trend_summary, chains)
    idea_prompt_clean = idea_prompt.replace('User: ', '').replace(
        'Assistant: The next theorem this research will produce is:\n', '')
    theorem = _coi_openai_call(client, idea_prompt_clean, max_tokens=max_new_tokens, model_name=model_name)

    return theorem


def load_coi_model(decoder_model=None):
    """Load decoder for Chain of Ideas baseline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DECODER_MODEL = decoder_model or 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading CoI base model: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('CoI model loaded')
    return model, tokenizer


def generate_openai(sample, mode='goai', max_new_tokens=200, model_name='gpt-4o-mini', pool=None):
    """Generate using OpenAI API. mode='goai' uses graph-structured prompt, 'futuregen' uses flat RAG."""
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    if mode == 'goai':
        prompt = build_goai_prompt(sample)
    else:
        # futuregen: anchor paper + corpus-retrieved papers (real FutureGen)
        anchor_node, target_id = _get_anchor_paper(sample)
        phrase = sample.get('phrase', '') or sample.get('target_title', '')
        arxiv_id = sample.get('target_arxiv_id', '')

        if pool is not None and phrase:
            q_emb  = embed_texts([phrase], batch_size=1, prefix='query')
            scores = (q_emb @ pool['embeddings'].T)[0]
            pool_aids = pool['arxiv_ids']
            for idx in range(len(pool_aids)):
                if pool_aids[idx] == arxiv_id:
                    scores[idx] = -1
            top_idx = np.argsort(scores)[::-1][:3]
            retrieved = []
            for idx in top_idx:
                retrieved.append({
                    'title': str(pool['phrases'][idx]),
                    'text':  str(pool['targets'][idx]),
                })
        else:
            nodes = sample.get('paper_subgraph', {}).get('nodes', [])
            retrieved = []
            for node in nodes:
                if target_id in (node.get('paper_id') or ''):
                    continue
                retrieved.append({
                    'title': node.get('title_text', '') or '',
                    'text':  (node.get('summery_text', '') or '')[:600],
                })
                if len(retrieved) == 3:
                    break
        prompt = build_futuregen_prompt(sample, retrieved, anchor_node=anchor_node)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or '').strip()


def load_future_aligned_model(decoder_model=None):
    """Load decoder for Future-Aligned Research Proposals baseline (Wang et al., 2026)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DECODER_MODEL = decoder_model or 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading Future-Aligned base model: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('Future-Aligned model loaded')
    return model, tokenizer


def build_future_aligned_prompt(sample):
    """Build Future-Aligned Research Proposals prompt (Wang et al., 2026).

    Uses their "Research Question + Papers" prompting strategy (Figure 5):
    given a research question and inspiring papers, generate a structured
    proposal with hypothesis, proposed method, novelty claims, and
    experimental details.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '') or ''
    nodes  = sample.get('paper_subgraph', {}).get('nodes', [])

    # Collect inspiring papers from the subgraph (up to 5)
    inspiring = []
    for node in nodes:
        title   = node.get('title_text', '') or ''
        summary = (node.get('summery_text', '') or
                   node.get('abstract', '') or
                   node.get('text', '') or '')
        if not title and not summary:
            continue
        inspiring.append({'title': title, 'summary': summary[:500]})
        if len(inspiring) >= 5:
            break

    # Format inspiring papers block
    papers_block = []
    for i, paper in enumerate(inspiring, 1):
        papers_block.append(f"Paper {i}: {paper['title']}")
        papers_block.append(f"Abstract: {paper['summary']}")
        papers_block.append('')

    papers_text = '\n'.join(papers_block)

    # Exact prompt structure from Wang et al. 2026, Figure 5 —
    # "Standard Generation Prompt" with research question + inspiring papers
    prompt = (
        f"You are an expert AI research scientist. Given inspiring research papers "
        f"and a target research question, propose a novel research idea that addresses the question.\n\n"
        f"Target Research Question: {phrase}\n\n"
        f"Inspiring Papers:\n{papers_text}\n"
        f"Your response should include:\n"
        f"- A proposed research idea with title, research question, hypothesis, "
        f"proposed method, novelty claims, and experiment details\n\n"
        f"Format your response starting with \"## Proposed Research\"."
    )
    return prompt


@torch.no_grad()
def generate_future_aligned(model, tokenizer, sample, max_new_tokens=300):
    """Generate using Future-Aligned Research Proposals prompting strategy."""
    prompt = build_future_aligned_prompt(sample)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# =============================================================================
# ResearchAgent baseline (Baek et al., 2024)
# Prompts aligned with Tables 6-11 of the paper.
# Entity-centric knowledge augmentation substituted with E5 corpus retrieval.
# Step 3 adapted: experiment design → contribution prediction (task-specific).
# =============================================================================

def load_researchagent_model():
    """Load DeepSeek-Math-7B-Instruct for ResearchAgent baseline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DECODER_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
    logger.info(f"Loading ResearchAgent base model: {DECODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, torch_dtype=torch.float32, device_map='auto',
    )
    model.eval()
    logger.info('ResearchAgent model loaded')
    return model, tokenizer


def _gather_citation_context(sample, pool=None):
    """Gather related papers from subgraph neighbors + corpus retrieval.

    Returns a list of {'title': ..., 'text': ...} dicts (up to 5 papers).
    Mirrors ResearchAgent's citation-graph literature survey + entity retrieval.
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    arxiv_id = sample.get('target_arxiv_id', '')
    anchor_node, target_id = _get_anchor_paper(sample)
    nodes = sample.get('paper_subgraph', {}).get('nodes', [])

    # Subgraph neighbors (citation graph context)
    neighbors = []
    for node in nodes:
        if target_id in (node.get('paper_id') or ''):
            continue
        title   = node.get('title_text', '') or ''
        summary = node.get('summery_text', '') or node.get('abstract', '') or ''
        if not title and not summary:
            continue
        neighbors.append({'title': title, 'text': summary})

    # Corpus retrieval (entity-centric knowledge augmentation substitute)
    corpus_papers = []
    if pool is not None and phrase:
        q_emb  = embed_texts([phrase], batch_size=1, prefix='query')
        scores = (q_emb @ pool['embeddings'].T)[0]
        pool_aids = pool['arxiv_ids']
        for idx in range(len(pool_aids)):
            if pool_aids[idx] == arxiv_id:
                scores[idx] = -1
        top_idx = np.argsort(scores)[::-1][:3]
        for idx in top_idx:
            corpus_papers.append({
                'title': str(pool['phrases'][idx]),
                'text':  str(pool['targets'][idx]),
            })

    # Merge: up to 2 subgraph neighbors + up to 3 corpus papers = 5 max
    related = neighbors[:2] + corpus_papers[:3]
    return related, anchor_node


def _format_related_titles(papers):
    """Format related paper titles as comma-separated list."""
    return ', '.join(p['title'] for p in papers if p['title'])


def _format_related_abstracts(papers):
    """Format related paper abstracts as comma-separated list."""
    return ', '.join(p['text'][:400] for p in papers if p['text'])


def _format_entities(papers):
    """Format corpus-retrieved papers as entity-like concepts."""
    # In the original paper, entities come from BLINK entity linker.
    # We use corpus-retrieved paper topics as a substitute.
    return ', '.join(p['title'] for p in papers if p['title'])


# ---------------------------------------------------------------------------
# Step 1: Problem Identification (Table 6)
# ---------------------------------------------------------------------------
def _build_researchagent_step1(anchor_title, anchor_abstract, related_papers, entities):
    """Build problem identification prompt aligned with Table 6."""
    n_refs = len(related_papers)
    n_ents = len(entities.split(',')) if entities else 0
    related_titles = _format_related_titles(related_papers)
    related_abstracts = _format_related_abstracts(related_papers)

    system = (
        'You are an AI assistant whose primary goal is to identify promising, new, and key scientific '
        'problems based on existing scientific literature, in order to aid researchers in discovering novel '
        'and significant research opportunities that can advance the field.'
    )

    user = f"""{system}

You are going to generate a research problem that should be original, clear, feasible, relevant, and \
significant to its field. This will be based on the title and abstract of the target paper, those of \
{n_refs} related papers in the existing literature, and {n_ents} entities potentially \
connected to the research area.

Understanding of the target paper, related papers, and entities is essential:
- The target paper is the primary research study you aim to enhance or build upon through future \
research, serving as the central source and focus for identifying and developing the specific \
research problem.
- The related papers are studies that have cited the target paper, indicating their direct relevance \
and connection to the primary research topic you are focusing on, and providing additional context \
and insights that are essential for understanding and expanding upon the target paper.
- The entities can include topics, keywords, individuals, events, or any subjects with possible \
direct or indirect connections to the target paper or the related studies, serving as auxiliary sources \
of inspiration or information that may be instrumental in formulating the research problem.

Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the target paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective \
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of \
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the target paper, related papers, and entities, as follows:
Target paper title: {anchor_title}
Target paper abstract: {anchor_abstract}
Related paper titles: {related_titles}
Related paper abstracts: {related_abstracts}
Entities: {entities}

With the provided target paper, related papers, and entities, your objective now is to formulate a \
research problem that not only builds upon these existing studies but also strives to be original, \
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title \
and abstract of the target paper, to ensure it remains the focal point of your research problem \
identification process.

Target paper title: {anchor_title}
Target paper abstract: {anchor_abstract}

Then, following your review of the above content, please proceed to generate one research \
problem with the rationale, in the format of
Problem:
Rationale:"""

    return user


# ---------------------------------------------------------------------------
# Step 2: Method Development (Table 7)
# ---------------------------------------------------------------------------
def _build_researchagent_step2(anchor_title, anchor_abstract, problem, problem_rationale,
                                related_papers, entities):
    """Build method development prompt aligned with Table 7."""
    related_titles = _format_related_titles(related_papers)
    related_abstracts = _format_related_abstracts(related_papers)

    system = (
        'You are an AI assistant whose primary goal is to propose innovative, rigorous, and valid method'
        'ologies to solve newly identified scientific problems derived from existing scientific literature, in '
        'order to empower researchers to pioneer groundbreaking solutions that catalyze breakthroughs in '
        'their fields.'
    )

    user = f"""{system}

You are going to propose a scientific method to address a specific research problem. Your method \
should be clear, innovative, rigorous, valid, and generalizable. This will be based on a deep \
understanding of the research problem, its rationale, existing studies, and various entities.

Understanding of the research problem, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a \
potential exploration of relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as \
well as the related papers that have been additionally referenced in the problem discovery phase, \
all serving as foundational material for developing the method.
- The entities can include topics, keywords, individuals, events, or any subjects with possible \
direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or \
information that may be instrumental in method development.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale, to understand your primary \
focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective \
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of \
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, existing studies (target paper & related papers), and \
entities, as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Target paper title: {anchor_title}
Target paper abstract: {anchor_abstract}
Related paper titles: {related_titles}
Related paper abstracts: {related_abstracts}
Entities: {entities}

With the provided research problem, existing studies, your objective now is to \
formulate a method that not only leverages these resources but also strives to be clear, innovative, \
rigorous, valid, and generalizable. Before crafting the method, revisit the research problem, to \
ensure it remains the focal point of your method development process.

Research problem: {problem}
Rationale: {problem_rationale}

Then, following your review of the above content, please proceed to propose your method with \
its rationale, in the format of
Method:
Rationale:"""

    return user


# ---------------------------------------------------------------------------
# Step 3: Contribution Prediction (adapted from Table 8 — experiment design)
# ---------------------------------------------------------------------------
def _build_researchagent_step3(anchor_title, anchor_abstract, problem, problem_rationale,
                                method, method_rationale, related_papers, entities):
    """Build contribution prediction prompt (adapted from Table 8 experiment design)."""
    related_titles = _format_related_titles(related_papers)
    related_abstracts = _format_related_abstracts(related_papers)

    system = (
        'You are an AI assistant whose primary goal is to predict concrete future scientific contributions '
        '— theorems, lemmas, or technical results — based on identified scientific problems and proposed '
        'methodologies from existing scientific literature, in order to enable researchers to anticipate '
        'and validate groundbreaking discoveries that can transform their respective fields.'
    )

    user = f"""{system}

You are going to predict a concrete scientific contribution (theorem, lemma, or technical result) \
that would follow from a proposed method addressing a specific research problem. Your prediction \
should be clear, specific, technically grounded, valid, and feasible. This will be based on a deep \
understanding of the research problem, scientific method, existing studies, and various entities.

Understanding of the research problem, scientific method, existing studies, and entities is essential:
- The research problem has been formulated based on an in-depth review of existing studies and a \
potential exploration of relevant entities.
- The scientific method has been proposed to tackle the research problem, which has been \
informed by insights gained from existing studies and relevant entities.
- The existing studies refer to the target paper that has been pivotal in identifying the problem and \
method, as well as the related papers that have been additionally referenced in the discovery phase \
of the problem and method, all serving as foundational material for predicting the contribution.
- The entities can include topics, keywords, individuals, events, or any subjects with possible \
direct or indirect connections to the existing studies, serving as auxiliary sources of inspiration or \
information that may be instrumental in your prediction.

Your approach should be systematic:
- Start by thoroughly reading the research problem and its rationale followed by the proposed \
method and its rationale, to pinpoint your primary focus.
- Next, proceed to review the titles and abstracts of existing studies, to gain a broader perspective \
and insights relevant to the primary research topic.
- Finally, explore the entities to further broaden your perspective, drawing upon a diverse pool of \
inspiration and information, while keeping in mind that not all may be relevant.

I am going to provide the research problem, scientific method, existing studies (target paper & \
related papers), and entities, as follows:
Research problem: {problem}
Rationale: {problem_rationale}
Scientific method: {method}
Rationale: {method_rationale}
Target paper title: {anchor_title}
Target paper abstract: {anchor_abstract}
Related paper titles: {related_titles}
Related paper abstracts: {related_abstracts}
Entities: {entities}

With the provided research problem, scientific method, existing studies, your \
objective now is to predict a contribution that not only leverages these resources but also strives to \
be clear, specific, technically grounded, valid, and feasible. Before crafting the prediction, revisit \
the research problem and proposed method, to ensure they remain at the center of your prediction \
process.

Research problem: {problem}
Rationale: {problem_rationale}
Scientific method: {method}
Rationale: {method_rationale}

Then, following your review of the above content, please proceed to state your predicted \
contribution with its rationale, in the format of
Contribution:
Rationale:"""

    return user


# ---------------------------------------------------------------------------
# ReviewingAgent — contribution review (adapted from Tables 9-11)
# Criteria adapted from Table 12 for contribution prediction.
# ---------------------------------------------------------------------------

# Review criteria for predicted contributions (adapted from Table 12).
# The paper reviews problems (5 criteria), methods (5), experiments (5).
# We merge the most relevant ones for our task: a single predicted contribution.
_REVIEW_CRITERIA = [
    ('Clarity', 'It assesses whether the predicted contribution is defined in a clear, precise, '
     'and understandable manner.'),
    ('Relevance', 'It measures whether the prediction is pertinent and applicable to the current '
     'field or context of study.'),
    ('Originality', 'It evaluates whether the prediction presents a novel challenge or unique '
     'perspective that has not been extensively explored before.'),
    ('Validity', 'It measures the accuracy, relevance, and soundness of the prediction in addressing '
     'the research problem, ensuring that it is appropriate and directly relevant to the objectives '
     'of the study.'),
    ('Feasibility', 'It examines whether the predicted contribution can realistically be investigated '
     'or verified with the available resources and within reasonable constraints.'),
]


def _build_researchagent_review(anchor_title, anchor_abstract, related_papers,
                                 prediction, prediction_rationale, metric_name, criteria_text):
    """Build ReviewingAgent prompt aligned with Tables 9-11.

    Each review evaluates one metric at a time, with the existing studies as context.
    """
    related_titles = _format_related_titles(related_papers)
    related_abstracts = _format_related_abstracts(related_papers)

    system = (
        'You are an AI assistant whose primary goal is to assess the quality and validity of scientific '
        'predictions across diverse dimensions, in order to aid researchers in refining their predictions based '
        'on your evaluations and feedback, thereby enhancing the impact and reach of their work.'
    )

    user = f"""{system}

You are going to evaluate a predicted scientific contribution for its {metric_name}, focusing on how well it is defined \
in a clear, precise, and understandable manner.

As part of your evaluation, you can refer to the existing studies that may be related to the prediction, \
which will help in understanding the context of the prediction for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as \
well as the related papers that have been additionally referenced in the discovery phase of the \
prediction.

The existing studies (target paper & related papers) are as follows:
Target paper title: {anchor_title}
Target paper abstract: {anchor_abstract}
Related paper titles: {related_titles}
Related paper abstracts: {related_abstracts}

Now, proceed with your {metric_name} evaluation approach that should be systematic:
- Start by thoroughly reading the predicted contribution and its rationale, keeping in mind the context \
provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing \
on the {metric_name} of the prediction.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a \
discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless \
fully justified:
{criteria_text}

I am going to provide the predicted contribution with its rationale, as follows:
Predicted contribution: {prediction}
Rationale: {prediction_rationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in \
the format of
Review:
Feedback:
Rating (1-5):"""

    return user


def _parse_problem_output(text):
    """Parse 'Problem:\nRationale:' format from step 1."""
    problem, rationale = text, ''
    if 'Problem:' in text:
        parts = text.split('Problem:', 1)[1]
        if 'Rationale:' in parts:
            problem = parts.split('Rationale:', 1)[0].strip()
            rationale = parts.split('Rationale:', 1)[1].strip()
        else:
            problem = parts.strip()
    return problem, rationale


def _parse_method_output(text):
    """Parse 'Method:\nRationale:' format from step 2."""
    method, rationale = text, ''
    if 'Method:' in text:
        parts = text.split('Method:', 1)[1]
        if 'Rationale:' in parts:
            method = parts.split('Rationale:', 1)[0].strip()
            rationale = parts.split('Rationale:', 1)[1].strip()
        else:
            method = parts.strip()
    return method, rationale


def _parse_contribution_output(text):
    """Parse 'Contribution:\nRationale:' format from step 3."""
    contribution, rationale = text, ''
    if 'Contribution:' in text:
        parts = text.split('Contribution:', 1)[1]
        if 'Rationale:' in parts:
            contribution = parts.split('Rationale:', 1)[0].strip()
            rationale = parts.split('Rationale:', 1)[1].strip()
        else:
            contribution = parts.strip()
    return contribution, rationale


def _parse_review_feedback(text):
    """Extract the Feedback section from a review response."""
    feedback = text
    if 'Feedback:' in text:
        feedback = text.split('Feedback:', 1)[1]
        if 'Rating' in feedback:
            feedback = feedback.split('Rating')[0]
        feedback = feedback.strip()
    return feedback


@torch.no_grad()
def _run_llm(model, tokenizer, prompt, max_new_tokens=150):
    """Run a single LLM generation pass."""
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, enc['input_ids'].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_researchagent(model, tokenizer, sample, pool=None, max_new_tokens=200, n_refine=3):
    """Generate using ResearchAgent pipeline (Baek et al., 2024).

    Three-step structured generation aligned with Tables 6-8:
      1. Problem identification (Table 6)
      2. Method development (Table 7)
      3. Contribution prediction (adapted from Table 8)
    Followed by n_refine iterations of ReviewingAgent review (Tables 9-11)
    across 5 criteria (Table 12).
    """
    phrase = sample.get('phrase', '') or sample.get('target_title', '')
    related_papers, anchor_node = _gather_citation_context(sample, pool=pool)
    anchor_title = (anchor_node or {}).get('title_text', '') or ''
    anchor_abstract = ((anchor_node or {}).get('summery_text', '') or '')[:800]
    entities = _format_entities(related_papers)

    # Step 1: Problem Identification (Table 6)
    prompt1 = _build_researchagent_step1(anchor_title, anchor_abstract, related_papers, entities)
    raw1 = _run_llm(model, tokenizer, prompt1, max_new_tokens=200)
    problem, problem_rationale = _parse_problem_output(raw1)

    # Step 2: Method Development (Table 7)
    prompt2 = _build_researchagent_step2(anchor_title, anchor_abstract, problem, problem_rationale,
                                          related_papers, entities)
    raw2 = _run_llm(model, tokenizer, prompt2, max_new_tokens=200)
    method, method_rationale = _parse_method_output(raw2)

    # Step 3: Contribution Prediction (adapted from Table 8)
    prompt3 = _build_researchagent_step3(anchor_title, anchor_abstract, problem, problem_rationale,
                                          method, method_rationale, related_papers, entities)
    raw3 = _run_llm(model, tokenizer, prompt3, max_new_tokens=max_new_tokens)
    prediction, pred_rationale = _parse_contribution_output(raw3)

    # Iterative refinement with ReviewingAgent (Tables 9-11)
    # Each round: review across all 5 criteria, collect feedback, regenerate.
    for _ in range(n_refine):
        all_feedback = []
        for metric_name, criteria_text in _REVIEW_CRITERIA:
            review_prompt = _build_researchagent_review(
                anchor_title, anchor_abstract, related_papers,
                prediction, pred_rationale, metric_name, criteria_text,
            )
            review_out = _run_llm(model, tokenizer, review_prompt, max_new_tokens=200)
            fb = _parse_review_feedback(review_out)
            all_feedback.append(f"[{metric_name}] {fb}")

        # Regenerate contribution with feedback incorporated
        feedback_block = '\n'.join(all_feedback)
        refine_prompt = f"""{_build_researchagent_step3(
            anchor_title, anchor_abstract, problem, problem_rationale,
            method, method_rationale, related_papers, entities)}

The following reviewer feedback has been provided on the previous prediction. \
Please revise your prediction to address this feedback:
{feedback_block}

Revised contribution:
Rationale:"""
        raw_revised = _run_llm(model, tokenizer, refine_prompt, max_new_tokens=max_new_tokens)
        prediction, pred_rationale = _parse_contribution_output(raw_revised)

    return prediction


def _openai_call(client, prompt, max_tokens=150, model_name='gpt-4o-mini'):
    """Single OpenAI chat completion call."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or '').strip()


def generate_researchagent_openai(sample, pool=None, max_new_tokens=200, n_refine=3, model_name='gpt-4o-mini'):
    """ResearchAgent pipeline using OpenAI API (Baek et al., 2024).

    Same 3-step + iterative review pipeline as the local variant,
    but uses OpenAI chat completions instead of local DeepSeek.
    """
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    related_papers, anchor_node = _gather_citation_context(sample, pool=pool)
    anchor_title = (anchor_node or {}).get('title_text', '') or ''
    anchor_abstract = ((anchor_node or {}).get('summery_text', '') or '')[:800]
    entities = _format_entities(related_papers)

    # Step 1: Problem Identification (Table 6)
    prompt1 = _build_researchagent_step1(anchor_title, anchor_abstract, related_papers, entities)
    raw1 = _openai_call(client, prompt1, max_tokens=300, model_name=model_name)
    problem, problem_rationale = _parse_problem_output(raw1)

    # Step 2: Method Development (Table 7)
    prompt2 = _build_researchagent_step2(anchor_title, anchor_abstract, problem, problem_rationale,
                                          related_papers, entities)
    raw2 = _openai_call(client, prompt2, max_tokens=300, model_name=model_name)
    method, method_rationale = _parse_method_output(raw2)

    # Step 3: Contribution Prediction (adapted from Table 8)
    prompt3 = _build_researchagent_step3(anchor_title, anchor_abstract, problem, problem_rationale,
                                          method, method_rationale, related_papers, entities)
    raw3 = _openai_call(client, prompt3, max_tokens=max_new_tokens, model_name=model_name)
    prediction, pred_rationale = _parse_contribution_output(raw3)

    # Iterative refinement with ReviewingAgent (Tables 9-11)
    for _ in range(n_refine):
        all_feedback = []
        for metric_name, criteria_text in _REVIEW_CRITERIA:
            review_prompt = _build_researchagent_review(
                anchor_title, anchor_abstract, related_papers,
                prediction, pred_rationale, metric_name, criteria_text,
            )
            review_out = _openai_call(client, review_prompt, max_tokens=300, model_name=model_name)
            fb = _parse_review_feedback(review_out)
            all_feedback.append(f"[{metric_name}] {fb}")

        feedback_block = '\n'.join(all_feedback)
        refine_prompt = f"""{_build_researchagent_step3(
            anchor_title, anchor_abstract, problem, problem_rationale,
            method, method_rationale, related_papers, entities)}

The following reviewer feedback has been provided on the previous prediction. \
Please revise your prediction to address this feedback:
{feedback_block}

Revised contribution:
Rationale:"""
        raw_revised = _openai_call(client, refine_prompt, max_tokens=max_new_tokens, model_name=model_name)
        prediction, pred_rationale = _parse_contribution_output(raw_revised)

    return prediction


def load_retrieval_index():
    """Load retrieval index."""
    logger.info(f"Loading retrieval index: {RETRIEVAL_IDX}")
    idx = np.load(RETRIEVAL_IDX, allow_pickle=True)
    return {
        'embeddings': idx['embeddings'],
        'targets':    idx['targets'],
        'arxiv_ids':  idx['arxiv_ids'],
        'phrases':    idx['phrases'],
    }



def generate_retrieval(retrieval_idx, sample):
    """Retrieve nearest neighbor target from v7 index using anchor abstract + neighbors."""
    arxiv_id = sample.get('target_arxiv_id', '')

    # Build richer query: anchor abstract + up to 3 neighbor abstracts
    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
    sg_id = sample.get('paper_subgraph', {}).get('subgraph_id', '')
    target_id = sg_id.split('_h2')[0] if '_h2' in sg_id else sg_id
    parts = []
    neighbors = []
    for n in nodes:
        abst = (n.get('summery_text', '') or n.get('abstract', '') or '').strip()
        if target_id in (n.get('paper_id', '') or ''):
            if abst: parts.insert(0, abst[:400])
        else:
            if abst: neighbors.append(abst[:200])
    parts += neighbors[:3]
    query = ' '.join(parts).strip() or (sample.get('phrase', '') or sample.get('target_title', ''))

    # Embed the query
    q_emb = embed_texts([query], batch_size=1, prefix='query')  # [1, 1024]

    # Cosine similarity (both L2-normalized)
    scores = q_emb @ retrieval_idx['embeddings'].T  # [1, N]
    scores = scores[0]

    top_idx = np.argmax(scores)
    # Avoid returning same paper
    if retrieval_idx['arxiv_ids'][top_idx] == arxiv_id:
        scores[top_idx] = -1
        top_idx = np.argmax(scores)

    return str(retrieval_idx['targets'][top_idx])


# =============================================================================
# Main evaluation loop
# =============================================================================

def evaluate(args):
    from compute_metrics import (future_paper_metrics, retrieval_rank_metrics_multi,
                                  build_pool_from_future_embs,
                                  generation_quality_metrics, distinct_metrics,
                                  citation_relevance_metrics, gt_generation_metrics,
                                  input_novelty_metrics)

    # Load future paper embeddings
    paper_id_to_embs, sg_to_arxids = load_future_embs()

    # Build correct pool: all unique future papers (14K) — one embedding per paper
    logger.info("Building future paper pool for rank metrics...")
    pool_embs, pool_paper_ids = build_pool_from_future_embs(paper_id_to_embs)
    logger.info(f"Pool: {pool_embs.shape[0]} unique future papers")

    # Load canonical subgraph IDs filter if specified
    canonical_sg_ids = None
    if getattr(args, 'subgraph_ids_file', None):
        with open(args.subgraph_ids_file) as _f:
            _canon = json.load(_f)
        canonical_sg_ids = set(_canon['subgraph_ids'])
        logger.info(f"Canonical subset: {len(canonical_sg_ids)} subgraph IDs loaded from {args.subgraph_ids_file}")

    # Load val samples
    val_samples = load_val_samples(sg_to_arxids, n=args.n, skip=args.skip, canonical_sg_ids=canonical_sg_ids)
    if not val_samples:
        logger.error("No eligible val samples found.")
        return

    # Load the chosen model
    model_name = args.model
    gen_model = None
    gen_tokenizer = None
    retrieval_idx = None
    hybrid_dual_model = None

    dec_model = getattr(args, 'decoder_model', None)

    # Prompt variant templates (only used when --prompt_variant is set)
    PROMPT_VARIANTS = {
        'theorem_prefix': (
            "[INST] Research direction: {phrase}\nUsing the provided paper context, state the main theorem of this paper. Output only a clean, formal theorem statement. [/INST] Theorem: ",
            "[INST] Using the provided papers and theorems, state the main theorem. Output only a clean formal theorem statement. [/INST] Theorem: ",
        ),
        'goai_style': (
            "[INST] You are a mathematician writing a research paper on {phrase}. Based on the provided paper context, state the main theorem formally. Begin directly with 'Theorem:' or 'Lemma:'. [/INST] ",
            "[INST] You are a mathematician. Based on the provided paper context, state the main theorem formally. Begin directly with 'Theorem:' or 'Lemma:'. [/INST] ",
        ),
        'abstract_style': (
            "[INST] Research direction: {phrase}\nUsing the provided paper context, write a concise arxiv-style abstract for this paper in 2-3 sentences. [/INST] In this paper, ",
            "[INST] Using the provided papers and theorems, write a concise arxiv-style abstract in 2-3 sentences. [/INST] In this paper, ",
        ),
        'abstract_we': (
            "[INST] You are a mathematician. Research direction: {phrase}\nBased on the provided paper context, write a short abstract describing what this paper proves. [/INST] We ",
            "[INST] You are a mathematician. Based on the provided paper context, write a short abstract describing what this paper proves. [/INST] We ",
        ),
        # ── 5 forward-looking inference variants ──────────────────────────────
        'future_we_prove': (
            "[INST] Given papers about {phrase}, predict the main result of a future follow-up paper in this area. [/INST] We prove that ",
            "[INST] Given these papers, predict the main result of a future follow-up paper in this area. [/INST] We prove that ",
        ),
        'future_theorem': (
            "[INST] Given papers about {phrase}, state the main theorem that a future paper in this research direction will prove. Output only the theorem statement. [/INST] Theorem: Let ",
            "[INST] Given these papers, state the main theorem that a future paper will prove. Output only the theorem statement. [/INST] Theorem: Let ",
        ),
        'future_in_paper': (
            "[INST] Based on the research on {phrase} in the provided papers, describe the key result of a future paper that cites this work. [/INST] In this paper, we ",
            "[INST] Based on the provided papers, describe the key result of a future paper that cites this work. [/INST] In this paper, we ",
        ),
        'future_building': (
            "[INST] Papers about {phrase} are provided. Predict a theorem from a future paper that builds on this work. [/INST] Building on these results, we show that ",
            "[INST] The provided papers are given. Predict a theorem from a future paper that builds on this work. [/INST] Building on these results, we show that ",
        ),
        'future_extend': (
            "[INST] Research context: {phrase}. Using the provided paper context, write the abstract of a future paper that extends these results. [/INST] We extend ",
            "[INST] Using the provided paper context, write the abstract of a future paper that extends these results. [/INST] We extend ",
        ),
    }

    def apply_target_title_variant(model):
        """Use target paper title as the phrase — upper bound oracle test."""
        import model_clean as mc
        import types
        _target_sample_ref = [None]

        def patched(self, encoder_states, encoder_mask, phrases,
                    max_new_tokens=200, root_indices=None,
                    citation_importance_scores=None, type_ids=None, **generation_kwargs):
            batch_size = encoder_states.size(0)
            sample = _target_sample_ref[0]
            prompts = []
            for b in range(batch_size):
                if sample is not None:
                    title = sample.get('target_title', '') or sample.get('phrase', '')
                    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
                    paper_lines = []
                    for i, n in enumerate(nodes[:4]):
                        t = n.get('title_text', '') or ''
                        s = (n.get('summery_text', '') or '')[:200]
                        if t:
                            paper_lines.append(f"Paper {i+1}: {t}\n  {s}" if s else f"Paper {i+1}: {t}")
                    papers_ctx = '\n'.join(paper_lines)
                    prompt = (
                        f"[INST] The target paper is titled: \"{title}\"\n\n"
                        f"Context papers:\n{papers_ctx}\n\n"
                        f"Write the main theorem or key result of the target paper. "
                        f"State only the mathematical claim. [/INST] Theorem: "
                    )
                else:
                    phrase = phrases[b] if phrases and len(phrases) > b else ''
                    prompt = f"[INST] Research direction: {phrase}\nPredict the main theorem. [/INST] Theorem: "
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(self.device)
            generated_texts = self._generate_with_cross_attention(
                input_ids=inputs['input_ids'],
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.5, length_penalty=1.0, no_repeat_ngram_size=3,
                citation_importance_scores=citation_importance_scores,
                type_ids=type_ids, **generation_kwargs,
            )
            return None, generated_texts, None

        mc.MistralDecoder._inference_forward = patched
        model.decoder._inference_forward = types.MethodType(patched, model.decoder)
        model._rich_ctx_sample_ref = _target_sample_ref
        logger.info("Applied target_title variant: using actual target title as oracle prompt")

    def apply_prompt_variant(model, variant_name):
        """Patch _inference_forward prompt strings on the decoder."""
        import model_clean as mc
        phrase_prompt, fallback_prompt = PROMPT_VARIANTS[variant_name]
        orig = mc.MistralDecoder._inference_forward
        def patched(self, encoder_states, encoder_mask, phrases,
                    max_new_tokens=200, root_indices=None,
                    citation_importance_scores=None, type_ids=None, **generation_kwargs):
            batch_size = encoder_states.size(0)
            if phrases is not None and len(phrases) == batch_size:
                prompts = [phrase_prompt.format(phrase=p) for p in phrases]
            else:
                prompts = [fallback_prompt] * batch_size
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
            generated_texts = self._generate_with_cross_attention(
                input_ids=inputs['input_ids'],
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.5, length_penalty=1.0, no_repeat_ngram_size=3,
                citation_importance_scores=citation_importance_scores,
                type_ids=type_ids, **generation_kwargs,
            )
            return None, generated_texts, None
        mc.MistralDecoder._inference_forward = patched
        logger.info(f"Applied prompt variant: {variant_name}")

    def apply_rich_context_variant(model):
        """Patch _inference_forward to use GoAI-style rich graph prompt + cross-attention.

        This gives the decoder BOTH: (1) explicit paper titles/summaries in the prompt text
        (so the LLM can reference specific terms for good E5-Mistral embedding), AND
        (2) cross-attention to the fused encoder states (deep graph reasoning).
        """
        import model_clean as mc
        _rich_ctx_sample_ref = [None]  # mutable ref to current sample

        def patched(self, encoder_states, encoder_mask, phrases,
                    max_new_tokens=200, root_indices=None,
                    citation_importance_scores=None, type_ids=None, **generation_kwargs):
            sample = _rich_ctx_sample_ref[0]
            batch_size = encoder_states.size(0)

            # Build rich prompt from sample's graph context (like GoAI but shorter)
            prompts = []
            for b in range(batch_size):
                phrase = phrases[b] if phrases and len(phrases) > b else ''
                if sample is not None:
                    nodes = sample.get('paper_subgraph', {}).get('nodes', [])
                    # Build GoAI-style prompt with full paper context
                    paper_lines = []
                    for i, n in enumerate(nodes[:6]):
                        t = n.get('title_text', '') or ''
                        s = (n.get('summery_text', '') or '')[:250]
                        if t:
                            paper_lines.append(f"Paper {i+1}: {t}\n  {s}" if s else f"Paper {i+1}: {t}")
                    papers_ctx = '\n'.join(paper_lines)
                    prompt = (
                        f"[INST] You are given mathematical research papers in the area of: {phrase}\n\n"
                        f"{papers_ctx}\n\n"
                        f"Based on these papers, predict one specific novel theorem or lemma "
                        f"that this line of work will produce next. "
                        f"State only the mathematical claim. [/INST] Theorem: "
                    )
                else:
                    prompt = f"[INST] Research direction: {phrase}\nPredict the main theorem. [/INST] Theorem: "
                prompts.append(prompt)

            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(self.device)
            generated_texts = self._generate_with_cross_attention(
                input_ids=inputs['input_ids'],
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.5, length_penalty=1.0, no_repeat_ngram_size=3,
                citation_importance_scores=citation_importance_scores,
                type_ids=type_ids, **generation_kwargs,
            )
            return None, generated_texts, None

        mc.MistralDecoder._inference_forward = patched
        model._rich_ctx_sample_ref = _rich_ctx_sample_ref
        logger.info("Applied rich_context variant: decoder gets explicit graph text + cross-attention")

    def apply_goai_full_variant(model):
        """Patch _inference_forward to use the full GoAI prompt (8 nodes, 12 edges, summaries)
        fed into our decoder, with cross-attention encoder states on top.
        Best of both worlds: GoAI text context + our graph encoder signal.
        """
        import model_clean as mc
        _goai_sample_ref = [None]

        def patched(self, encoder_states, encoder_mask, phrases,
                    max_new_tokens=200, root_indices=None,
                    citation_importance_scores=None, type_ids=None, **generation_kwargs):
            sample = _goai_sample_ref[0]
            batch_size = encoder_states.size(0)
            prompts = []
            for b in range(batch_size):
                if sample is not None:
                    prompts.append(build_goai_prompt(sample))
                else:
                    phrase = phrases[b] if phrases and len(phrases) > b else ''
                    prompts.append(f"[INST] Research direction: {phrase}\nPredict the main theorem. [/INST] Theorem: ")
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1536).to(self.device)
            generated_texts = self._generate_with_cross_attention(
                input_ids=inputs['input_ids'],
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.5, length_penalty=1.0, no_repeat_ngram_size=3,
                citation_importance_scores=citation_importance_scores,
                type_ids=type_ids, **generation_kwargs,
            )
            return None, generated_texts, None

        import types
        mc.MistralDecoder._inference_forward = patched
        # Also bind directly to the instance to override any instance-level method
        model.decoder._inference_forward = types.MethodType(patched, model.decoder)
        model._rich_ctx_sample_ref = _goai_sample_ref  # reuse same hook in generation loop
        logger.info("Applied goai_full variant: full GoAI text prompt + cross-attention encoder states")

    if model_name == 'full_graph':
        ckpt = args.checkpoint or DEFAULT_CKPTS['full_graph']
        gen_model = load_dual_model(ckpt, no_graph=getattr(args, 'no_graph', False), no_enc2=False, no_fusion=getattr(args, 'no_fusion', False), no_enrichment=getattr(args, 'no_enrichment', False), decoder_model=dec_model)
        if args.prompt_variant == 'rich_context':
            apply_rich_context_variant(gen_model)
        elif args.prompt_variant == 'goai_full':
            apply_goai_full_variant(gen_model)
        elif args.prompt_variant == 'target_title':
            apply_target_title_variant(gen_model)
        elif args.prompt_variant:
            apply_prompt_variant(gen_model, args.prompt_variant)
        if args.suppress_layers:
            layers = [int(x) for x in args.suppress_layers.split(',')]
            for layer_idx in layers:
                key = str(layer_idx)
                if key in gen_model.decoder.cross_attn_layers:
                    gen_model.decoder.cross_attn_layers[key].gate_graph.data.fill_(-10.0)
            logger.info(f"Suppressed cross-attn gates for layers: {layers}")
        if getattr(args, 'force_gates', None):
            # Force all gates open to a fixed value (e.g. 0.8 → raw=1.386)
            import math
            target_gate = float(args.force_gates)
            raw_val = math.log(target_gate / (1 - target_gate))  # inverse sigmoid
            for key, cross_attn in gen_model.decoder.cross_attn_layers.items():
                cross_attn.gate_graph.data.fill_(raw_val)
            logger.info(f"Forced all cross-attn gates to {target_gate:.2f} (raw={raw_val:.3f})")
    elif model_name == 'paper_graph_only':
        ckpt = args.checkpoint or DEFAULT_CKPTS['paper_graph_only']
        gen_model = load_dual_model(ckpt, no_graph=False, no_enc2=True, decoder_model=dec_model)
    elif model_name == 'bag_v1':
        if not args.checkpoint:
            logger.error("--checkpoint required for bag_v1")
            return
        gen_model = load_bag_model(args.checkpoint, version='v1')
    elif model_name == 'bag_v2':
        if not args.checkpoint:
            logger.error("--checkpoint required for bag_v2")
            return
        gen_model = load_bag_model(args.checkpoint, version='v2')
    elif model_name == 'text_only':
        ckpt = args.checkpoint
        if not ckpt:
            logger.error("--checkpoint required for text_only model")
            return
        gen_model, gen_tokenizer = load_text_only_model(ckpt)
    elif model_name == 'prompt_only':
        gen_model, gen_tokenizer = load_prompt_only_model(decoder_model=dec_model)
    elif model_name == 'llemma':
        gen_model, gen_tokenizer = load_prompt_only_model(decoder_model=dec_model)
    elif model_name == 'retrieval':
        retrieval_idx = load_retrieval_index()
    elif model_name == 'giants':
        gen_model, gen_tokenizer = load_giants_model()
    elif model_name == 'futuregen':
        gen_model, gen_tokenizer = load_futuregen_model()
        futuregen_pool = load_futuregen_pool()
    elif model_name == 'goai':
        gen_model, gen_tokenizer = load_goai_model()
    elif model_name == 'future_aligned':
        gen_model, gen_tokenizer = load_future_aligned_model(decoder_model=dec_model)
    elif model_name == 'researchagent':
        gen_model, gen_tokenizer = load_researchagent_model()
        futuregen_pool = load_futuregen_pool()
    elif model_name == 'goai_hybrid':
        # Only needs GoAI decoder — theorem context comes from sample data directly
        gen_model, gen_tokenizer = load_goai_model()
    elif model_name == 'coi':
        gen_model, gen_tokenizer = load_coi_model(decoder_model=dec_model)
    elif model_name in ('scimuse', 'scimuse_local'):
        # SciMuse reads pre-generated ideas from JSONL — no model to load
        backend_tag = 'openai' if model_name == 'scimuse' else 'local'
        scimuse_path = os.path.join(BASE, f'scimuse_generated_{backend_tag}.jsonl')
        if not os.path.exists(scimuse_path):
            logger.error(f"SciMuse generated file not found: {scimuse_path}")
            logger.error("Run scimuse_generate.py first.")
            return
        scimuse_texts = {}
        with open(scimuse_path) as f:
            for line in f:
                r = json.loads(line)
                scimuse_texts[r['subgraph_id']] = r.get('generated_text', '')
        logger.info(f"Loaded {len(scimuse_texts)} SciMuse generated texts from {scimuse_path}")
    elif model_name in ('openai_futuregen', 'openai_goai', 'openai_researchagent', 'openai_coi'):
        if 'OPENAI_API_KEY' not in os.environ:
            logger.error("OPENAI_API_KEY not set")
            return
        logger.info(f"Using OpenAI GPT-4o-mini for {model_name}")
        if model_name in ('openai_futuregen', 'openai_researchagent'):
            futuregen_pool = load_futuregen_pool()
    else:
        logger.error(f"Unknown model: {model_name}")
        return

    # Phase 1: generate all texts
    logger.info(f"\nPhase 1: Generating {len(val_samples)} texts with model={model_name}")
    logger.info("=" * 70)

    gen_texts    = []
    gen_times    = []
    meta         = []
    bok_candidates = []   # list of lists; populated only when best_of_k > 1

    for i, sample in enumerate(val_samples):
        sg_id    = sample['paper_subgraph']['subgraph_id']
        arxiv_id = sample.get('target_arxiv_id', '')
        src      = sample.get('target_text_source', '')

        t0 = time.time()
        try:
            if model_name in ('full_graph', 'paper_graph_only'):
                # Set sample ref for rich_context variant (if active)
                if hasattr(gen_model, '_rich_ctx_sample_ref'):
                    gen_model._rich_ctx_sample_ref[0] = sample
                if args.best_of_k > 1:
                    # Generate K candidates; reranking with LoRA theorem embedder happens after gen_model is unloaded
                    if args.use_goai_prompt:
                        phrase = build_goai_hybrid_prompt(sample)[:800]
                    elif args.use_root_title:
                        phrase = _get_root_title(sample)
                    else:
                        phrase = sample.get('phrase', '') or sample.get('target_title', '')
                    phrases = [phrase] if phrase else None
                    candidates = []
                    _, greedy_out, _ = gen_model([sample], target_texts=None, max_new_tokens=args.max_new_tokens, phrases=phrases, do_sample=False)
                    candidates.append(greedy_out[0] if greedy_out else '')
                    for _ in range(args.best_of_k - 1):
                        _, out, _ = gen_model([sample], target_texts=None, max_new_tokens=args.max_new_tokens, phrases=phrases, do_sample=True, temperature=0.8, top_p=0.9)
                        candidates.append(out[0] if out else '')
                    bok_candidates.append(candidates)
                    gen_text = candidates[0]  # placeholder; will be updated after Mistral reranking
                    for ci, cand in enumerate(candidates):
                        label = "greedy" if ci == 0 else f"sample{ci}"
                        logger.info(f"  [Sample {i+1} / {label}]: {cand[:300]}")
                else:
                    gen_text = generate_dual(gen_model, sample, args.max_new_tokens, no_phrase=args.no_phrase, use_title=args.use_title, use_graph_context=args.use_graph_context, use_root_title=args.use_root_title, use_goai_prompt=args.use_goai_prompt)
            elif model_name in ('bag_v1', 'bag_v2'):
                gen_text = generate_bag(gen_model, sample, args.max_new_tokens)
            elif model_name in ('text_only', 'prompt_only'):
                gen_text = generate_text_only(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'llemma':
                gen_text = generate_llemma(gen_model, gen_tokenizer, sample, args.max_new_tokens, use_root_title=args.use_root_title)
            elif model_name == 'retrieval':
                gen_text = generate_retrieval(retrieval_idx, sample)
            elif model_name == 'giants':
                gen_text = generate_giants(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'futuregen':
                gen_text = generate_futuregen(gen_model, gen_tokenizer, sample, pool=futuregen_pool, max_new_tokens=args.max_new_tokens)
            elif model_name == 'goai':
                gen_text = generate_goai(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'goai_hybrid':
                gen_text = generate_goai_hybrid(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'future_aligned':
                gen_text = generate_future_aligned(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'openai_futuregen':
                gen_text = generate_openai(sample, mode='futuregen', max_new_tokens=args.max_new_tokens, pool=futuregen_pool)
            elif model_name == 'openai_goai':
                gen_text = generate_openai(sample, mode='goai', max_new_tokens=args.max_new_tokens)
            elif model_name == 'researchagent':
                gen_text = generate_researchagent(gen_model, gen_tokenizer, sample, pool=futuregen_pool, max_new_tokens=args.max_new_tokens)
            elif model_name == 'openai_researchagent':
                gen_text = generate_researchagent_openai(sample, pool=futuregen_pool, max_new_tokens=args.max_new_tokens)
            elif model_name == 'coi':
                gen_text = generate_coi(gen_model, gen_tokenizer, sample, args.max_new_tokens)
            elif model_name == 'openai_coi':
                gen_text = generate_coi_openai(sample, max_new_tokens=args.max_new_tokens)
            elif model_name in ('scimuse', 'scimuse_local'):
                gen_text = scimuse_texts.get(sg_id, '')
        except Exception as e:
            logger.error(f"  Sample {i+1}: generation error: {e}")
            gen_text = ''
            if args.best_of_k > 1 and model_name in ('full_graph', 'paper_graph_only'):
                bok_candidates.append([''])  # keep index alignment
        gen_time = time.time() - t0

        gen_texts.append(gen_text)
        gen_times.append(gen_time)
        meta.append({'sg_id': sg_id, 'arxiv_id': arxiv_id, 'src': src,
                     'gt_text': sample.get('target_text', '')})

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"  Generated {i+1}/{len(val_samples)}  ({gen_time:.1f}s/sample)")
        if i < 5:
            logger.info(f"  [sample {i+1} preview] {repr(gen_text[:200])}")

    # Unload inference model before loading e5-mistral to free GPU memory
    if gen_model is not None:
        del gen_model
        torch.cuda.empty_cache()
        logger.info("Inference model unloaded from GPU")

    # Best-of-K reranking with finetuned theorem LoRA embedder (sequential — after dual model freed)
    if args.best_of_k > 1 and bok_candidates:
        logger.info(f"\nBest-of-K reranking: embedding {sum(len(c) for c in bok_candidates)} candidates with theorem LoRA embedder...")
        # Flatten all candidates
        flat_cands = [c for cands in bok_candidates for c in cands]
        flat_embs = embed_texts_thm_lora(flat_cands, batch_size=8,
                                         instruction="Retrieve mathematical research papers related to this theorem")  # [total_K, 4096]
        # Load finetuned theorem pool (per-theorem embeddings aggregated to paper level via max)
        logger.info(f"  Loading finetuned theorem pool from {FUTURE_THM_LORA_POOL}")
        thm_pool_data = np.load(FUTURE_THM_LORA_POOL)
        thm_pool_embs = thm_pool_data['embeddings'].astype(np.float32)   # [T, 4096]
        thm_pool_pids = list(thm_pool_data['paper_ids'])
        thm_pool_offs = [int(x) for x in thm_pool_data['offsets']]
        # Build paper-level max-pool embeddings (max over theorems per paper)
        n_papers = len(thm_pool_pids)
        paper_embs = np.zeros((n_papers, thm_pool_embs.shape[1]), dtype=np.float32)
        for pi in range(n_papers):
            s, e = thm_pool_offs[pi], thm_pool_offs[pi + 1]
            seg = thm_pool_embs[s:e]
            paper_embs[pi] = seg.max(axis=0) if len(seg) > 0 else 0.0
        paper_embs_norm = paper_embs / np.maximum(np.linalg.norm(paper_embs, axis=1, keepdims=True), 1e-9)
        logger.info(f"  Theorem pool: {n_papers} papers from {thm_pool_embs.shape[0]} segments")
        # Rerank each sample
        offset = 0
        for i, cands in enumerate(bok_candidates):
            k = len(cands)
            cand_embs = flat_embs[offset:offset + k]  # [k, 4096]
            cand_embs = cand_embs / np.maximum(np.linalg.norm(cand_embs, axis=1, keepdims=True), 1e-9)
            scores = cand_embs @ paper_embs_norm.T  # [k, n_papers]
            best_idx = int(np.argmax(scores.max(axis=1)))
            gen_texts[i] = cands[best_idx]
            logger.info(f"  Sample {i+1}: best_idx={best_idx} (max_pool_sim={scores[best_idx].max():.4f})")
            offset += k
        logger.info("Best-of-K reranking complete; LoRA embedder kept in cache for Phase 2")

    # Phase 2: batch embed all generated texts at once
    logger.info(f"\nPhase 2: Embedding {len(gen_texts)} generated texts...")
    valid_mask = [bool(t) for t in gen_texts]
    # Optionally truncate to N tokens before embedding (for fair length-normalized comparison)
    if args.truncate_tokens:
        texts_to_embed = [' '.join(t.split()[:args.truncate_tokens]) if t else ' ' for t in gen_texts]
        logger.info(f"  Truncating to {args.truncate_tokens} tokens before embedding")
    else:
        texts_to_embed = [t if t else ' ' for t in gen_texts]

    if getattr(args, 'no_embed', False):
        logger.info("  --no_embed set: skipping embedding, using zero vectors")
        all_gen_embs    = np.zeros((len(gen_texts), 4096), dtype=np.float32)
        all_gen_embs_e5 = np.zeros((len(gen_texts), 1024), dtype=np.float32)
    else:
        # Embed with E5-Mistral-7B (primary, used for retrieval metrics)
        logger.info("  Embedding with E5-Mistral-7B...")
        all_gen_embs = embed_texts_mistral(texts_to_embed, batch_size=16,
                                           instruction="Retrieve mathematical research papers related to this theorem")  # [N, 4096]
        # Zero out embeddings for failed generations
        for i, valid in enumerate(valid_mask):
            if not valid:
                all_gen_embs[i] = 0.0

        # Embed with E5-large-v2 (secondary)
        logger.info("  Embedding with E5-large-v2...")
        # Unload Mistral embedder to free GPU for E5-large
        if 'model' in _mistral_emb_cache:
            del _mistral_emb_cache['model']
            torch.cuda.empty_cache()
        all_gen_embs_e5 = embed_texts(texts_to_embed, batch_size=128, prefix='passage')  # [N, 1024]
        for i, valid in enumerate(valid_mask):
            if not valid:
                all_gen_embs_e5[i] = 0.0
        # Unload E5-large
        if 'model' in _e5_cache:
            del _e5_cache['model']
            torch.cuda.empty_cache()

    # Phase 3: compute all metrics
    logger.info("\nPhase 3: Computing metrics...")
    results = []
    all_metrics = defaultdict(list)

    for i, sample in enumerate(val_samples):
        sg_id    = meta[i]['sg_id']
        arxiv_id = meta[i]['arxiv_id']
        src      = meta[i]['src']
        gen_text = gen_texts[i]
        gen_emb  = all_gen_embs[i]

        future_aids     = sg_to_arxids.get(sg_id, [])
        paper_embs_list = [paper_id_to_embs[aid] for aid in future_aids if aid in paper_id_to_embs]

        fp_metrics  = future_paper_metrics(gen_emb, paper_embs_list)
        rr_metrics  = retrieval_rank_metrics_multi(gen_emb, future_aids, paper_id_to_embs)
        gq_metrics  = generation_quality_metrics(gen_text)
        cit_metrics = {'citation_mean_raw': 0.0, 'citation_percentile': 0.0}
        gt_metrics  = gt_generation_metrics(gen_text, meta[i]['gt_text'])

        # Input novelty: ROUGE-L vs input papers to detect copying
        input_texts = []
        for node in val_samples[i].get('paper_subgraph', {}).get('nodes', []):
            t = node.get('summery_text', '') or node.get('abstract', '') or ''
            if t:
                input_texts.append(t[:500])
        novelty_metrics = input_novelty_metrics(gen_text, input_texts, gt_metrics['rouge_l'])

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(
                f"[{i+1}/{len(val_samples)}] arxiv={arxiv_id} src={src} "
                f"n_future={fp_metrics['n_future']} "
                f"max_sim={fp_metrics['max_sim']:.3f} mean_sim={fp_metrics['mean_sim']:.3f} "
                f"hits@10={rr_metrics['hits_at_10']:.0f} hits@100={rr_metrics['hits_at_100']:.0f} "
                f"mrr={rr_metrics['mrr']:.3f} best_rank={rr_metrics['best_rank']} "
                f"recall@100={rr_metrics['recall_at_100']:.3f} gap={rr_metrics['score_gap']:.4f} "
                f"math_ratio={gq_metrics['math_token_ratio']:.2f} struct={gq_metrics['has_structure']:.0f} "
                f"len={gq_metrics['gen_length']}"
            )

        result = {
            'idx':              i,
            'arxiv_id':         arxiv_id,
            'subgraph_id':      sg_id,
            'source':           src,
            'model':            model_name,
            'gen_text':         gen_text,
            'gt_text':          meta[i]['gt_text'],
            'n_future':         fp_metrics['n_future'],
            'max_sim':          fp_metrics['max_sim'],
            'mean_sim':         fp_metrics['mean_sim'],
            'std_sim':          fp_metrics['std_sim'],
            'hits_at_10':       rr_metrics['hits_at_10'],
            'hits_at_100':      rr_metrics['hits_at_100'],
            'mrr':              rr_metrics['mrr'],
            'median_rank':      rr_metrics['median_rank'],
            'best_rank':        rr_metrics['best_rank'],
            'rank_percentile':  rr_metrics['rank_percentile'],
            'recall_at_10':     rr_metrics['recall_at_10'],
            'recall_at_100':    rr_metrics['recall_at_100'],
            'average_precision': rr_metrics['average_precision'],
            'ndcg_at_100':      rr_metrics['ndcg_at_100'],
            'best_pos_score':   rr_metrics['best_pos_score'],
            'best_neg_score':   rr_metrics['best_neg_score'],
            'score_gap':        rr_metrics['score_gap'],
            'pos_mean_score':   rr_metrics['pos_mean_score'],
            'neg_mean_score':   rr_metrics['neg_mean_score'],
            'score_std':        rr_metrics['score_std'],
            'math_token_ratio':    gq_metrics['math_token_ratio'],
            'has_structure':       gq_metrics['has_structure'],
            'gen_length':          gq_metrics['gen_length'],
            'citation_mean_raw':   cit_metrics['citation_mean_raw'],
            'citation_percentile': cit_metrics['citation_percentile'],
            'rouge_l':             gt_metrics['rouge_l'],
            'bertscore_f1':        gt_metrics['bertscore_f1'],
            'lean_syntax_score':   gt_metrics['lean_syntax_score'],
            'input_rouge_l':       novelty_metrics['input_rouge_l'],
            'future_input_ratio':  novelty_metrics['future_input_ratio'],
            'gen_time_s':          round(gen_times[i], 2),
        }
        results.append(result)

        for k in ('max_sim', 'mean_sim', 'std_sim',
                  'hits_at_10', 'hits_at_100', 'mrr', 'median_rank',
                  'best_rank', 'rank_percentile', 'recall_at_10', 'recall_at_100',
                  'average_precision', 'ndcg_at_100', 'best_pos_score',
                  'best_neg_score', 'score_gap', 'pos_mean_score', 'neg_mean_score',
                  'score_std',
                  'math_token_ratio', 'has_structure', 'gen_length',
                  'citation_mean_raw', 'citation_percentile',
                  'rouge_l', 'bertscore_f1', 'lean_syntax_score',
                  'input_rouge_l', 'future_input_ratio'):
            all_metrics[k].append(result[k])

    # Corpus-level distinct metrics — use full untruncated gen_texts
    dist = distinct_metrics(gen_texts)

    # Print summary
    print("\n" + "=" * 70)
    print(f"RESULTS: model={model_name}  samples={len(results)}  pool={pool_embs.shape[0]} papers")
    print("=" * 70)
    print("  --- Future paper similarity ---")
    for k in ('max_sim', 'mean_sim', 'std_sim'):
        vals = all_metrics[k]
        print(f"  {k:20s}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")
    print("  --- Retrieval rank (14K future paper pool) ---")
    for k in ('hits_at_10', 'hits_at_100', 'mrr', 'median_rank',
              'best_rank', 'recall_at_100', 'average_precision', 'ndcg_at_100',
              'score_gap', 'best_pos_score', 'best_neg_score'):
        vals = all_metrics[k]
        print(f"  {k:20s}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")
    print("  --- Generation quality ---")
    for k in ('math_token_ratio', 'has_structure', 'gen_length',
              'rouge_l', 'bertscore_f1', 'lean_syntax_score'):
        vals = all_metrics[k]
        print(f"  {k:20s}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")
    print(f"  {'distinct_1':20s}  {dist['distinct_1']:.4f}  (corpus-level)")
    print(f"  {'distinct_2':20s}  {dist['distinct_2']:.4f}  (corpus-level)")
    print("  --- Input novelty (higher ratio = more forward-looking) ---")
    for k in ('input_rouge_l', 'future_input_ratio'):
        vals = all_metrics[k]
        print(f"  {k:20s}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")
    print("  --- Citation relevance (47K 2024-2025 papers) ---")
    for k in ('citation_mean_raw', 'citation_percentile'):
        vals = [v for v in all_metrics[k] if v >= 0]
        if vals:
            print(f"  {k:20s}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_path = os.path.join(out_dir, f'eval_{model_name}_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model':      model_name,
            'n_samples':  len(results),
            'pool_size':  int(pool_embs.shape[0]),
            'aggregate':  {
                **{k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                        'median': float(np.median(v))}
                   for k, v in all_metrics.items()},
                'distinct_1': dist['distinct_1'],
                'distinct_2': dist['distinct_2'],
            },
            'results':    results,
        }, f, indent=2)
    logger.info(f"\nSaved results to {out_path}")

    # Save gen embeddings for offline re-evaluation (theorem-only retrieval etc.)
    emb_path = out_path.replace('.json', '_gen_embs.npy')
    np.save(emb_path, all_gen_embs)
    logger.info(f"Saved gen embeddings — Mistral ({all_gen_embs.shape}) to {emb_path}")

    emb_path_e5 = out_path.replace('.json', '_gen_embs_e5.npy')
    np.save(emb_path_e5, all_gen_embs_e5)
    logger.info(f"Saved gen embeddings — E5-large ({all_gen_embs_e5.shape}) to {emb_path_e5}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--embed_targets', action='store_true',
                   help='Pre-embed all future paper targets (run once)')
    p.add_argument('--model', choices=['full_graph', 'paper_graph_only', 'text_only', 'prompt_only', 'llemma', 'retrieval', 'bag_v1', 'bag_v2', 'giants', 'futuregen', 'goai', 'goai_hybrid', 'future_aligned', 'openai_futuregen', 'openai_goai', 'researchagent', 'openai_researchagent', 'scimuse', 'scimuse_local', 'coi', 'openai_coi'],
                   help='Which model to evaluate')
    p.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    p.add_argument('--n', type=int, default=None, help='Number of val samples to eval')
    p.add_argument('--skip', type=int, default=0, help='Skip first N eligible samples')
    p.add_argument('--max_new_tokens', type=int, default=300)
    p.add_argument('--truncate_tokens', type=int, default=None,
                   help='Truncate generated text to N tokens before embedding (for fair length-normalized comparison)')
    p.add_argument('--no_phrase', action='store_true', help='Mask phrase — force model to rely on graph only')
    p.add_argument('--use_title', action='store_true', help='Use target_title as phrase instead of the dataset phrase')
    p.add_argument('--use_root_title', action='store_true', help='Use root node title_text (anchor paper) as phrase')
    p.add_argument('--use_graph_context', action='store_true', help='Build phrase from graph node titles for richer topical context')
    p.add_argument('--use_goai_prompt', action='store_true', help='Use full GoAI-style citation graph prompt as decoder phrase')
    p.add_argument('--prompt_variant', type=str, default=None,
                   choices=['theorem_prefix', 'goai_style', 'abstract_style', 'abstract_we', 'rich_context', 'goai_full', 'target_title',
                            'future_we_prove', 'future_theorem', 'future_in_paper', 'future_building', 'future_extend'],
                   help='Override decoder prompt style for full_graph/paper_graph_only')
    p.add_argument('--decoder_model', type=str, default=None,
                   help='Decoder model name (e.g. mistralai/Mistral-7B-Instruct-v0.3). Default: deepseek-math-7b')
    p.add_argument('--no_embed', action='store_true', help='Skip embedding phase (use zero vectors); embed separately later')
    p.add_argument('--suppress_layers', type=str, default=None,
                   help='Comma-separated layer indices to suppress (set gate→0), e.g. "3,7,11,15"')
    p.add_argument('--no_graph', action='store_true',
                   help='Load with no_graph=True (enc2/theorem encoder only, no paper graph enc1)')
    p.add_argument('--no_fusion', action='store_true',
                   help='Load with no_fusion=True (enc1+enc2 but no cross-attention fusion)')
    p.add_argument('--no_enrichment', action='store_true',
                   help='Load with no_enrichment=True (no theorem enrichment from Mathlib)')
    p.add_argument('--force_gates', type=float, default=None,
                   help='Force all cross-attention gates to this value (0-1), e.g. 0.8')
    p.add_argument('--best_of_k', type=int, default=1,
                   help='Generate K candidates per sample, pick best by pool cosine similarity')
    p.add_argument('--subgraph_ids_file', type=str, default=None,
                   help='JSON file with canonical subgraph_ids list (e.g. canonical_200_subset.json) — restrict eval to these subgraphs')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.embed_targets:
        embed_future_targets()
    elif args.model:
        evaluate(args)
    else:
        print("Specify --embed_targets or --model <name>")
