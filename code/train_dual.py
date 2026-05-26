"""
Dual Encoder Training Script
=============================
Architecture:
  Encoder 1 (paper graph)   → [N, 1152]  → bridge MLP(1152→2048→4224) → [N, 4224]
  Encoder 2 (mathlib graph) → [M, 4224]                                → [M, 4224]
                                    ↓
                     type embeddings (0=paper, 1=theorem) added
                                    ↓
                          FusionCrossAttention (bidirectional, dim=4224)
                                    ↓
                      graph dropout (30%, training only)
                                    ↓
                            [N+M, 4224] → DeepSeek-Math decoder (k_proj: 4224→4096)

Loss:
  L_total = L_CE + 0.1 * L_contrastive + 0.05 * L_cross_contr
  L_contrastive = InfoNCE(fused_proj(fused_pool)[4096], mean(last4_layers)[target_tokens].detach()[4096])
  L_cross_contr = InfoNCE(cross_proj_paper(paper_thm_emb)[128], cross_proj_mathlib(enc2_emb)[128])
  Target = decoder last-4-layer hidden states, mean-pooled over TARGET tokens only (prompt excluded), detached

enc1 graph enriched with paper's own theorem statements (from paper_theorems_filtered.jsonl)
  joined via target_arxiv_id, encoded with E5-large-v2, added as extra nodes connected to target paper

Checkpoints:
  Encoder 1: mistral_cross_31/encoder_only.pt  (or best_model.pt fallback)
  Encoder 2: theorem_checkpoints_94/theorem_best_model.pt
  Decoder:   deepseek-ai/deepseek-math-7b-instruct (fresh)

Training:
  - enc1.gnn:                      unfrozen, lr=2e-6  (2.3M params)
  - enc2.gnn.W_struct + layer_norms: unfrozen, lr=2e-6  (~0.5M params)
  - bridge + fusion + type_embed:  trainable, lr=2e-5
  - decoder LoRA + cross-attn:     trainable, lr=2e-5
  - contrastive_proj:              trainable, lr=2e-5
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
# BASE: root of downloaded data files (see data/MANIFEST.md).
# Override with COMPOSE_DATA_DIR env var or --data / --enc1-ckpt / --enc2-ckpt CLI args.
_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE          = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
_CKPT_ROOT    = os.environ.get('COMPOSE_CKPT_DIR', os.path.join(_REPO_ROOT, 'checkpoints'))
LEANDOJO_BASE = os.path.join(BASE, 'LeanDojo/leandojo_benchmark_4')

ENC1_CKPT     = os.environ.get('ENC1_CKPT', os.path.join(_CKPT_ROOT, 'encoder_only.pt'))
ENC2_CKPT     = os.environ.get('ENC2_CKPT', os.path.join(_CKPT_ROOT, 'theorem_best_model.pt'))
ENC2_EMB_DIR  = os.path.join(LEANDOJO_BASE, 'embeddings_deepseek')
DUAL_DATA     = os.path.join(BASE, 'dual_training_samples_v7_clean.jsonl')
CKPT_DIR      = os.path.join(_CKPT_ROOT, 'training_output')

ENC1_DIM    = 1152   # E5 1024 + GNN 128
ENC2_DIM    = 4224   # DeepSeek 4096 + GNN 128
FUSED_DIM   = 4224   # working dimension after bridge
DECODER_DIM = 4096   # DeepSeek hidden size

DEFAULT_DECODER = "deepseek-ai/deepseek-math-7b-instruct"

DECODER_CONFIGS = {
    "deepseek-ai/deepseek-math-7b-instruct": {"decoder_dim": 4096, "cross_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31]},
    "Qwen/Qwen2-0.5B":                       {"decoder_dim": 896,  "cross_attn_layers": [3, 7, 11, 15, 19, 23]},
    "meta-llama/Llama-2-7b-hf":              {"decoder_dim": 4096, "cross_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31]},
    "mistralai/Mistral-7B-Instruct-v0.3":    {"decoder_dim": 4096, "cross_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31]},
    "meta-llama/Llama-3.2-3B-Instruct":      {"decoder_dim": 3072, "cross_attn_layers": [3, 6, 9, 12, 15, 18, 21, 24, 27]},
}

GRAPH_DROPOUT    = 0.20   # fraction of context vectors zeroed during training
CONTRASTIVE_TAU  = 0.05   # InfoNCE temperature (lower = sharper contrast, better retrieval)
CONTRASTIVE_W    = 0.50   # weight of contrastive loss (raised from 0.30 for stronger retrieval signal)
CROSS_CONTR_W    = 0.20   # weight of cross-encoder contrastive loss
MEMORY_BANK_SIZE = 512    # FIFO queue of past target_hs for extra negatives (raised from 256)

MAX_PAPER_THEOREMS    = 10
PAPER_THM_EMBS_PATH   = os.path.join(BASE, 'paper_theorem_embs.npy')
TRAIN_TARGET_THMFT_EMBS = os.path.join(BASE, 'train_target_thmft_embs.npz')
PAPER_THM_STMT_IDX    = os.path.join(BASE, 'paper_theorem_stmt_to_idx.json')
PAPER_THM_BY_PAPER    = os.path.join(BASE, 'paper_theorem_by_paper.json')
PAPER_THM_TO_MATHLIB  = os.path.join(BASE, 'paper_thm_to_mathlib.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── add paths so we can import both models ─────────────────────────────────────
sys.path.insert(0, BASE)
sys.path.insert(0, LEANDOJO_BASE)


# ==============================================================================
# DATASET
# ==============================================================================

GOOD_SOURCES = {
    'abstract_full', 'theorem_1x', 'theorem_letter',
    'main_theorem_label', 'main_result_of',
}

class DualDataset(Dataset):
    def __init__(self, path: str, mode: str = 'train', max_samples: int = None,
                 filter_sources: bool = False):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        if filter_sources:
            before = len(self.samples)
            self.samples = [s for s in self.samples
                            if s.get('target_text_source', '') in GOOD_SOURCES]
            logger.info(f"DualDataset: filtered {before} → {len(self.samples)} samples (good sources only)")
        split = int(0.9 * len(self.samples))
        if mode == 'train':
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        elif max_samples == 0:
            self.samples = []
        logger.info(f"DualDataset [{mode}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==============================================================================
# BRIDGE: lifts Encoder 1 output from 1152 → 4224  (2-layer MLP)
# ==============================================================================

class BridgeProjection(nn.Module):
    """Projects Encoder 1 [N, 1152] into Encoder 2 space [N, 4224]."""
    def __init__(self, in_dim: int = ENC1_DIM, out_dim: int = FUSED_DIM):
        super().__init__()
        mid_dim = 2048
        self.proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ==============================================================================
# FUSION: cross-attention between paper nodes and theorem nodes
# ==============================================================================

class FusionCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between paper nodes and theorem nodes.
    Both are in the same [*, 4224] space (after bridge + type embed).

    Paper nodes   attend to theorem nodes → paper_updated  [N, 4224]
    Theorem nodes attend to paper nodes   → theorem_updated [M, 4224]

    Output: concat [N+M, 4224]
    """
    def __init__(self, dim: int = FUSED_DIM, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Papers attend to theorems
        self.q_paper   = nn.Linear(dim, dim, bias=False)
        self.k_theorem = nn.Linear(dim, dim, bias=False)
        self.v_theorem = nn.Linear(dim, dim, bias=False)
        self.o_paper   = nn.Linear(dim, dim, bias=False)

        # Theorems attend to papers
        self.q_theorem = nn.Linear(dim, dim, bias=False)
        self.k_paper   = nn.Linear(dim, dim, bias=False)
        self.v_paper   = nn.Linear(dim, dim, bias=False)
        self.o_theorem = nn.Linear(dim, dim, bias=False)

        # Zero-init output gates so fusion starts neutral
        nn.init.zeros_(self.o_paper.weight)
        nn.init.zeros_(self.o_theorem.weight)

        self.norm_paper   = nn.LayerNorm(dim)
        self.norm_theorem = nn.LayerNorm(dim)

    def _attn(self, Q, K, V, key_mask=None):
        B, Nq, _ = Q.shape
        Nk = K.shape[1]
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            mask = (~key_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.dim)
        return out

    def forward(
        self,
        paper_vecs: torch.Tensor,      # [B, N, 4224]
        theorem_vecs: torch.Tensor,    # [B, M, 4224]
        paper_mask: torch.Tensor,      # [B, N] bool
        theorem_mask: torch.Tensor,    # [B, M] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Papers attend to theorems
        Q_p = self.q_paper(paper_vecs)
        K_t = self.k_theorem(theorem_vecs)
        V_t = self.v_theorem(theorem_vecs)
        paper_ctx = self._attn(Q_p, K_t, V_t, key_mask=theorem_mask)
        paper_updated = self.norm_paper(paper_vecs + self.o_paper(paper_ctx))

        # Theorems attend to papers
        Q_t = self.q_theorem(theorem_vecs)
        K_p = self.k_paper(paper_vecs)
        V_p = self.v_paper(paper_vecs)
        theorem_ctx = self._attn(Q_t, K_p, V_p, key_mask=paper_mask)
        theorem_updated = self.norm_theorem(theorem_vecs + self.o_theorem(theorem_ctx))

        return paper_updated, theorem_updated


# ==============================================================================
# DUAL ENCODER MODEL
# ==============================================================================

class DualEncoderModel(nn.Module):
    def __init__(self, device: str = 'cuda', no_graph: bool = False, no_enc2: bool = False, no_fusion: bool = False, no_enrichment: bool = False, decoder_model: str = DEFAULT_DECODER):
        super().__init__()
        self.no_graph = no_graph
        self.no_enc2  = no_enc2
        self.no_fusion = no_fusion
        self.no_enrichment = no_enrichment
        # Encoders on cuda:0. For large decoders (7B), spread across all GPUs via device_map='auto'.
        # For small decoders (≤3B), keep everything on cuda:0 to avoid cross-device tensor issues.
        self.encoder_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        _dec_cfg_tmp = DECODER_CONFIGS.get(decoder_model, DECODER_CONFIGS[DEFAULT_DECODER])
        _is_small_decoder = _dec_cfg_tmp['decoder_dim'] <= 3072
        # Use device_map='auto' when multiple GPUs available, even for small decoders
        self.decoder_device = 'auto' if torch.cuda.device_count() > 1 else self.encoder_device
        self.device = self.encoder_device

        # ── Encoder 1 (paper graph) ────────────────────────────────────────────
        logger.info("Loading Encoder 1 (paper graph)...")
        from model_clean import FuturePredictionModel, MistralDecoder
        self.enc1 = FuturePredictionModel(
            graph_input_dim=1024,
            text_hidden_dim=1024,
            hidden_dim=1152,
            num_graph_layers=3,
            use_precomputed_embeddings=True,
            device=self.encoder_device,
        )
        self.enc1.to(self.encoder_device)

        # Freeze everything in enc1, then selectively unfreeze gnn
        self._freeze(self.enc1)
        for p in self.enc1.gnn.parameters():
            p.requires_grad = True
        enc1_gnn_params = sum(p.numel() for p in self.enc1.gnn.parameters())
        logger.info(f"  Encoder 1 loaded [gnn unfrozen: {enc1_gnn_params:,} params]")

        # ── Encoder 2 (mathlib graph) ──────────────────────────────────────────
        if not no_enc2:
            logger.info("Loading Encoder 2 (mathlib/theorem graph)...")
            from model_clean_theorem import TheoremEncoder
            self.enc2 = TheoremEncoder(
                embed_dim=4096,
                hidden_dim=4096,
                struct_dim=128,
                num_gnn_layers=4,
                use_dependency_head=False,
                device=self.encoder_device,
            )
            emb_path = os.path.join(ENC2_EMB_DIR, 'embeddings.npy')
            idx_path = os.path.join(ENC2_EMB_DIR, 'name_to_idx.json')
            self.enc2_embeddings = torch.tensor(
                np.load(emb_path, mmap_mode='r'), dtype=torch.float32
            )
            with open(idx_path) as f:
                self.enc2_name_to_idx = json.load(f)
            logger.info(f"  DeepSeek embeddings loaded: {self.enc2_embeddings.shape}")

            self.enc2.to(self.encoder_device)

            # Freeze everything in enc2, then selectively unfreeze W_struct + layer_norms
            self._freeze(self.enc2)
            for p in self.enc2.gnn.W_struct.parameters():
                p.requires_grad = True
            for p in self.enc2.gnn.layer_norms.parameters():
                p.requires_grad = True
            enc2_unfrozen = (
                sum(p.numel() for p in self.enc2.gnn.W_struct.parameters()) +
                sum(p.numel() for p in self.enc2.gnn.layer_norms.parameters())
            )
            logger.info(f"  Encoder 2 loaded [W_struct+layer_norms unfrozen: {enc2_unfrozen:,} params]")
        else:
            logger.info("  Encoder 2 skipped (no_enc2=True)")

        # ── Bridge (MLP: 1152 → 2048 → 4224) ──────────────────────────────────
        self.bridge = BridgeProjection(ENC1_DIM, FUSED_DIM).to(device)
        logger.info(f"  Bridge MLP: {ENC1_DIM}→2048→{FUSED_DIM} [trainable]")

        # ── Type embeddings (paper=0, theorem=1) ───────────────────────────────
        self.type_embed = nn.Embedding(2, FUSED_DIM).to(device)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        logger.info(f"  Type embeddings: 2×{FUSED_DIM} [trainable]")

        # ── Fusion cross-attention ─────────────────────────────────────────────
        self.fusion = FusionCrossAttention(dim=FUSED_DIM, num_heads=8).to(device)
        logger.info(f"  Fusion cross-attention: dim={FUSED_DIM} [trainable]")

        # ── Resolve decoder dim early (needed for fused_proj and mem_bank) ────────
        dec_cfg_early = DECODER_CONFIGS.get(decoder_model, DECODER_CONFIGS[DEFAULT_DECODER])
        _decoder_dim  = dec_cfg_early['decoder_dim']

        # ── Contrastive projection head (fused_pool [4224] → decoder space) ─────
        # Projects graph representation into decoder's semantic space for InfoNCE
        self.fused_proj = nn.Linear(FUSED_DIM, _decoder_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.fused_proj.weight)
        logger.info(f"  Contrastive fused_proj: {FUSED_DIM}→{_decoder_dim} [trainable]")

        # ── Cross-encoder contrastive projections (paper_thm E5[1024] ↔ mathlib DeepSeek[4096] → shared[128]) ─
        self.cross_proj_paper   = nn.Linear(1024, 128, bias=False).to(device)
        self.cross_proj_mathlib = nn.Linear(4096, 128, bias=False).to(device)
        nn.init.xavier_uniform_(self.cross_proj_paper.weight)
        nn.init.xavier_uniform_(self.cross_proj_mathlib.weight)
        logger.info("  Cross-encoder projections: E5[1024]→[128], DeepSeek[4096]→[128] [trainable]")

        # ── E5 alignment projection (fused_pool [4224] → E5 space [1024]) ─
        self.e5_proj = nn.Linear(FUSED_DIM, 1024, bias=False).to(device)
        nn.init.xavier_uniform_(self.e5_proj.weight)
        logger.info(f"  E5 alignment projection: {FUSED_DIM}→1024 [trainable]")

        # ── Generated-text E5 alignment (decoder hidden → E5 space [1024]) ─
        # Forces decoder to produce text whose hidden states embed close to target paper in E5 space
        self.gen_e5_proj = nn.Linear(_decoder_dim, 1024, bias=False).to(device)
        nn.init.xavier_uniform_(self.gen_e5_proj.weight)
        logger.info(f"  Gen-text E5 projection: {_decoder_dim}→1024 [trainable]")

        # ── Generated-text thm_ft alignment (decoder hidden → thm_ft space [4096]) ─
        # Aligns decoder output with finetuned theorem embedder — directly optimizes for thm_ft eval metric
        self.gen_thm_ft_proj = nn.Linear(_decoder_dim, 4096, bias=False).to(device)
        nn.init.xavier_uniform_(self.gen_thm_ft_proj.weight)
        logger.info(f"  Gen-text thm_ft projection: {_decoder_dim}→4096 [trainable]")

        # ── Attention pooling: learns which nodes matter for fused_pool ────────
        # Single linear layer [4224 → 1] → softmax over valid nodes → weighted sum
        self.pool_attn = nn.Linear(FUSED_DIM, 1, bias=True).to(device)
        nn.init.zeros_(self.pool_attn.weight)
        nn.init.zeros_(self.pool_attn.bias)
        logger.info("  Attention pooling: 4224→1 [trainable]")

        # ── Paper theorem node embeddings (precomputed E5, for enc1 graph augmentation) ─
        _thm_embs_ok = os.path.exists(PAPER_THM_EMBS_PATH)
        if _thm_embs_ok:
            self.paper_thm_embs = torch.tensor(
                np.load(PAPER_THM_EMBS_PATH, mmap_mode='r'), dtype=torch.float32
            )  # [N_stmts, 1024] — not a model parameter, stays on CPU
            with open(PAPER_THM_STMT_IDX) as _f:
                self.paper_thm_stmt_to_idx = json.load(_f)
            with open(PAPER_THM_BY_PAPER) as _f:
                self.paper_thm_by_paper = json.load(_f)
            with open(PAPER_THM_TO_MATHLIB) as _f:
                self.paper_thm_to_mathlib = json.load(_f)
            logger.info(
                f"  Paper theorem nodes: {self.paper_thm_embs.shape[0]:,} stmts, "
                f"{len(self.paper_thm_by_paper):,} papers, "
                f"{len(self.paper_thm_to_mathlib):,} with mathlib match"
            )
        else:
            self.paper_thm_embs        = None
            self.paper_thm_stmt_to_idx = {}
            self.paper_thm_by_paper    = {}
            self.paper_thm_to_mathlib  = {}
            logger.warning(f"  Paper theorem embeddings not found at {PAPER_THM_EMBS_PATH} — skipping")

        # ── Precomputed thm_ft embeddings for training target papers ─────────
        if os.path.exists(TRAIN_TARGET_THMFT_EMBS):
            _thmft_data = np.load(TRAIN_TARGET_THMFT_EMBS)
            self.train_thmft_embs = _thmft_data['embeddings'].astype(np.float32)  # [N, 4096]
            self.train_thmft_arxiv_ids = list(_thmft_data['arxiv_ids'])
            self.train_thmft_idx = {a: i for i, a in enumerate(self.train_thmft_arxiv_ids)}
            logger.info(f"  Train target thm_ft embeddings: {self.train_thmft_embs.shape[0]} papers loaded")
        else:
            self.train_thmft_embs = None
            self.train_thmft_idx = {}
            logger.warning(f"  Train target thm_ft embeddings not found at {TRAIN_TARGET_THMFT_EMBS} — thm_ft loss disabled")

        # ── E5 model for on-the-fly embedding of informal_pairs texts ──────────
        from transformers import AutoTokenizer, AutoModel
        _e5_name = "intfloat/e5-large-v2"
        logger.info(f"  Loading E5 tokenizer/model ({_e5_name}) for informal pair embedding...")
        self.enc1_tokenizer = AutoTokenizer.from_pretrained(_e5_name)
        self.enc1_e5 = AutoModel.from_pretrained(_e5_name).to(self.encoder_device)
        self.enc1_e5.eval()
        for p in self.enc1_e5.parameters():
            p.requires_grad = False
        logger.info(f"  E5 model loaded and frozen on {self.encoder_device}")

        # ── Memory bank: FIFO queue of past target_hs for extra negatives ─────
        # Stored as buffers (no grad). Ptr tracks next write position.
        self.register_buffer('mem_bank',
            torch.zeros(MEMORY_BANK_SIZE, _decoder_dim, device=self.encoder_device))
        self.register_buffer('mem_ptr',
            torch.zeros(1, dtype=torch.long, device=self.encoder_device))
        self.register_buffer('mem_full',
            torch.zeros(1, dtype=torch.bool, device=self.encoder_device))
        logger.info(f"  Memory bank: {MEMORY_BANK_SIZE}×{_decoder_dim} FIFO [no grad]")

        # ── Decoder ────────────────────────────────────────────────────────────
        dec_cfg = DECODER_CONFIGS.get(decoder_model, DECODER_CONFIGS[DEFAULT_DECODER])
        logger.info(f"Initializing decoder: {decoder_model}...")
        self.decoder = MistralDecoder(
            encoder_dim=FUSED_DIM,
            model_name=decoder_model,
            cross_attn_layers=dec_cfg['cross_attn_layers'],
            use_lora=True,
            lora_rank=32,
            use_8bit=False,
            device=self.decoder_device,
            freeze_first_n_layers=0,
            lora_checkpoint_path=None,
        )
        logger.info(f"  Decoder initialized on {self.decoder_device} (encoders on {self.encoder_device})")

        # ── Fix cross-attention projections: summary_dim 1152 → 4224 ──────────
        # CrossAttentionLayer in model_clean.py defaults to summary_dim=1152
        # We need 4224 since our fused vectors are 4224-dim.
        for layer_idx_str, cross_attn in self.decoder.cross_attn_layers.items():
            cross_attn_device = next(cross_attn.parameters()).device
            old_k = cross_attn.k_proj_summary
            hidden_size = old_k.weight.shape[0]  # 4096
            cross_attn.k_proj_summary = nn.Linear(FUSED_DIM, hidden_size, bias=False).to(cross_attn_device)
            cross_attn.v_proj_summary = nn.Linear(FUSED_DIM, hidden_size, bias=False).to(cross_attn_device)
            cross_attn.summary_dim = FUSED_DIM
            nn.init.xavier_uniform_(cross_attn.k_proj_summary.weight)
            nn.init.xavier_uniform_(cross_attn.v_proj_summary.weight)
        logger.info(f"  Cross-attn k/v projections replaced: {FUSED_DIM}→{hidden_size} (was 1152→{hidden_size})")

        # ── Log trainable param count ──────────────────────────────────────────
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def _get_theorem_embeddings(self, names: List[str]) -> torch.Tensor:
        vecs = []
        for name in names:
            idx = self.enc2_name_to_idx.get(name)
            if idx is not None:
                vecs.append(self.enc2_embeddings[idx])
            else:
                vecs.append(torch.zeros(4096))
        return torch.stack(vecs).to(self.device)

    def _get_paper_theorem_nodes(self, arxiv_id: str) -> Optional[torch.Tensor]:
        """Look up precomputed E5 embeddings for the paper's own theorems → [K, 1024] or None."""
        if self.paper_thm_embs is None or not arxiv_id:
            return None
        stmt_ids = self.paper_thm_by_paper.get(arxiv_id, [])[:MAX_PAPER_THEOREMS]
        vecs = []
        for sid in stmt_ids:
            idx = self.paper_thm_stmt_to_idx.get(sid)
            if idx is not None:
                vecs.append(self.paper_thm_embs[idx])
        if not vecs:
            return None
        return torch.stack(vecs).to(self.device)  # [K, 1024]

    def encode_paper_subgraph(
        self,
        subgraph: Dict,
        arxiv_id: str = '',
        informal_pairs: Optional[List[Dict]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """Encode one paper subgraph → ([N+K+P, 1152], paired_indices)

        All extra nodes (enrichment K + paired P) are fed INTO the GNN so they
        participate in message passing. Each theorem node is connected to the
        paper node it came from via an extra edge.

        Returns:
          paper_vecs:     [N+K+P, 1152]  (GNN-aware for all nodes)
          paired_indices: list of row indices into paper_vecs for the P paired nodes
        """
        if 'papers' not in subgraph and 'nodes' in subgraph:
            subgraph = {'papers': subgraph['nodes'], 'edges': subgraph.get('edges', [])}

        # ── Build paper_id → arxiv_id lookup for edge construction ────────────
        paper_nodes = subgraph['papers']
        paper_id_to_arxiv = {}
        for node in paper_nodes:
            for ident in node.get('identifiers', []):
                if ident:
                    paper_id_to_arxiv[node['paper_id']] = ident
                    break

        # ── Collect all extra node embeddings [1024] and their paper edges ─────
        extra_embs_list = []   # each entry: [1024] tensor
        extra_edges = []       # (extra_node_idx, paper_node_idx) — paper_node_idx relative to N
        paper_id_to_node_idx = {p['paper_id']: i for i, p in enumerate(paper_nodes)}
        arxiv_to_node_idx = {v: k for k, v in paper_id_to_arxiv.items()}  # arxiv_id → paper_id
        # convert to node index
        arxiv_to_paper_idx = {}
        for pid, arxiv in paper_id_to_arxiv.items():
            if pid in paper_id_to_node_idx:
                arxiv_to_paper_idx[arxiv] = paper_id_to_node_idx[pid]

        # Enrichment nodes: all paper theorems from precomputed embeddings (no loss)
        thm_vecs = None if self.no_enrichment else self._get_paper_theorem_nodes(arxiv_id)
        if thm_vecs is not None:
            target_paper_idx = arxiv_to_paper_idx.get(arxiv_id)
            for i in range(thm_vecs.size(0)):
                extra_idx = len(extra_embs_list)
                extra_embs_list.append(thm_vecs[i])  # [1024]
                if target_paper_idx is not None:
                    extra_edges.append((extra_idx, target_paper_idx))

        # Paired nodes: informal theorems from informal_pairs (used for contrastive loss)
        paired_indices = []
        pair_embs = None
        if informal_pairs:
            pair_embs = self._embed_informal_pairs(informal_pairs)  # [P, 1024] or None
            if pair_embs is not None:
                for i, pair in enumerate(informal_pairs):
                    extra_idx = len(extra_embs_list)
                    extra_embs_list.append(pair_embs[i])  # [1024]
                    src_arxiv = pair.get('paper_id', '')
                    paper_idx = arxiv_to_paper_idx.get(src_arxiv)
                    if paper_idx is not None:
                        extra_edges.append((extra_idx, paper_idx))
                    paired_indices.append(extra_idx)  # relative to extra nodes, offset below

        # ── Call encode_subgraph with extra nodes fed into the GNN ────────────
        extra_tensor = None
        if extra_embs_list:
            extra_tensor = torch.stack(extra_embs_list, dim=0).to(self.encoder_device)  # [K+P, 1024]

        enc_dict = self.enc1.encode_subgraph(subgraph, extra_node_embs=extra_tensor, extra_edges=extra_edges)
        if enc_dict is None:
            return None, []

        paper_vecs = enc_dict['cross_attn_embeds']  # [2N + K + P, 1152] (summaries+titles+theorems)
        N = enc_dict['num_paper_nodes']             # original paper count

        # paired_indices are into the extra nodes section of cross_attn_embeds.
        # Layout: [N summaries | N titles | K enrichment | P paired]
        # theorem nodes start at 2*N
        paired_indices_global = [2 * N + idx for idx in paired_indices]

        return paper_vecs, paired_indices_global

    @torch.no_grad()
    def _embed_informal_pairs(self, informal_pairs: List[Dict]) -> Optional[torch.Tensor]:
        """Embed informal theorem texts with E5 → [P, 1024], L2-normalized."""
        if not informal_pairs:
            return None
        texts = [f"passage: {p['text']}" for p in informal_pairs if p.get('text')]
        if not texts:
            return None
        enc = self.enc1_tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        )
        enc = {k: v.to(self.encoder_device) for k, v in enc.items()}
        out = self.enc1_e5(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(embs.float(), dim=-1)  # [P, 1024]

    def encode_mathlib_subgraph(self, mathlib_subgraph: Dict) -> torch.Tensor:
        """Encode one mathlib subgraph → [M, 4224]"""
        theorems = mathlib_subgraph.get('subgraph_theorems', [])
        edges    = mathlib_subgraph.get('subgraph_edges', [])
        if not theorems:
            return torch.zeros(1, ENC2_DIM, device=self.device)
        names = [t['name'] for t in theorems]
        node_embs = self._get_theorem_embeddings(names)
        enc_dict = self.enc2.encode_subgraph(node_embs, edges)
        return enc_dict['cross_attn_embeds']  # [M, 4224]

    @torch.no_grad()


    def _cross_encoder_contrastive_loss(
        self,
        batch: List[Dict],
        informal_embs_list: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """InfoNCE: paper theorem E5 embeddings ↔ matched Mathlib enc2 DeepSeek embeddings.

        Uses precomputed paper_thm_embs (E5) and paper_thm_to_mathlib to build pairs:
          arxiv_id → paper_thm_by_paper → stmt_keys → mathlib_names → enc2 embeddings
        Project both to shared 128-dim space. InfoNCE across all pairs in the batch.
        """
        if self.paper_thm_embs is None:
            return None

        query_vecs = []  # E5 paper theorem embeddings [1024]
        key_vecs   = []  # enc2 DeepSeek embeddings [4096]

        for sample in batch:
            arxiv_id = sample.get('target_arxiv_id', '')
            stmt_keys = self.paper_thm_by_paper.get(arxiv_id, [])
            for stmt_key in stmt_keys[:MAX_PAPER_THEOREMS]:
                mathlib_name = self.paper_thm_to_mathlib.get(stmt_key)
                if mathlib_name is None:
                    continue
                enc2_idx = self.enc2_name_to_idx.get(mathlib_name)
                if enc2_idx is None:
                    continue
                thm_idx = self.paper_thm_stmt_to_idx.get(stmt_key)
                if thm_idx is None:
                    continue
                query_vecs.append(self.paper_thm_embs[thm_idx])     # [1024]
                key_vecs.append(self.enc2_embeddings[enc2_idx])      # [4096]

        if len(query_vecs) < 2:
            return None

        q = torch.stack(query_vecs).to(self.device)   # [P, 1024]
        k = torch.stack(key_vecs).to(self.device)     # [P, 4096]

        z_q = F.normalize(self.cross_proj_paper(q),   dim=-1)  # [P, 128]
        z_k = F.normalize(self.cross_proj_mathlib(k), dim=-1)  # [P, 128]

        logits = torch.matmul(z_q, z_k.T) / CONTRASTIVE_TAU   # [P, P]
        labels = torch.arange(len(query_vecs), device=self.device)
        return F.cross_entropy(logits, labels)

    def _link_prediction_loss(
        self,
        batch: List[Dict],
        paper_vecs_list: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """BCE loss on edge existence between node pairs in each subgraph.

        For each subgraph:
          - positive pairs: edges that exist in paper_subgraph['edges']
          - negative pairs: random non-edge node pairs (same count as positives)

        Uses node embeddings BEFORE bridge (enc1 output [N, 1152]) mapped to
        paper_id so we can match edge source/target to node indices.
        """
        total_loss = None
        loss_count = 0

        # Simple dot-product scorer on 4224-dim fused vecs (after bridge)
        for sample, pv in zip(batch, paper_vecs_list):
            nodes  = sample['paper_subgraph']['nodes']
            edges  = sample['paper_subgraph']['edges']
            N = len(nodes)
            if N < 2 or not edges:
                continue

            # Build paper_id → node index map
            pid_to_idx = {n['paper_id']: i for i, n in enumerate(nodes)}

            # Collect positive pairs
            pos_pairs = []
            for e in edges:
                src = pid_to_idx.get(e.get('source', ''))
                tgt = pid_to_idx.get(e.get('target', ''))
                if src is not None and tgt is not None and src != tgt:
                    pos_pairs.append((src, tgt))

            if not pos_pairs:
                continue

            # Sample same number of negative pairs
            import random
            neg_pairs = []
            attempts = 0
            pos_set = set(pos_pairs) | {(b, a) for a, b in pos_pairs}
            while len(neg_pairs) < len(pos_pairs) and attempts < len(pos_pairs) * 10:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a != b and (a, b) not in pos_set:
                    neg_pairs.append((a, b))
                attempts += 1

            if not neg_pairs:
                continue

            all_pairs = pos_pairs + neg_pairs
            labels = torch.tensor(
                [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs),
                device=pv.device
            )

            # pv is [num_vecs, 4224] — use only first N (paper node summaries)
            # Layout: [N summaries | N titles | K extra], take summaries
            node_embs = pv[:N]  # [N, 4224]
            if node_embs.size(0) < N:
                continue

            scores = torch.stack([
                torch.dot(
                    F.normalize(node_embs[a], dim=0),
                    F.normalize(node_embs[b], dim=0)
                )
                for a, b in all_pairs
            ])

            pair_loss = F.binary_cross_entropy_with_logits(scores * 5.0, labels)
            total_loss = pair_loss if total_loss is None else total_loss + pair_loss
            loss_count += 1

        if total_loss is None or loss_count == 0:
            return None
        return total_loss / loss_count

    def _e5_alignment_loss(
        self,
        batch: List[Dict],
        fused_pool: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """InfoNCE: fused_pool[i] should be close to E5 embedding of target paper i.

        Uses positive_papers[0].summary_e5 path (already computed, stored as .npy).
        Negatives = all other samples in the batch.
        """
        query_vecs = []   # fused_pool rows
        key_vecs   = []   # E5 target embeddings

        for i, sample in enumerate(batch):
            # Try to load target paper E5 from positive_papers
            pos_papers = sample.get('positive_papers', [])
            e5_path = None
            for pp in pos_papers:
                path = pp.get('summary_e5', '')
                if path and path != 'None' and isinstance(path, str) and path.endswith('.npy'):
                    e5_path = path
                    break

            # Fallback: try summery_emb_e5 from the target node itself
            if e5_path is None:
                for node in sample['paper_subgraph']['nodes']:
                    aids = str(node.get('identifiers', ''))
                    target_id = sample.get('target_arxiv_id', '')
                    if target_id and target_id in aids:
                        path = node.get('summery_emb_e5', '')
                        if path and path != 'None' and isinstance(path, str) and path.endswith('.npy'):
                            e5_path = path
                        break

            if e5_path is None:
                continue

            try:
                e5_emb = torch.tensor(
                    np.load(e5_path), dtype=torch.float32, device=fused_pool.device
                )  # [1024]
                if e5_emb.shape[0] != 1024:
                    continue
                query_vecs.append(fused_pool[i])
                key_vecs.append(e5_emb)
            except Exception:
                continue

        if len(query_vecs) < 2:
            return None

        q = torch.stack(query_vecs)   # [P, 4224]
        k = torch.stack(key_vecs)     # [P, 1024]

        # Project fused_pool → 1024 via learned linear (gradient flows through all 4224 dims)
        q_proj = F.normalize(self.e5_proj(q.to(next(self.e5_proj.parameters()).device)).float(), dim=-1)   # [P, 1024]
        k_proj = F.normalize(k.to(q_proj.device).float(), dim=-1)              # [P, 1024]

        logits = torch.matmul(q_proj, k_proj.T) / CONTRASTIVE_TAU  # [P, P]
        labels = torch.arange(len(query_vecs), device=fused_pool.device)
        return F.cross_entropy(logits, labels)

    def _gen_e5_alignment_loss(
        self,
        batch: List[Dict],
        decoder_hidden_states: List[torch.Tensor],
        prompt_lengths: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """InfoNCE: decoder hidden states for target tokens should embed close to
        the target paper's precomputed E5 embedding.

        Takes the last layer's hidden states, mean-pools over target token positions
        (after prompt), projects to 1024-dim E5 space, and computes InfoNCE against
        precomputed E5 embeddings of target papers.

        This forces the model to generate text that naturally embeds well in E5 space,
        directly optimizing for the retrieval metric.
        """
        # Use last layer hidden states: [B, seq_len, 4096]
        last_hidden = decoder_hidden_states[-1]
        batch_size = last_hidden.size(0)

        query_vecs = []  # mean-pooled decoder hidden states for target tokens
        key_vecs = []    # precomputed E5 embeddings of target papers

        for i, sample in enumerate(batch):
            # Get target paper E5 embedding (same path as _e5_alignment_loss)
            pos_papers = sample.get('positive_papers', [])
            e5_path = None
            for pp in pos_papers:
                path = pp.get('summary_e5', '')
                if path and path != 'None' and isinstance(path, str) and path.endswith('.npy'):
                    e5_path = path
                    break
            if e5_path is None:
                for node in sample['paper_subgraph']['nodes']:
                    aids = str(node.get('identifiers', ''))
                    target_id = sample.get('target_arxiv_id', '')
                    if target_id and target_id in aids:
                        path = node.get('summery_emb_e5', '')
                        if path and path != 'None' and isinstance(path, str) and path.endswith('.npy'):
                            e5_path = path
                        break
            if e5_path is None:
                continue

            try:
                e5_emb = torch.tensor(
                    np.load(e5_path), dtype=torch.float32, device=last_hidden.device
                )  # [1024]
                if e5_emb.shape[0] != 1024:
                    continue
            except Exception:
                continue

            # Mean-pool decoder hidden states over target token positions
            prompt_len = int(prompt_lengths[i].item())
            seq_len = last_hidden.size(1)
            if prompt_len >= seq_len:
                continue  # no target tokens
            target_hidden = last_hidden[i, prompt_len:, :]  # [T, 4096]
            pooled = target_hidden.mean(dim=0)  # [4096]

            query_vecs.append(pooled)
            key_vecs.append(e5_emb)

        if len(query_vecs) < 2:
            return None

        q = torch.stack(query_vecs)  # [P, 4096]
        k = torch.stack(key_vecs)    # [P, 1024]

        # Project decoder hidden → 1024 via gen_e5_proj (move proj to q's device for multi-GPU)
        dev = q.device
        q_proj = F.normalize(self.gen_e5_proj.to(dev)(q.float()).float(), dim=-1)  # [P, 1024]
        k_proj = F.normalize(k.to(dev).float(), dim=-1)                            # [P, 1024]

        logits = torch.matmul(q_proj, k_proj.T) / CONTRASTIVE_TAU  # [P, P]
        labels = torch.arange(len(query_vecs), device=dev)
        return F.cross_entropy(logits, labels)

    def _gen_thm_ft_alignment_loss(
        self,
        batch: List[Dict],
        decoder_hidden_states: List[torch.Tensor],
        prompt_lengths: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """InfoNCE: decoder hidden states for target tokens should embed close to
        the target paper's precomputed thm_ft embedding.

        This directly optimizes for the thm_ft retrieval metric we evaluate on.
        """
        if self.train_thmft_embs is None:
            return None

        last_hidden = decoder_hidden_states[-1]

        query_vecs = []
        key_vecs = []

        for i, sample in enumerate(batch):
            arxiv_id = sample.get('target_arxiv_id', '')
            if not arxiv_id or arxiv_id not in self.train_thmft_idx:
                continue

            thmft_idx = self.train_thmft_idx[arxiv_id]
            thmft_emb = torch.tensor(
                self.train_thmft_embs[thmft_idx], dtype=torch.float32, device=last_hidden.device
            )  # [4096]

            prompt_len = int(prompt_lengths[i].item())
            seq_len = last_hidden.size(1)
            if prompt_len >= seq_len:
                continue
            target_hidden = last_hidden[i, prompt_len:, :]  # [T, 4096]
            pooled = target_hidden.mean(dim=0)  # [4096]

            query_vecs.append(pooled)
            key_vecs.append(thmft_emb)

        if len(query_vecs) == 0:
            return None

        q = torch.stack(query_vecs)  # [P, 4096]
        k = torch.stack(key_vecs)    # [P, 4096]

        dev = q.device
        q_proj = F.normalize(self.gen_thm_ft_proj.to(dev)(q.float()).float(), dim=-1)  # [P, 4096]
        k_proj = F.normalize(k.to(dev).float(), dim=-1)                                # [P, 4096]

        if len(query_vecs) == 1:
            # cosine similarity loss when batch=1 (InfoNCE needs ≥2)
            return (1.0 - (q_proj * k_proj).sum(dim=-1)).mean()

        logits = torch.matmul(q_proj, k_proj.T) / CONTRASTIVE_TAU  # [P, P]
        labels = torch.arange(len(query_vecs), device=dev)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        batch: List[Dict],
        target_texts: Optional[List[str]] = None,
        phrases: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        freeze_phase: bool = False,
        **generation_kwargs,
    ):
        batch_size = len(batch)

        # ── Encode both graphs ─────────────────────────────────────────────────
        paper_vecs_list   = []
        theorem_vecs_list = []

        arxiv_ids = [sample.get('target_arxiv_id', '') for sample in batch]

        # informal_embs_list[i] = [P_i, 1024] E5 embeddings of paired informal theorems
        informal_embs_list = []

        for sample, arxiv_id in zip(batch, arxiv_ids):
            informal_pairs = sample.get('mathlib_subgraph', {}).get('informal_pairs', [])
            pv, _ = self.encode_paper_subgraph(
                sample['paper_subgraph'], arxiv_id, informal_pairs
            )
            if pv is None:
                pv = torch.zeros(1, ENC1_DIM, device=self.device)
            paper_vecs_list.append(pv)

            # collect pair embs separately for contrastive loss
            if informal_pairs:
                pair_embs = self._embed_informal_pairs(informal_pairs)
            else:
                pair_embs = None
            informal_embs_list.append(pair_embs)

            tv = self.encode_mathlib_subgraph(sample.get('mathlib_subgraph', {}))
            theorem_vecs_list.append(tv)

        # ── Bridge: lift paper vecs 1152 → 4224 ───────────────────────────────
        paper_vecs_list = [self.bridge(pv) for pv in paper_vecs_list]

        # ── Type embeddings: paper=0, theorem=1 ───────────────────────────────
        paper_type   = self.type_embed(torch.tensor(0, device=self.device))  # [4224]
        theorem_type = self.type_embed(torch.tensor(1, device=self.device))  # [4224]
        paper_vecs_list   = [pv + paper_type   for pv in paper_vecs_list]
        theorem_vecs_list = [tv + theorem_type for tv in theorem_vecs_list]

        # ── Pad to batch tensors ───────────────────────────────────────────────
        max_N = max(pv.size(0) for pv in paper_vecs_list)
        max_M = max(tv.size(0) for tv in theorem_vecs_list)

        paper_batch   = torch.zeros(batch_size, max_N, FUSED_DIM, device=self.device)
        theorem_batch = torch.zeros(batch_size, max_M, FUSED_DIM, device=self.device)
        paper_mask    = torch.zeros(batch_size, max_N, dtype=torch.bool, device=self.device)
        theorem_mask  = torch.zeros(batch_size, max_M, dtype=torch.bool, device=self.device)

        for i, (pv, tv) in enumerate(zip(paper_vecs_list, theorem_vecs_list)):
            n, m = pv.size(0), tv.size(0)
            paper_batch[i, :n]   = pv
            theorem_batch[i, :m] = tv
            paper_mask[i, :n]    = True
            theorem_mask[i, :m]  = True

        # ── Fusion cross-attention ─────────────────────────────────────────────
        if getattr(self, 'no_fusion', False):
            # Ablation: skip cross-attention, just concatenate
            paper_fused, theorem_fused = paper_batch, theorem_batch
        else:
            paper_fused, theorem_fused = self.fusion(
                paper_batch, theorem_batch, paper_mask, theorem_mask
            )

        # ── Concat → [B, N+M, 4224] ───────────────────────────────────────────
        fused      = torch.cat([paper_fused, theorem_fused], dim=1)
        fused_mask = torch.cat([paper_mask, theorem_mask], dim=1)

        # ── Graph dropout (training only) ──────────────────────────────────────
        if self.training and GRAPH_DROPOUT > 0:
            # Zero out random context vectors (whole token positions)
            drop_mask = torch.bernoulli(
                torch.full((batch_size, fused.size(1)), GRAPH_DROPOUT, device=self.device)
            ).bool()
            # Don't drop if it would zero out ALL tokens for a sample
            for i in range(batch_size):
                valid = fused_mask[i].sum().item()
                if drop_mask[i, :int(valid)].all():
                    drop_mask[i, 0] = False
            fused = fused.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # ── Baseline: zero out enc2 (theorem graph only) ──────────────────────
        if getattr(self, 'no_enc2', False):
            fused      = torch.cat([paper_fused, torch.zeros_like(theorem_fused)], dim=1)
            fused_mask = torch.cat([paper_mask,  torch.zeros_like(theorem_mask)],  dim=1)

        # ── Baseline: zero out full graph context ──────────────────────────────
        if getattr(self, 'no_graph', False):
            fused      = torch.zeros_like(fused)
            fused_mask = torch.zeros_like(fused_mask)
            fused_mask[:, 0] = True  # keep one valid token to avoid empty-mask errors

        # ── Attention pooling → [B, 4224] ─────────────────────────────────────
        # Learns which nodes matter — gradients flow non-uniformly to important nodes
        attn_logits = self.pool_attn(fused).squeeze(-1)          # [B, N+M]
        attn_logits = attn_logits.masked_fill(~fused_mask, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)        # [B, N+M]
        fused_pool = (attn_weights.unsqueeze(-1) * fused).sum(dim=1)  # [B, 4224]

        # ── Freeze phase: skip decoder entirely, use graph-only losses ───────
        if freeze_phase:
            link_loss  = self._link_prediction_loss(batch, paper_vecs_list)
            e5_loss    = self._e5_alignment_loss(batch, fused_pool)
            xcontr_loss = self._cross_encoder_contrastive_loss(batch, informal_embs_list)

            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if link_loss is not None:
                loss = loss + 0.4 * link_loss
            if e5_loss is not None:
                loss = loss + 1.0 * e5_loss
            if xcontr_loss is not None:
                loss = loss + 0.2 * xcontr_loss.to(loss.device)

            loss_details = {
                'total':       loss.item(),
                'ce':          0.0,
                'contrastive': 0.0,
                'cross_contr': xcontr_loss.item() if xcontr_loss is not None else 0.0,
                'align':       0.0,
                'link':        link_loss.item() if link_loss is not None else 0.0,
                'e5_align':    e5_loss.item() if e5_loss is not None else 0.0,
                'n_papers':    sum(paper_mask[i].sum().item() for i in range(batch_size)),
                'n_theorems':  sum(theorem_mask[i].sum().item() for i in range(batch_size)),
            }
            return loss, None, loss_details

        # ── Decoder ───────────────────────────────────────────────────────────
        # Move fused vectors to the decoder's input device (cuda:0, where embed_tokens lives)
        dd = self.encoder_device  # cuda:0 always; decoder.device may be 'auto' with device_map
        fused_d      = fused.to(dd)
        fused_mask_d = fused_mask.to(dd)
        type_ids = torch.zeros(batch_size, fused.size(1), dtype=torch.long, device=dd)

        decoder_out = self.decoder(
            encoder_hidden_states=fused_d,
            encoder_attention_mask=fused_mask_d,
            type_ids=type_ids,
            target_texts=target_texts,
            phrases=phrases,
            max_length=max_new_tokens,
            output_hidden_states=self.training,
            **generation_kwargs,
        )

        ce_loss, generated, hidden_info = decoder_out
        ce_neg = None
        gen_e5_loss = None

        # ── Graph margin loss: CE(correct graph) < CE(wrong graph) ────────────
        # Wrong graph = batch shifted by 1 (each sample gets a neighbour's graph)
        # Skip pairs where target text is identical — margin would be meaningless
        graph_margin_loss = None
        if self.training and batch_size > 1 and ce_loss is not None:
            wrong_fused    = torch.roll(fused_d, 1, dims=0)
            wrong_mask     = torch.roll(fused_mask_d, 1, dims=0)
            wrong_type_ids = torch.zeros(batch_size, fused.size(1), dtype=torch.long, device=dd)
            with torch.no_grad():
                wrong_decoder_out = self.decoder(
                    encoder_hidden_states=wrong_fused,
                    encoder_attention_mask=wrong_mask,
                    type_ids=wrong_type_ids,
                    target_texts=target_texts,
                    phrases=phrases,
                    max_length=max_new_tokens,
                    output_hidden_states=False,
                )
            ce_neg = wrong_decoder_out[0]
            if ce_neg is not None:
                # Skip pairs with identical target text
                valid_pairs = [
                    i for i in range(batch_size)
                    if (target_texts[i] if target_texts else '') !=
                       (target_texts[(i - 1) % batch_size] if target_texts else '')
                ]
                if valid_pairs:
                    graph_margin_loss = torch.clamp(
                        0.2 + ce_loss - ce_neg, min=0.0
                    )

        # ── Generated-text E5 alignment loss ─────────────────────────────────
        # Forces decoder to produce text whose hidden states embed close to target paper
        gen_thm_ft_loss = None
        if self.training and hidden_info is not None and ce_loss is not None:
            last4_hidden, prompt_lengths = hidden_info
            gen_e5_loss = self._gen_e5_alignment_loss(batch, last4_hidden, prompt_lengths)
            gen_thm_ft_loss = self._gen_thm_ft_alignment_loss(batch, last4_hidden, prompt_lengths)

        # ── Combined loss: CE + graph_margin + gate_penalty + gen_e5 + gen_thm_ft ─
        loss = ce_loss

        if loss is not None and graph_margin_loss is not None:
            loss = loss + 0.5 * graph_margin_loss.to(loss.device)

        # Gate penalty removed — don't push gates toward 0.6, let them learn freely

        # Gen-text E5 alignment: weight controlled by E5_GEN_W
        E5_GEN_W = getattr(self, '_e5_gen_weight', 0.5)
        if loss is not None and gen_e5_loss is not None:
            loss = loss + E5_GEN_W * gen_e5_loss.to(loss.device)

        # Gen-text thm_ft alignment: directly optimizes for thm_ft retrieval metric
        THM_FT_GEN_W = getattr(self, '_thm_ft_gen_weight', 0.0)
        if loss is not None and gen_thm_ft_loss is not None and THM_FT_GEN_W > 0:
            loss = loss + THM_FT_GEN_W * gen_thm_ft_loss.to(loss.device)

        ce_neg_val = ce_neg.item() if (ce_neg is not None and self.training) else 0.0
        ce_pos_val = ce_loss.item() if ce_loss is not None else 0.0
        loss_details = {
            'total':        loss.item()                  if loss is not None                  else 0.0,
            'ce':           ce_pos_val,
            'contrastive':  0.0,
            'cross_contr':  0.0,
            'align':        0.0,
            'graph_margin': graph_margin_loss.item()     if graph_margin_loss is not None      else 0.0,
            'gen_e5':       gen_e5_loss.item()           if gen_e5_loss is not None             else 0.0,
            'gen_thm_ft':   gen_thm_ft_loss.item()       if gen_thm_ft_loss is not None         else 0.0,
            'ce_neg':       ce_neg_val,
            'ce_gap':       ce_neg_val - ce_pos_val,
            'n_papers':     sum(paper_mask[i].sum().item()   for i in range(batch_size)),
            'n_theorems':   sum(theorem_mask[i].sum().item() for i in range(batch_size)),
        }
        return loss, generated, loss_details


# ==============================================================================
# TRAINER
# ==============================================================================

def get_root_title(sample: dict) -> str:
    """Return the root node's title + abstract as the phrase.
    Falls back to sample['phrase'] if root cannot be determined."""
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
    # Append root abstract for richer decoder context
    if root_node:
        abst_raw = root_node.get('summery_text', '') or root_node.get('abstract', '') or ''
        abst = ' '.join(abst_raw) if isinstance(abst_raw, list) else abst_raw
        abst = abst.strip()[:400]  # truncate to ~400 chars
        if abst:
            return f"{title}\nAbstract: {abst}"
    return title


class DualTrainer:
    def __init__(self, model: DualEncoderModel, train_loader, val_loader, config: Dict):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = config['device']

        # ── Separate parameter groups with different learning rates ────────────
        gnn_params        = []
        gate_params       = []
        cross_attn_params = []
        other_params      = []
        special_param_ids = set()

        # enc1.gnn
        for p in model.enc1.gnn.parameters():
            if p.requires_grad:
                gnn_params.append(p)
                special_param_ids.add(id(p))

        # enc2 W_struct + layer_norms
        for p in model.enc2.gnn.W_struct.parameters():
            if p.requires_grad:
                gnn_params.append(p)
                special_param_ids.add(id(p))
        for p in model.enc2.gnn.layer_norms.parameters():
            if p.requires_grad:
                gnn_params.append(p)
                special_param_ids.add(id(p))

        # Gate scalars (one per cross-attn layer) — need very high LR due to sigmoid squashing
        for cross_attn in model.decoder.cross_attn_layers.values():
            if cross_attn.gate_graph.requires_grad:
                gate_params.append(cross_attn.gate_graph)
                special_param_ids.add(id(cross_attn.gate_graph))

        # Cross-attn K/V projections — randomly initialized, need higher LR than LoRA
        for cross_attn in model.decoder.cross_attn_layers.values():
            for p in cross_attn.parameters():
                if p.requires_grad and id(p) not in special_param_ids:
                    cross_attn_params.append(p)
                    special_param_ids.add(id(p))

        # LoRA params — low lr like LeanDojo (1e-6) so they don't dominate over graph path
        lora_params = []
        # Everything else (bridge, fusion, type_embed, contrastive_proj)
        # LoRA params get lower lr (1e-6) so they don't dominate over graph path (LeanDojo uses 1e-6)
        for name, p in model.named_parameters():
            if p.requires_grad and id(p) not in special_param_ids:
                if 'lora_' in name:
                    lora_params.append(p)
                    special_param_ids.add(id(p))
                else:
                    other_params.append(p)

        lr = config['learning_rate']  # 2e-5
        cross_attn_lr = config.get('cross_attn_lr') or lr * 10  # default 2e-4
        lora_lr       = 1e-6           # LeanDojo: 1e-6 is stable, 1e-5 collapses
        gate_lr       = lora_lr * 10   # reduced from 50x — gates were saturating and corrupting decoder
        xattn_lr      = lora_lr * 10   # LeanDojo: 10x decoder lr

        # ── Staged training (Flamingo/BLIP-2 style) ──────────────────────────
        self.freeze_decoder_epochs = config.get('freeze_decoder_epochs', 0)
        self.decoder_lr_after = config.get('decoder_lr', lora_lr)  # LR for decoder after unfreeze
        # Store param references for freeze/unfreeze
        self._lora_params = lora_params
        self._cross_attn_params = cross_attn_params

        # Split param groups by device to avoid multi-GPU AdamW errors
        def _split_by_device(params, lr, weight_decay, tag):
            """Split a param list into per-device groups."""
            from collections import defaultdict
            by_dev = defaultdict(list)
            for p in params:
                by_dev[str(p.device)].append(p)
            groups = []
            for dev, ps in by_dev.items():
                groups.append({'params': ps, 'lr': lr, 'weight_decay': weight_decay, '_tag': tag, '_device': dev})
            return groups

        opt_groups = []
        opt_groups += _split_by_device(gnn_params,        lr / 10, 1e-2, 'gnn')
        lora_start = len(opt_groups)
        opt_groups += _split_by_device(lora_params,       lora_lr, 0.05, 'lora')
        lora_end = len(opt_groups)
        opt_groups += _split_by_device(other_params,      lr,      1e-2, 'other')
        xattn_start = len(opt_groups)
        opt_groups += _split_by_device(cross_attn_params, xattn_lr,1e-2, 'cross_attn')
        xattn_end = len(opt_groups)
        opt_groups += _split_by_device(gate_params,       gate_lr, 0.0,  'gates')

        self.optimizer = torch.optim.AdamW(opt_groups)
        # Store indices for freeze/unfreeze
        self._lora_group_indices = list(range(lora_start, lora_end))
        self._xattn_group_indices = list(range(xattn_start, xattn_end))

        logger.info(f"Optimizer param groups:")
        logger.info(f"  GNN:        lr={lr/10:.2e} ({len(gnn_params)} tensors)")
        logger.info(f"  LoRA:       lr={lora_lr:.2e} ({len(lora_params)} tensors) ← LeanDojo ratio")
        logger.info(f"  Other:      lr={lr:.2e} ({len(other_params)} tensors)")
        logger.info(f"  Cross-attn: lr={xattn_lr:.2e} ({len(cross_attn_params)} tensors) ← 10x LoRA")
        logger.info(f"  Gates:      lr={gate_lr:.2e} ({len(gate_params)} tensors) ← 250x LoRA")
        if self.freeze_decoder_epochs > 0:
            logger.info(f"  STAGED TRAINING: decoder frozen for first {self.freeze_decoder_epochs} epochs")
            logger.info(f"  Decoder LR after unfreeze: lora={self.decoder_lr_after:.2e}, xattn={self.decoder_lr_after*10:.2e}")

        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        from tqdm import tqdm
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_margin = 0.0
        total_gap = 0.0
        total_link = 0.0
        total_e5 = 0.0
        total_gen_e5 = 0.0
        total_gen_thm_ft = 0.0
        count = 0
        grad_accum = self.config.get('grad_accum', 1)
        accum_count = 0
        accum_details = None
        grad_norm = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        first_batch = True
        for batch in pbar:
            targets = [s.get('target_text') or s.get('target_title', '') for s in batch]
            phrases = [get_root_title(s) for s in batch]

            if first_batch and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"[MEM before first batch] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")
                first_batch = False

            in_freeze = (self.freeze_decoder_epochs > 0 and epoch <= self.freeze_decoder_epochs)
            try:
                loss, _, details = self.model(batch, target_texts=targets, phrases=phrases, freeze_phase=in_freeze)
            except torch.cuda.OutOfMemoryError as e:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.error(f"[OOM in forward] allocated={alloc:.2f}GB reserved={reserved:.2f}GB | {e}")
                raise
            if loss is None:
                continue
            # Gradient accumulation: scale loss, accumulate gradients
            try:
                (loss / grad_accum).backward()
            except torch.cuda.OutOfMemoryError as e:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.error(f"[OOM in backward] allocated={alloc:.2f}GB reserved={reserved:.2f}GB | {e}")
                raise
            accum_count += 1
            if accum_details is None:
                accum_details = {k: v for k, v in details.items()}
            else:
                for k in details:
                    accum_details[k] = accum_details.get(k, 0) + details.get(k, 0)
            if accum_count < grad_accum:
                # accumulate postfix from last sub-batch but don't step yet
                pbar.set_postfix(
                    loss=f"{details['total']:.4f}",
                    ce=f"{details['ce']:.4f}",
                    accum=f"{accum_count}/{grad_accum}",
                )
                continue
            # Average accumulated details
            for k in accum_details:
                accum_details[k] = accum_details[k] / grad_accum
            details = accum_details
            accum_details = None
            accum_count = 0
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += details['total']
            total_ce += details['ce']
            total_margin += details.get('graph_margin', 0.0)
            total_gap += details.get('ce_gap', 0.0)

            total_link += details.get('link', 0.0)
            total_e5 += details.get('e5_align', 0.0)
            total_gen_e5 += details.get('gen_e5', 0.0)
            total_gen_thm_ft += details.get('gen_thm_ft', 0.0)
            count += 1
            # Mid-epoch checkpoint every 2000 steps to survive crashes
            if count % 2000 == 0:
                mid_ckpt = {k: v for k, v in {
                    'epoch': epoch, 'val_loss': float('inf'),
                    'bridge_state_dict': self.model.bridge.state_dict(),
                    'fusion_state_dict': self.model.fusion.state_dict(),
                    'type_embed_state_dict': self.model.type_embed.state_dict(),
                    'e5_proj_state_dict': self.model.e5_proj.state_dict(),
                    'gen_e5_proj_state_dict': self.model.gen_e5_proj.state_dict(),
                    'gen_thm_ft_proj_state_dict': self.model.gen_thm_ft_proj.state_dict(),
                    'pool_attn_state_dict': self.model.pool_attn.state_dict(),
                    'fused_proj_state_dict': self.model.fused_proj.state_dict(),
                    'cross_proj_paper_state_dict': self.model.cross_proj_paper.state_dict(),
                    'cross_proj_mathlib_state_dict': self.model.cross_proj_mathlib.state_dict(),
                    'mem_bank': self.model.mem_bank, 'mem_ptr': self.model.mem_ptr, 'mem_full': self.model.mem_full,
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'enc1_gnn_state_dict': self.model.enc1.gnn.state_dict(),
                    'enc2_gnn_adapt_state_dict': {'W_struct': self.model.enc2.gnn.W_struct.state_dict(), 'layer_norms': self.model.enc2.gnn.layer_norms.state_dict()},
                }.items()}
                mid_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_midepoch_{epoch}_step{count}.pt')
                torch.save(mid_ckpt, mid_path)
                logger.info(f"  Mid-epoch checkpoint saved: {mid_path}")
            pbar.set_postfix(
                loss=f"{details['total']:.4f}",
                ce=f"{details['ce']:.4f}",
                gap=f"{details.get('ce_gap', 0):.4f}",
                margin=f"{details.get('graph_margin', 0):.4f}",
                ge5=f"{details.get('gen_e5', 0):.4f}",
                gtf=f"{details.get('gen_thm_ft', 0):.4f}",
                grad=f"{grad_norm:.2f}",
            )
            if count % 50 == 0:
                avg_ce = total_ce / count
                avg_margin = total_margin / count
                avg_gap = total_gap / count
                avg_link = total_link / count
                avg_e5 = total_e5 / count
                avg_gen_e5 = total_gen_e5 / count
                avg_gen_thm_ft = total_gen_thm_ft / count
                logger.info(
                    f"  Epoch {epoch} step {count} | "
                    f"loss={details['total']:.4f} ce={details['ce']:.4f} "
                    f"gap(neg-pos)={details.get('ce_gap',0):.4f} "
                    f"margin={details.get('graph_margin',0):.4f} "
                    f"link={details.get('link',0):.4f} e5={details.get('e5_align',0):.4f} "
                    f"gen_e5={details.get('gen_e5',0):.4f} gen_thm_ft={details.get('gen_thm_ft',0):.4f} | "
                    f"avg_ce={avg_ce:.4f} avg_gap={avg_gap:.4f} avg_margin={avg_margin:.4f} "
                    f"avg_link={avg_link:.4f} avg_e5={avg_e5:.4f} avg_gen_e5={avg_gen_e5:.4f} avg_gen_thm_ft={avg_gen_thm_ft:.4f} | "
                    f"grad_norm={grad_norm:.4f} | "
                    f"papers={details['n_papers']:.0f} theorems={details['n_theorems']:.0f}"
                )
        n = max(count, 1)
        return {
            'loss':        total_loss / n,
            'ce':          total_ce / n,
            'margin':      total_margin / n,
            'gap':         total_gap / n,
            'link':        total_link / n,
            'e5':          total_e5 / n,
            'gen_e5':      total_gen_e5 / n,
            'gen_thm_ft':  total_gen_thm_ft / n,
        }

    def val_epoch(self, epoch: int) -> float:
        from tqdm import tqdm
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"  Val {epoch}", dynamic_ncols=True)
            for batch in pbar:
                targets = [s.get('target_text') or s.get('target_title', '') for s in batch]
                phrases = [get_root_title(s) for s in batch]
                loss, _, details = self.model(batch, target_texts=targets, phrases=phrases)
                if loss is None:
                    continue
                total_loss += details['total']
                count += 1
                pbar.set_postfix(loss=f"{details['total']:.4f}")
        return total_loss / max(count, 1)

    def save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        ckpt = {
            'epoch': epoch,
            'val_loss': val_loss,
            'bridge_state_dict':             self.model.bridge.state_dict(),
            'fusion_state_dict':             self.model.fusion.state_dict(),
            'type_embed_state_dict':         self.model.type_embed.state_dict(),
            'e5_proj_state_dict':            self.model.e5_proj.state_dict(),
            'gen_e5_proj_state_dict':        self.model.gen_e5_proj.state_dict(),
            'gen_thm_ft_proj_state_dict':    self.model.gen_thm_ft_proj.state_dict(),
            'pool_attn_state_dict':          self.model.pool_attn.state_dict(),
            'fused_proj_state_dict':         self.model.fused_proj.state_dict(),
            'cross_proj_paper_state_dict':   self.model.cross_proj_paper.state_dict(),
            'cross_proj_mathlib_state_dict': self.model.cross_proj_mathlib.state_dict(),
            'mem_bank':                      self.model.mem_bank,
            'mem_ptr':                       self.model.mem_ptr,
            'mem_full':                      self.model.mem_full,
            'decoder_state_dict':            self.model.decoder.state_dict(),
            'enc1_gnn_state_dict':           self.model.enc1.gnn.state_dict(),
            'enc2_gnn_adapt_state_dict': {
                'W_struct':    self.model.enc2.gnn.W_struct.state_dict(),
                'layer_norms': self.model.enc2.gnn.layer_norms.state_dict(),
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
        torch.save(ckpt, path)
        logger.info(f"  Saved checkpoint: {path}")
        if best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(ckpt, best_path)
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

    def load_graph_only_checkpoint(self, path: str):
        """Load only graph/bridge/fusion weights, skip decoder. Used for stage 2 with a different decoder."""
        logger.info(f"Graph-only resume from checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.bridge.load_state_dict(ckpt['bridge_state_dict'])
        self.model.fusion.load_state_dict(ckpt['fusion_state_dict'])
        self.model.type_embed.load_state_dict(ckpt['type_embed_state_dict'])
        if 'e5_proj_state_dict' in ckpt:
            self.model.e5_proj.load_state_dict(ckpt['e5_proj_state_dict'])
        if 'gen_e5_proj_state_dict' in ckpt:
            ckpt_gen_shape = ckpt['gen_e5_proj_state_dict']['weight'].shape
            if ckpt_gen_shape == self.model.gen_e5_proj.weight.shape:
                self.model.gen_e5_proj.load_state_dict(ckpt['gen_e5_proj_state_dict'])
            else:
                logger.info(f"  Skipping gen_e5_proj: checkpoint shape {ckpt_gen_shape} != model shape {self.model.gen_e5_proj.weight.shape} (different decoder)")
        if 'gen_thm_ft_proj_state_dict' in ckpt:
            self.model.gen_thm_ft_proj.load_state_dict(ckpt['gen_thm_ft_proj_state_dict'])
        if 'pool_attn_state_dict' in ckpt:
            self.model.pool_attn.load_state_dict(ckpt['pool_attn_state_dict'])
        ckpt_fused_proj_shape = ckpt['fused_proj_state_dict']['weight'].shape
        if ckpt_fused_proj_shape == self.model.fused_proj.weight.shape:
            self.model.fused_proj.load_state_dict(ckpt['fused_proj_state_dict'])
            self.model.mem_bank.copy_(ckpt['mem_bank'])
            self.model.mem_ptr.copy_(ckpt['mem_ptr'])
            self.model.mem_full.copy_(ckpt['mem_full'])
        else:
            logger.info(f"  Skipping fused_proj/mem_bank: checkpoint shape {ckpt_fused_proj_shape} != model shape {self.model.fused_proj.weight.shape} (different decoder)")
        if 'cross_proj_paper_state_dict' in ckpt:
            self.model.cross_proj_paper.load_state_dict(ckpt['cross_proj_paper_state_dict'])
            self.model.cross_proj_mathlib.load_state_dict(ckpt['cross_proj_mathlib_state_dict'])
        self.model.enc1.gnn.load_state_dict(ckpt['enc1_gnn_state_dict'])
        self.model.enc2.gnn.W_struct.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['W_struct'])
        self.model.enc2.gnn.layer_norms.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['layer_norms'])
        logger.info(f"  Graph weights loaded. Decoder ({self.model.decoder.__class__.__name__}) starts fresh.")
        return 1  # always start from epoch 1

    def load_enc2_only_checkpoint(self, path: str):
        """Load enc2/bridge/fusion weights only, skip enc1 GNN and decoder.
        Symmetric counterpart to load_graph_only_checkpoint for the w/o enc1 ablation."""
        logger.info(f"Enc2-only resume from checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.bridge.load_state_dict(ckpt['bridge_state_dict'])
        self.model.fusion.load_state_dict(ckpt['fusion_state_dict'])
        self.model.type_embed.load_state_dict(ckpt['type_embed_state_dict'])
        if 'e5_proj_state_dict' in ckpt:
            self.model.e5_proj.load_state_dict(ckpt['e5_proj_state_dict'])
        if 'gen_e5_proj_state_dict' in ckpt:
            ckpt_gen_shape = ckpt['gen_e5_proj_state_dict']['weight'].shape
            if ckpt_gen_shape == self.model.gen_e5_proj.weight.shape:
                self.model.gen_e5_proj.load_state_dict(ckpt['gen_e5_proj_state_dict'])
            else:
                logger.info(f"  Skipping gen_e5_proj: checkpoint shape {ckpt_gen_shape} != model shape {self.model.gen_e5_proj.weight.shape} (different decoder)")
        if 'gen_thm_ft_proj_state_dict' in ckpt:
            self.model.gen_thm_ft_proj.load_state_dict(ckpt['gen_thm_ft_proj_state_dict'])
        if 'pool_attn_state_dict' in ckpt:
            self.model.pool_attn.load_state_dict(ckpt['pool_attn_state_dict'])
        ckpt_fused_proj_shape = ckpt['fused_proj_state_dict']['weight'].shape
        if ckpt_fused_proj_shape == self.model.fused_proj.weight.shape:
            self.model.fused_proj.load_state_dict(ckpt['fused_proj_state_dict'])
            self.model.mem_bank.copy_(ckpt['mem_bank'])
            self.model.mem_ptr.copy_(ckpt['mem_ptr'])
            self.model.mem_full.copy_(ckpt['mem_full'])
        else:
            logger.info(f"  Skipping fused_proj/mem_bank: checkpoint shape {ckpt_fused_proj_shape} != model shape {self.model.fused_proj.weight.shape} (different decoder)")
        if 'cross_proj_paper_state_dict' in ckpt:
            self.model.cross_proj_paper.load_state_dict(ckpt['cross_proj_paper_state_dict'])
            self.model.cross_proj_mathlib.load_state_dict(ckpt['cross_proj_mathlib_state_dict'])
        # enc2 only — skip enc1 GNN
        self.model.enc2.gnn.W_struct.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['W_struct'])
        self.model.enc2.gnn.layer_norms.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['layer_norms'])
        logger.info(f"  Enc2+bridge/fusion weights loaded. Enc1 GNN and decoder start fresh.")
        return 1

    def load_checkpoint(self, path: str):
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.bridge.load_state_dict(ckpt['bridge_state_dict'])
        self.model.fusion.load_state_dict(ckpt['fusion_state_dict'])
        self.model.type_embed.load_state_dict(ckpt['type_embed_state_dict'])
        if 'e5_proj_state_dict' in ckpt:
            self.model.e5_proj.load_state_dict(ckpt['e5_proj_state_dict'])
        if 'gen_e5_proj_state_dict' in ckpt:
            ckpt_gen_shape = ckpt['gen_e5_proj_state_dict']['weight'].shape
            if ckpt_gen_shape == self.model.gen_e5_proj.weight.shape:
                self.model.gen_e5_proj.load_state_dict(ckpt['gen_e5_proj_state_dict'])
            else:
                logger.info(f"  Skipping gen_e5_proj: checkpoint shape {ckpt_gen_shape} != model shape {self.model.gen_e5_proj.weight.shape} (different decoder)")
        if 'gen_thm_ft_proj_state_dict' in ckpt:
            self.model.gen_thm_ft_proj.load_state_dict(ckpt['gen_thm_ft_proj_state_dict'])
        if 'pool_attn_state_dict' in ckpt:
            self.model.pool_attn.load_state_dict(ckpt['pool_attn_state_dict'])
        self.model.fused_proj.load_state_dict(ckpt['fused_proj_state_dict'])
        if 'cross_proj_paper_state_dict' in ckpt:
            self.model.cross_proj_paper.load_state_dict(ckpt['cross_proj_paper_state_dict'])
            self.model.cross_proj_mathlib.load_state_dict(ckpt['cross_proj_mathlib_state_dict'])
        self.model.mem_bank.copy_(ckpt['mem_bank'])
        self.model.mem_ptr.copy_(ckpt['mem_ptr'])
        self.model.mem_full.copy_(ckpt['mem_full'])
        self.model.decoder.load_state_dict(ckpt['decoder_state_dict'])
        self.model.enc1.gnn.load_state_dict(ckpt['enc1_gnn_state_dict'])
        self.model.enc2.gnn.W_struct.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['W_struct'])
        self.model.enc2.gnn.layer_norms.load_state_dict(ckpt['enc2_gnn_adapt_state_dict']['layer_norms'])
        # Only load optimizer state if param groups are fully compatible
        try:
            ckpt_groups = ckpt['optimizer_state_dict']['param_groups']
            cur_groups  = self.optimizer.param_groups
            compatible  = (len(ckpt_groups) == len(cur_groups) and
                           all(len(cg['params']) == len(og['params'])
                               for cg, og in zip(ckpt_groups, cur_groups)))
            if compatible:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                logger.info(f"  Optimizer state restored")
            else:
                logger.info(f"  Optimizer state SKIPPED (param group mismatch — fresh optimizer)")
        except Exception as e:
            logger.info(f"  Optimizer state SKIPPED ({e})")
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Resumed at epoch {start_epoch}, best val_loss={ckpt['val_loss']:.4f}")
        return start_epoch

    def _log_gates(self, epoch: int):
        """Log cross-attention gate values for each decoder cross-attn layer."""
        lines = []
        for layer_idx_str, cross_attn in sorted(
            self.model.decoder.cross_attn_layers.items(), key=lambda x: int(x[0])
        ):
            gate_val = torch.sigmoid(cross_attn.gate_graph).item()
            if gate_val < 0.45:
                status = "⬇️  Suppressed"
            elif gate_val < 0.52:
                status = "⏳ Starting"
            elif gate_val < 0.65:
                status = "📈 Learning"
            else:
                status = "✅ Active"
            lines.append(f"  Layer {int(layer_idx_str):2d}: gate={gate_val:.4f}  {status}")
        logger.info(f"Cross-attention gates (epoch {epoch}):\n" + "\n".join(lines))

    def _log_samples(self, epoch: int, n: int = 3):
        """Run inference on a few val samples and log the generated text vs target."""
        self.model.eval()
        samples = []
        for batch in self.val_loader:
            samples.extend(batch)
            if len(samples) >= n:
                break
        samples = samples[:n]
        lines = [f"Generated samples (epoch {epoch}):"]
        with torch.no_grad():
            for i, s in enumerate(samples):
                phrase  = get_root_title(s)
                target  = s.get('target_text', '')[:120].replace('\n', ' ')
                _, generated, _ = self.model(
                    batch=[s],
                    phrases=[phrase],
                    target_texts=None,
                )
                gen = generated[0].strip()[:200].replace('\n', ' ') if generated else '(none)'
                lines.append(f"\n  [{i+1}] phrase: {phrase}")
                lines.append(f"       target : {target}...")
                lines.append(f"       generated: {gen}")
        logger.info("\n".join(lines))
        self.model.train()

    def train(self, resume_path: str = None, graph_only_resume: bool = False, enc2_only_resume: bool = False):
        import time, csv
        best_val = float('inf')
        start_epoch = 1
        loss_log_path = os.path.join(self.config['checkpoint_dir'], 'loss_log.csv')
        loss_log_exists = os.path.exists(loss_log_path)
        loss_log_file = open(loss_log_path, 'a', newline='')
        loss_log_writer = csv.writer(loss_log_file)
        if not loss_log_exists:
            loss_log_writer.writerow(['epoch', 'train_loss', 'train_ce', 'train_margin', 'train_gap', 'train_link', 'train_e5', 'train_gen_e5', 'val_loss'])
        if resume_path and os.path.exists(resume_path):
            if graph_only_resume:
                start_epoch = self.load_graph_only_checkpoint(resume_path)
            elif enc2_only_resume:
                start_epoch = self.load_enc2_only_checkpoint(resume_path)
            else:
                start_epoch = self.load_checkpoint(resume_path)
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[MEM after checkpoint load] allocated={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.1f}GB")
        for epoch in range(start_epoch, self.config['num_epochs'] + 1):
            # ── Staged training: freeze/unfreeze decoder ─────────────────
            if self.freeze_decoder_epochs > 0:
                if epoch <= self.freeze_decoder_epochs:
                    # Stage 1: freeze decoder LoRA only — cross-attn k/v stays trainable
                    for gi in self._lora_group_indices:
                        self.optimizer.param_groups[gi]['lr'] = 0.0
                    if epoch == start_epoch:
                        n_frozen = sum(p.numel() for p in self._lora_params)
                        logger.info(f"STAGE 1: Decoder LoRA FROZEN ({n_frozen:,} params at lr=0)")
                        logger.info(f"  Training: bridge, fusion, GNNs, gates, cross-attn k/v, contrastive")
                elif epoch == self.freeze_decoder_epochs + 1:
                    # Stage 2: unfreeze decoder LoRA with lower LR
                    for gi in self._lora_group_indices:
                        self.optimizer.param_groups[gi]['lr'] = self.decoder_lr_after
                    logger.info(f"STAGE 2: Decoder LoRA UNFROZEN")
                    logger.info(f"  LoRA lr={self.decoder_lr_after:.2e}")
            t0 = time.time()
            train_stats = self.train_epoch(epoch)
            train_loss = train_stats['loss']
            train_time = time.time() - t0
            t1 = time.time()
            has_val = len(self.val_loader) > 0
            if has_val:
                val_loss = self.val_epoch(epoch)
            else:
                val_loss = 0.0
            val_time   = time.time() - t1
            is_best = has_val and val_loss < best_val
            marker = " ★ BEST" if is_best else ""
            logger.info(
                f"\n{'='*70}\n"
                f"  EPOCH {epoch}/{self.config['num_epochs']} COMPLETE{marker}\n"
                f"  train_loss={train_loss:.4f}" + (f"  val_loss={val_loss:.4f}" if has_val else "  val=skipped") + f"\n"
                f"  breakdown: CE={train_stats['ce']:.4f}  "
                f"margin(×0.5)={train_stats['margin']:.4f}  "
                f"gap(neg-pos)={train_stats['gap']:.4f}  "
                f"link={train_stats['link']:.4f}  e5={train_stats['e5']:.4f}  gen_e5={train_stats['gen_e5']:.4f}\n"
                f"  train_time={train_time/60:.1f}min\n"
                f"{'='*70}"
            )
            self._log_gates(epoch)
            if has_val:
                self._log_samples(epoch)
            loss_log_writer.writerow([
                epoch,
                f'{train_loss:.4f}',
                f'{train_stats["ce"]:.4f}',
                f'{train_stats["margin"]:.4f}',
                f'{train_stats["gap"]:.4f}',
                f'{train_stats["link"]:.4f}',
                f'{train_stats["e5"]:.4f}',
                f'{train_stats["gen_e5"]:.4f}',
                f'{val_loss:.4f}' if has_val else '',
            ])
            loss_log_file.flush()
            if is_best:
                best_val = val_loss
            if epoch % self.config['checkpoint_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, best=is_best)
        loss_log_file.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     type=int,   default=4)
    parser.add_argument('--epochs',         type=int,   default=30)
    parser.add_argument('--lr',             type=float, default=2e-5)
    parser.add_argument('--checkpoint_dir', type=str,   default=CKPT_DIR)
    parser.add_argument('--max_train',      type=int,   default=None)
    parser.add_argument('--max_val',        type=int,   default=None)
    parser.add_argument('--data',           type=str,   default=DUAL_DATA)
    parser.add_argument('--filter_sources', action='store_true',
                        help='Keep only abstract_full/theorem_* samples (drop vague we_prove etc)')
    parser.add_argument('--resume',          type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--graph_only_resume', action='store_true', default=False,
                        help='Load only graph weights (enc1/enc2/bridge/fusion) from --resume, skip decoder (for stage 2 with different decoder)')
    parser.add_argument('--enc2_only_resume', action='store_true', default=False,
                        help='Load only enc2/bridge/fusion weights from --resume, skip enc1 GNN and decoder (for w/o enc1 ablation)')
    parser.add_argument('--no_enc2',        action='store_true', default=False,
                        help='Zero out enc2 (theorem graph) — paper-graph-only baseline')
    parser.add_argument('--no_graph',       action='store_true', default=False,
                        help='Baseline: feed zero context to decoder (no graph signal)')
    parser.add_argument('--no_fusion',     action='store_true', default=False,
                        help='Ablation: skip fusion cross-attention, just concatenate enc1+enc2 outputs')
    parser.add_argument('--no_enrichment', action='store_true', default=False,
                        help='Ablation: remove theorem enrichment nodes from enc1 (paper abstracts only)')
    parser.add_argument('--cross_attn_lr',  type=float, default=None,
                        help='LR for cross-attn K/V projections (default: lr * 10)')
    parser.add_argument('--freeze_decoder_epochs', type=int, default=0,
                        help='Freeze decoder LoRA+cross-attn for first N epochs (staged training)')
    parser.add_argument('--decoder_lr',    type=float, default=None,
                        help='Decoder LoRA LR after unfreeze (default: 1e-6)')
    parser.add_argument('--decoder_model', type=str,   default=DEFAULT_DECODER,
                        help='Decoder model. Options: deepseek-ai/deepseek-math-7b-instruct, Qwen/Qwen2-0.5B, meta-llama/Llama-2-7b-hf')
    parser.add_argument('--e5_gen_weight', type=float, default=0.5,
                        help='Weight for generated-text E5 alignment loss (retrieval-aware training). 0 disables it.')
    parser.add_argument('--thm_ft_gen_weight', type=float, default=0.5,
                        help='Weight for generated-text thm_ft alignment loss. Directly optimizes for thm_ft eval metric. 0 disables it.')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * grad_accum.')
    args = parser.parse_args()

    # Save every epoch for small runs, every 2 for full runs
    ckpt_freq = 1  # save every epoch — never lose 5h of training to a crash

    config = {
        'batch_size':      args.batch_size,
        'num_epochs':      args.epochs,
        'learning_rate':   args.lr,
        'cross_attn_lr':   args.cross_attn_lr,  # None = use default lr*10
        'checkpoint_dir':  args.checkpoint_dir,
        'checkpoint_freq': ckpt_freq,
        'device':          'cuda' if torch.cuda.is_available() else 'cpu',
        'no_graph':        args.no_graph,
        'no_enc2':         args.no_enc2,
        'no_fusion':       args.no_fusion,
        'no_enrichment':   args.no_enrichment,
        'freeze_decoder_epochs': args.freeze_decoder_epochs,
        'decoder_lr':      args.decoder_lr,
        'decoder_model':   args.decoder_model,
        'grad_accum':      args.grad_accum,
    }

    logger.info("="*70)
    logger.info("DUAL ENCODER TRAINING")
    logger.info("="*70)
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    train_ds = DualDataset(args.data, mode='train', max_samples=args.max_train, filter_sources=args.filter_sources)
    val_ds   = DualDataset(args.data, mode='val',   max_samples=args.max_val,   filter_sources=args.filter_sources) if args.max_val != 0 else []

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0, collate_fn=lambda x: x)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=0, collate_fn=lambda x: x) if val_ds else []

    model = DualEncoderModel(device=config['device'], no_graph=config.get('no_graph', False), no_enc2=config.get('no_enc2', False), no_fusion=config.get('no_fusion', False), no_enrichment=config.get('no_enrichment', False), decoder_model=config.get('decoder_model', DEFAULT_DECODER))
    model._e5_gen_weight = args.e5_gen_weight
    model._thm_ft_gen_weight = args.thm_ft_gen_weight
    logger.info(f"  Gen-text E5 alignment weight: {args.e5_gen_weight}")
    logger.info(f"  Gen-text thm_ft alignment weight: {args.thm_ft_gen_weight}")
    # Memory layout:
    # - Decoder base (7B): float16, spread across all visible GPUs via device_map='auto'
    # - LoRA adapters + cross-attn + gates: float32 (small, ~200MB, needed for stable training)
    # - Encoders + bridge/fusion: float32 on cuda:0
    for module in [model.enc1, model.enc2, model.bridge, model.fusion,
                   model.type_embed, model.pool_attn, model.fused_proj,
                   model.e5_proj, model.gen_e5_proj, model.gen_thm_ft_proj,
                   model.cross_proj_paper, model.cross_proj_mathlib]:
        module.to(torch.float32)
    # Cast LoRA adapters, cross-attention layers, and gates to float32 for training stability
    for name, param in model.decoder.named_parameters():
        if 'lora_' in name or 'cross_attn' in name or 'gate_graph' in name:
            param.data = param.data.to(torch.float32)

    trainer = DualTrainer(model, train_loader, val_loader, config)
    trainer.train(resume_path=args.resume, graph_only_resume=args.graph_only_resume, enc2_only_resume=args.enc2_only_resume)


if __name__ == '__main__':
    main()
