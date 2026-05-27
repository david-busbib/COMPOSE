"""
Clean Graph-to-Text Model with Proper Cross-Attention

This file contains a clean, well-structured implementation that:
1. Uses the EXACT encoder from train_model.py (FuturePredictionModel)
2. Implements a clean Mistral decoder with proper cross-attention
3. Fixes all type issues
4. Has clear documentation

Architecture:
-------------
Input Papers → Encoder (UniGLM + Multi-level Cross-Attention) → Node Embeddings
                                                                  ↓
                                                          Cross-Attention
                                                                  ↓
Title Tokens → Mistral Decoder (with LoRA) → Generated Title

The decoder learns from the encoder through cross-attention at layers [7, 15, 23, 31]
"""

import os
import time
import random
import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

from cross_attention_layers import (
    GraphAwareCrossAttention,
    HierarchicalCrossAttentionPooling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# COMPOSE arXiv release: only 7 defs reachable from FuturePredictionModel
# and MistralDecoder are retained. The full original file is at
# s2orc/twoencoder/model_clean.py (5453 lines).
#
# Removed (~3031 lines):
#   Trainer (1338L)                              - old training class, train_dual.py has its own loop
#   main + __name__ (260L)                       - model_clean.py-as-script entry, unused
#   GraphToTextModel (379L)                      - alternative architecture, never wired up
#   SubgraphDataset / TrainingSamplesDataset /   - superseded by train_dual.py's DualDataset
#     DualTrainingSamplesDataset (307L)
#   unified_contrastive_novelty_loss (193L)      - old loss, train_dual.py defines its own
#   info_nce_loss, compute_contrastive_loss,     - standalone loss fns, all unreferenced
#     reconstruction_loss, attention_entropy_loss,
#     contrastive_diversity_loss,
#     compute_{late_gate_saturation,edge_prediction,cross_attention_alignment}_loss (~383L)
#   create_root_focused_mask, find_latest_checkpoint, load_e5_model (~104L)
#
# COMPOSE staged training lives in train_dual.py + scripts/*.sh, not here.
# ============================================================================

class AttentionPool(nn.Module):
    """Attention pooling over a set of node embeddings"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [num_nodes, hidden_dim] node embeddings
            mask: [num_nodes] boolean mask (True = valid, False = masked/padding)
        Returns:
            [1, hidden_dim] pooled embedding
        """
        # 🔥 FIX C: Support masking to exclude masked nodes from attention
        scores = self.score(torch.tanh(self.proj(x)))  # [num_nodes, 1]

        if mask is not None:
            # Mask out invalid positions (set scores to -inf)
            mask = mask.unsqueeze(-1)  # [num_nodes, 1]
            scores = scores.masked_fill(~mask, float('-inf'))

            # 🛡️ Safety: Check if all nodes are masked (prevents NaN from softmax on all -inf)
            if torch.all(~mask):
                print("[WARN] All nodes masked in AttentionPool! Returning zero embedding.")
                return torch.zeros(1, x.size(-1), device=x.device, dtype=x.dtype)

        attn = torch.softmax(scores, dim=0)            # normalize
        pooled = torch.sum(attn * x, dim=0, keepdim=True)  # weighted sum
        return pooled

class TextEncoder(nn.Module):
    """
    Simplified TextEncoder: combines SPECTER embeddings at 768-dim, then projects to hidden_dim.

    Weights: 0.6*summary + 0.0*text + 0.3*title
    Single projection: 768 → hidden_dim (saves ~2.3M params vs 4 separate projections)
    """
    def __init__(self, hidden_dim: int = 1024, use_precomputed: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_precomputed = use_precomputed

        # Single projection layer: 768 → hidden_dim
        self.proj = nn.Linear(1024, hidden_dim)

        # Keep 0.0 * text_emb explicitly for future use
        self.weights = [0.6, 0.1, 0.3]  # summary, text, title
    
    
    
    def encode_papers(self, papers: List[Dict], mask_status: Dict[str, bool], return_raw: bool = False, title_only: bool = False, return_separate: bool = False):
        """
        Encode papers: combine SPECTER embeddings at 768-dim, then project to hidden_dim.

        Args:
            papers: List of paper dicts with embedding paths
            mask_status: Dict indicating which papers are masked
            return_raw: If True, return 768-dim embeddings without projection
            title_only: If True, return only title embedding (for target computation)
            return_separate: If True, return dict with separate 'summary', 'title' embeddings

        Returns:
            [N, hidden_dim] tensor (or [N, 768] if return_raw=True)
            Or dict with separate embeddings if return_separate=True
        """
        device = self.proj.weight.device

        if not self.use_precomputed:
            raise NotImplementedError("Only precomputed embeddings supported")

        embeddings = []

        # Check if E5 embedding files exist
        num_missing = sum(
            1 for p in papers
            if not all(p.get(k) and isinstance(p.get(k), str) and os.path.exists(p[k])
                    for k in ['title_emb_e5', 'summery_emb_e5'])  # Check E5 embeddings
        )
        if num_missing > 0.3 * len(papers):
            return None

        def safe_load(path_or_array):
            """Load embedding from path or array, normalize to unit length."""
            # 🔥 UPDATED: E5 embeddings are 1024-dim (not 768)
            if path_or_array is None:
                return torch.zeros(1024, device=device)
            if isinstance(path_or_array, str):
                if os.path.exists(path_or_array):
                    try:
                        if os.path.getsize(path_or_array) == 0:
                            print(f"Empty file: {path_or_array}")
                            return torch.zeros(1024, device=device)
                        e = torch.from_numpy(np.load(path_or_array)).float().to(device)
                        e = e / (e.norm() + 1e-6)  # normalize to unit length
                        return e
                    except (EOFError, ValueError, OSError) as ex:
                        print(f"Corrupted file: {path_or_array} - {ex}")
                        return torch.zeros(1024, device=device)
                else:
                    print(f"Missing file: {path_or_array}")
                    return torch.zeros(1024, device=device)
            if isinstance(path_or_array, np.ndarray):
                return torch.from_numpy(path_or_array).float().to(device)
            return torch.zeros(1024, device=device)
        
        # 🔥 NEW: Support returning separate embeddings for cross-attention
        if return_separate:
            summaries, titles = [], [] 
            for paper in papers:
                sum_emb = safe_load(paper.get("summery_emb_e5"))
                title_emb = safe_load(paper.get("title_emb_e5"))

                # Check if title should be masked
                paper_id = paper.get('paper_id')
                if mask_status.get(paper_id, False):
                    title_emb = torch.zeros_like(title_emb)

                summaries.append(sum_emb)
                titles.append(title_emb)

            return {
                'summary': torch.stack(summaries), 
                'title': torch.stack(titles),  
                "root_abstract":None # [N, 768]
            }

        for paper in papers:
            # Load all embeddings (E5: 1024-dim each)
            sum_emb   = safe_load(paper.get("summery_emb_e5"))
            title_emb = safe_load(paper.get("title_emb_e5"))

            # 🔥 FIX 2: Support title-only mode and selective masking
            if title_only:
                combined = title_emb  # [1024]
            else:
                # For INPUT: Check if title should be masked for this paper
                paper_id = paper.get('paper_id')
                if mask_status.get(paper_id, False):
                    title_emb = torch.zeros_like(title_emb)

                # If masked: 0.6*summary + 0.0*zeros (masked title)
                combined = (self.weights[0] * sum_emb +
                           self.weights[2] * title_emb)  # Zeros if masked
                           # self.weights[3] * auth_emb)  # [1024]

            if return_raw:
                embeddings.append(combined)
            else:
                # Project to hidden_dim (768 → 1024)
                emb = self.proj(combined.unsqueeze(0)).squeeze(0)  # [hidden_dim]
                embeddings.append(emb)

        return torch.stack(embeddings)

class NodeCrossAttentionEncoder(nn.Module):
    """
    Per-node cross-attention: learns how each node's title relates to its summary.

    Key innovation: Instead of fixed weighted combination (0.6*summary + 0.3*title ),
    use learnable cross-attention where title queries summary to learn "which summary parts matter for title?"

    Benefits:
    - Adaptive per paper (not one-size-fits-all)
    - Explicit title-summary relationship learning
    - Interpretable attention weights
    """
    def __init__(self, embed_dim=1024, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Cross-attention: title queries summary
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Learnable query for masked titles: "what should title be?"
        # Use zeros initialization instead of randn for stability with large datasets
        self.title_query = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 🔥 COMMENTED OUT: Year doesn't help with novelty, focusing on content only
        # self.year_proj = nn.Linear(16, embed_dim)

        # Final fusion to hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, summary_emb, title_emb, is_masked=None):
        """
        Learn title-summary relationship through cross-attention.

        Args:
            summary_emb: [batch, 768] - paper content
            title_emb: [batch, 768] - paper title (or zeros if masked)
            is_masked: [batch] - boolean tensor indicating masked titles

        Returns:
            node_repr: [batch, hidden_dim]
            attn_weights: [batch, num_heads, 1, 1] - attention weights (for analysis)
        """
        batch_size = summary_emb.size(0)
        device = summary_emb.device

        # Summary as context (key/value for attention)
        summary_context = summary_emb.unsqueeze(1)  # [batch, 1, 768]

        # Title as query (or learned query if masked)
        # For masked titles: use learnable query "what should title be?"
        # For unmasked titles: use actual title
        title_query = torch.where(
            is_masked.unsqueeze(-1).unsqueeze(-1),  # [batch, 1, 1]
            self.title_query.expand(batch_size, 1, -1),  # Learnable query
            title_emb.unsqueeze(1)  # Actual title
        )  # [batch, 1, 768]

        # Cross-attention: title (or "what should title be?") queries summary
        # This learns: "Which parts of summary are relevant for the title?"
        title_aware_summary, attn_weights = self.cross_attn(
            query=title_query,      # What to look for
            key=summary_context,    # Where to look
            value=summary_context   # What to retrieve
        )  # title_aware_summary: [batch, 1, 768]

        title_aware_summary = title_aware_summary.squeeze(1)  # [batch, 768]

        # 🔥 COMMENTED OUT: Year doesn't help with novelty, focusing on content only
        # Add year context (temporal patterns)

        # Focusing purely on content relationships
        combined = title_aware_summary  # [batch, 768]

        # Project to hidden dimension with normalization
        node_repr = self.fusion(combined)  # [batch, 1024]

        return node_repr, attn_weights

class SimpleGNN(nn.Module):
    """
    Simple Graph Neural Network for encoding citation structure.

    Uses DIRECTED edges (not bidirectional) with separate incoming/outgoing message passing.

    For edge src → tgt (src cites tgt):
      - tgt receives incoming messages from src (papers that cite it)
      - src receives outgoing messages from tgt (papers it cites)

    This captures both "who I cite" and "who cites me" separately.

    Args:
        hidden_dim: Input dimension (1024 for E5 embeddings)
        struct_dim: Output structural vector dimension (128 recommended)
        num_layers: Number of message passing layers (2 = 2-hop neighbors)
    """
    def __init__(self, hidden_dim=1024, struct_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.struct_dim = struct_dim

        # Message passing layers (processes combined incoming+outgoing messages)
        self.W_msg = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Gate for residual connection (prevents over-smoothing)
        self.gate = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1)  # Input: [old_h, new_h]
            for _ in range(num_layers)
        ])

        # Final projection to structural space
        self.W_struct = nn.Linear(hidden_dim, struct_dim)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, node_embs, edges):
        """
        Args:
            node_embs: [N, 1024] - E5 embeddings of paper summaries
            edges: List of tuples [(src, tgt), ...] - directed citation edges
                   (src → tgt means "src cites tgt")

        Returns:
            struct_vecs: [N, struct_dim] - structural vectors per paper
        """
        N = node_embs.size(0)
        device = node_embs.device
        h = node_embs  # [N, 1024]

        # Build TWO directed adjacency matrices
        adj_out = torch.zeros(N, N, device=device)  # Papers I cite (outgoing)
        adj_in = torch.zeros(N, N, device=device)   # Papers that cite me (incoming)

        for src, tgt in edges:
            if 0 <= src < N and 0 <= tgt < N:
                # Edge: src → tgt (src cites tgt)
                adj_out[src, tgt] = 1.0  # src has outgoing edge to tgt
                adj_in[tgt, src] = 1.0   # tgt has incoming edge from src

        # Message passing layers
        for layer_idx in range(self.num_layers):
            # Aggregate messages from BOTH directions separately
            degree_out = adj_out.sum(dim=1, keepdim=True) + 1e-6  # [N, 1]
            degree_in = adj_in.sum(dim=1, keepdim=True) + 1e-6    # [N, 1]

            # Outgoing: messages from papers I cite
            msg_out = torch.matmul(adj_out, h) / degree_out  # [N, 1024]

            # Incoming: messages from papers that cite me
            msg_in = torch.matmul(adj_in, h) / degree_in  # [N, 1024]

            # Combine both message types (simple sum - could also use learned weights)
            messages = msg_out + msg_in  # [N, 1024]

            # Transform messages
            h_new = self.W_msg[layer_idx](messages)  # [N, 1024]
            h_new = F.relu(h_new)

            # Gated residual connection (prevents over-smoothing)
            gate_input = torch.cat([h, h_new], dim=-1)  # [N, 2048]
            gate_weight = torch.sigmoid(self.gate[layer_idx](gate_input))  # [N, 1]
            h = gate_weight * h_new + (1 - gate_weight) * h  # Weighted combination

            # Layer normalization
            h = self.layer_norms[layer_idx](h)

        # Project to structural space (compress to smaller dimension)
        struct_vecs = self.W_struct(h)  # [N, 1024] -> [N, struct_dim]

        # 🔥 DIAGNOSTIC: Log GNN statistics every 100 calls

        return struct_vecs

class FuturePredictionModel(nn.Module):
    """
    Simplified encoder model using TextEncoder (SPECTER) + GraphAwareCrossAttention.

    🔥 SIMPLIFIED: Removed UniGLM to prevent eval-mode diversity collapse.
    Now uses: TextEncoder (pretrained SPECTER) → GraphAwareCrossAttention → Decoder
    """

    def __init__(self,
                 graph_input_dim: int = 1024,
                 text_hidden_dim: int = 1024,
                 hidden_dim: int = 1024,
                 num_graph_layers: int = 3,
                 use_precomputed_embeddings: bool = True,
                 device: str = 'cuda',
                 tokenizer=None):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # 🔥 NEW: Tokenizer for root abstract token-level embeddings
        self.tokenizer = tokenizer
        if tokenizer is not None:
            # Token embedding layer (reuse vocab size from tokenizer)
            vocab_size = len(tokenizer)
            self.token_embeddings = nn.Embedding(vocab_size, hidden_dim).to(device)
            logger.info(f"\n{'='*60}")
            logger.info("TOKEN-LEVEL ROOT ENCODING")
            logger.info(f"{'='*60}")
            logger.info(f"Added token embeddings: vocab_size={vocab_size}, dim={hidden_dim}")
            logger.info(f"Root abstract will use token-level attention")
            logger.info(f"Neighbors will use E5 sentence embeddings")
            logger.info(f"{'='*60}\n")
        else:
            self.token_embeddings = None

        # 🔥 REMOVED: Type embeddings (replaced with simple *2 boost for summaries)
        # No learnable parameters - just multiply summary vectors by 2
        logger.info(f"{'='*60}")
        logger.info("SUMMARY BOOST (No Learnable Embeddings)")
        logger.info(f"{'='*60}")
        logger.info("Summary vectors: multiplied by 2 (attention boost)")
        logger.info("  Summary: [1152] * 2 = higher attention weight")
        logger.info("  Title: [1024] * 1 = normal attention weight")
        logger.info("  No learnable type embeddings - simpler!")
        logger.info(f"{'='*60}\n")

        self.text_encoder = TextEncoder(
            hidden_dim=text_hidden_dim,
            use_precomputed=use_precomputed_embeddings
        )
        self.log_vars = nn.Parameter(torch.zeros(3))

        # 🔥 NEW: Node Cross-Attention Encoder
        # Replaces fixed weighted combination with learnable title-summary attention
        self.node_cross_attn = NodeCrossAttentionEncoder(
            embed_dim=1024
            
            ,
            hidden_dim=hidden_dim,
            num_heads=8
        ).to(device)
        logger.info("\n" + "="*60)
        logger.info("NODE CROSS-ATTENTION ENCODER")
        logger.info("="*60)
        logger.info("Per-node learning: title queries summary to learn relationships")
        logger.info("="*60 + "\n")

        # 🔥 SIMPLIFIED ARCHITECTURE: Removed UniGLM (was collapsing in eval mode)
        # Now using: TextEncoder → NodeCrossAttention → GraphAwareCrossAttention → Decoder
        # This preserves diversity and uses pretrained SPECTER embeddings
        logger.info("\n" + "="*60)
        logger.info("SIMPLIFIED ENCODER (No UniGLM)")
        logger.info("="*60)
        logger.info("Using: TextEncoder (SPECTER) → NodeCrossAttn → GraphAwareCrossAttention → Decoder")
        logger.info("Removed UniGLM to prevent eval-mode collapse")
        logger.info("="*60 + "\n")

        # Multi-level cross-attention architecture
        logger.info("Initializing Graph-Aware Cross-Attention...")

        # Graph-Aware Cross-Attention (learns citation graph structure)
        self.graph_aware_cross_attn = GraphAwareCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        ).to(device)
        logger.info(f"GraphAwareCrossAttention ({hidden_dim}d, 8 heads)")

        # Hierarchical Cross-Attention Pooling (now at hidden_dim, not uniglm_dim)
        self.subgraph_pooling = HierarchicalCrossAttentionPooling(
            hidden_dim=hidden_dim,  # Changed from self.uniglm_dim to hidden_dim
            num_heads=8,
            dropout=0.1
        ).to(device)
        logger.info(f"HierarchicalCrossAttentionPooling ({hidden_dim}d, 8 heads)")
        logger.info("="*60 + "\n")

        # 🔥 NEW: Graph Neural Network for citation structure
        self.gnn = SimpleGNN(
            hidden_dim=1024,      # E5 embedding dimension
            struct_dim=128,       # Structural vector dimension (smaller as suggested)
            num_layers=2          # 2-hop message passing
        ).to(device)
        logger.info("\n" + "="*60)
        logger.info("SIMPLE GNN (Citation Structure Encoding)")
        logger.info("="*60)
        logger.info(f"Input: E5 summary embeddings [N, 1024]")
        logger.info(f"Output: Structural vectors [N, 128]")
        logger.info(f"Message passing: 2 layers (2-hop neighbors)")
        logger.info(f"Direction: Separate incoming (cited) + outgoing (cites)")
        logger.info(f"Gated residual: Prevents over-smoothing")
        logger.info(f"Final: Summaries become [N, 1152] = [1024 E5 + 128 structure]")
        logger.info("="*60 + "\n")

        self.year_encoder = nn.Linear(1, text_hidden_dim).to(device)
        self.ref_proj = nn.Linear(1024, self.text_encoder.hidden_dim).to(device)

        # 🎯 ROOT IDENTIFICATION: Root node is always placed at position 0
        # No learnable marker needed - structural reordering in encode_subgraph()
        logger.info("✅ Root identification: position-based (root always at index 0)")

        # 🔥 NEW: Simple projection for single-node graphs (bypass NodeCrossAttention)
        # Preserves pretrained SPECTER quality without learned transformations
        self.single_node_proj = nn.Linear(1024, hidden_dim).to(device)
        logger.info("✅ Added single_node_proj: 768 → 1024 (preserves SPECTER for 1-node graphs)")

        # Projection layers
        self.text_proj = nn.Linear(text_hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.min_year = 1900
        self.max_year = 2024

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.text_pool = AttentionPool(text_hidden_dim)

        # Semantic head
        self.text_sem_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.text_encoder.hidden_dim)
        )

        self.graph_hidden_dim = hidden_dim

        # Prediction heads
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, 1024)  # Project to E5 embedding space
        )

        # Initialize node_predictor
        for m in self.node_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.5)

        # 🔥 UPDATED: Latent projection layer (1024 → 1024) for E5 alignment
        # Projects encoder root embeddings to E5 title embedding space
        # E5 embeddings are 1024-dim, so this is now identity-like
        self.latent_proj = nn.Linear(hidden_dim, 1024).to(device)
        nn.init.xavier_uniform_(self.latent_proj.weight)

        logger.info("\n" + "="*60)
        logger.info("LATENT PROJECTION LAYER")
        logger.info("="*60)
        logger.info(f"Added latent_proj: {hidden_dim} → 1024 (E5 space)")
        logger.info("Purpose: Align encoder outputs with E5 title embeddings")
        logger.info("Note: Same dimension, learns refinement transformation")
        logger.info("="*60 + "\n")

        # 🔥 COMMENTED OUT: Citation Importance MLP (Multiple Instance Learning)
        # Will be replaced with graph attention bias approach
        # # Learns to predict importance scores from citation context embeddings
        # # Uses max pooling over multiple contexts per paper
        # self.citation_importance_mlp = nn.Sequential(
        #     nn.Linear(1024, 256),  # E5 embedding → hidden
        #     nn.ReLU(),
        #     nn.Linear(256, 1)  # hidden → importance score (unbounded)
        # ).to(device)
        # logger.info("\n" + "="*60)
        # logger.info("CITATION IMPORTANCE MLP (Multiple Instance Learning)")
        # logger.info("="*60)
        # logger.info("Architecture: E5(1024) → ReLU(256) → Score(1)")
        # logger.info("Purpose: Learn which citation contexts indicate paper importance")
        # logger.info("Approach: Pointwise scoring + Max pooling (MIL)")
        # logger.info("Output: Sigmoid-constrained to [0.5, 2.5] range")
        # logger.info("Effect: Modulates summary_bias in decoder cross-attention")
        # logger.info("="*60 + "\n")

    # 🔥 COMMENTED OUT: Citation Importance MLP computation
    # Will be replaced with graph attention bias approach
    #     """
    #     Compute citation importance using TARGET-AWARE approach.
    #
    #     Priority:
    #     1. If target paper cites this paper → use TARGET's citation context (HIGHEST weight)
    #     2. Otherwise → use MIL over all citation contexts (LOWER weight)
    #
    #     Target citations get higher weight because they're directly relevant to generation.
    #
    #     Args:
    #         paper: Paper dictionary with:
    #             - 'target_citation': Target paper's citation to this paper (if exists)
    #             - 'citation_context': All citation contexts from other papers
    #
    #     Returns:
    #         importance: Scalar tensor in range:
    #             - Target citations: [1.0, 1.5] (stronger boost)
    #             - Other citations: [0.8, 1.2] (gentle hint)
    #     """
    #     # 🔥 PRIORITY 1: Check if target paper cites this paper
    #
    #     # 🔍 DEBUG: Track if we're receiving target_citation data
    #         self._debug_citation_stats = {'total': 0, 'target_cited': 0, 'has_field': 0}
    #     self._debug_citation_stats['total'] += 1
    #         self._debug_citation_stats['has_field'] += 1
    #         self._debug_citation_stats['target_cited'] += 1
    #
    #         logger.info(f"[CITATION-DEBUG] Processed {self._debug_citation_stats['total']} papers: "
    #                    f"{self._debug_citation_stats['has_field']} with target_citation field, "
    #                    f"{self._debug_citation_stats['target_cited']} cited by target")
    #
    #         # Target cites this paper - use TARGET's citation context!
    #
    #             try:
    #
    #                 # Score target's citation context
    #
    #                 # 🔥 HIGHER WEIGHT for target citations: [1.0, 1.5]
    #                 # sigmoid(x) * 0.5 + 1.0 maps to [1.0, 1.5]
    #                 # This gives target-cited papers 25% more attention than neutral
    #
    #                 # Debug logging
    #                     self._target_citation_count = 0
    #                 self._target_citation_count += 1
    #
    #                     logger.info(f"[TARGET-AWARE] Used {self._target_citation_count} target citations (importance: {importance.item():.3f})")
    #
    #             except Exception as e:
    #                 # Fall through to normal MIL if loading fails
    #                 pass
    #
    #     # 🔥 PRIORITY 2: Fallback to MIL over all citation contexts
    #
    #     # Collect E5 embeddings for all citation contexts
    #         # Check if E5 embedding path exists
    #             # Load from .npy file (like title_emb_e5, summery_emb_e5)
    #                 try:
    #                     context_embs.append(emb)
    #                 except Exception as e:
    #                     # Skip if file not found or error
    #                     continue
    #             # Fallback: if it's already an embedding (list/tensor) - for backward compatibility
    #             elif isinstance(emb_path, list):
    #                 context_embs.append(emb)
    #             elif isinstance(emb_path, torch.Tensor):
    #                 context_embs.append(emb)
    #
    #     # If no citation contexts available, return neutral importance
    #
    #     # Stack embeddings: [num_contexts, 1024]
    #
    #     # Score each context independently: [num_contexts, 1]
    #
    #     # Max pooling: take the best evidence [1]
    #
    #     # Constrain with sigmoid and scale to [0.8, 1.2] (gentle hint)
    #     # sigmoid(x) maps to [0, 1]
    #     # sigmoid(x) * 0.4 + 0.8 maps to [0.8, 1.2]
    #     # Softer scaling allows attention to override if needed
    #

    def encode_subgraph(self, subgraph: Dict, extra_node_embs=None, extra_edges=None):
        """
        Encode a single subgraph and return structured cross-attention embeddings.

        🔥 UPDATED: Removed root abstract (not available in prediction scenario)
        🔥 UPDATED: Phrase moved to decoder self-attention (not cross-attention)
        Now returns: neighbor summaries + neighbor titles + optional theorem nodes

        Args:
            subgraph: Dict containing:
                - 'papers': List of paper dicts (all from subgraph, no target)
                - 'edges': List of edge tuples
            extra_node_embs: Optional [P, 1024] tensor of extra node E5 embeddings
                             (e.g. informal theorem nodes). Fed into GNN alongside
                             paper nodes so they participate in message passing.
            extra_edges: Optional list of (src_idx, tgt_idx) tuples for extra nodes,
                         where indices are 0-based into extra_node_embs rows
                         (offset by N added automatically).

        Returns:
            Dict containing:
                - 'cross_attn_embeds': [total_vectors, hidden_dim] - all cross-attention vectors
                - 'type_ids': [total_vectors] - type ID for each vector (0=summary, 1=title, 2=theorem)
                - 'num_vectors': int - total number of vectors
                - 'num_paper_nodes': int - N, number of original paper nodes
        """
        # 🔥 UPDATED ARCHITECTURE: E5 for all neighbors only (no root, no phrase)
        # Phrase is now in decoder's self-attention, not cross-attention

        # Get E5 embeddings for all papers in subgraph
        # No mask_status needed - we're not masking anything\
        
        mask_status = {p['paper_id']: False for p in subgraph['papers']}
        emb_dict = self.text_encoder.encode_papers(
            subgraph['papers'],
            mask_status,
            return_separate=True
        )

        if emb_dict is None:
            return None

        summary_embs = emb_dict['summary'].to(self.device)  # [num_papers, 1024]
        title_embs = emb_dict['title'].to(self.device)      # [num_papers, 1024]

        num_papers = summary_embs.size(0)

        # 🔥 NEW: GNN on summary embeddings to get structural vectors
        # Create mapping from paper_id to index
        paper_id_to_idx = {p['paper_id']: i for i, p in enumerate(subgraph['papers'])}

        # Convert edges from paper_ids to indices
        edges = subgraph.get('edges', [])
        edge_list = []
        for e in edges:
            src_id = e['source']
            tgt_id = e['target']
            # Only add edge if both nodes are in the subgraph
            if src_id in paper_id_to_idx and tgt_id in paper_id_to_idx:
                src_idx = paper_id_to_idx[src_id]
                tgt_idx = paper_id_to_idx[tgt_id]
                edge_list.append((src_idx, tgt_idx))

        # Run GNN on summaries + optional extra nodes (informal theorem nodes)
        if extra_node_embs is not None and extra_node_embs.size(0) > 0:
            extra_node_embs = extra_node_embs.to(self.device)
            gnn_input = torch.cat([summary_embs, extra_node_embs], dim=0)  # [N+P, 1024]
            combined_edges = list(edge_list)
            if extra_edges:
                for src, tgt in extra_edges:
                    combined_edges.append((src + num_papers, tgt + num_papers))
            struct_vecs_full = self.gnn(gnn_input, combined_edges)  # [N+P, 128]
            struct_vecs = struct_vecs_full[:num_papers]              # [N, 128]
            extra_struct = struct_vecs_full[num_papers:]             # [P, 128]
            extra_with_struct = torch.cat([extra_node_embs, extra_struct], dim=-1)  # [P, 1152]
        else:
            struct_vecs = self.gnn(summary_embs, edge_list)          # [N, 128]
            extra_with_struct = None

        # Concatenate structural vectors to summaries
        summary_with_struct = torch.cat([summary_embs, struct_vecs], dim=-1)  # [num_papers, 1152]

        # 🔥 DIAGNOSTIC: Check how much structure changes embeddings

        # 🔥 NEW: No root - all papers are neighbors from historical subgraph
        cross_attn_vectors = []
        type_ids = []

        # Add all paper summaries (no boost here - boost happens in decoder attention)
        neighbor_summaries = summary_with_struct  # [num_papers, 1152]
        cross_attn_vectors.append(neighbor_summaries)
        type_ids.extend([0] * num_papers)

        # 🔥 NEW: Pad titles to 1152 (same as summaries) for uniform tensor shape
        # Titles are 1024-dim, pad with zeros to 1152-dim
        # CrossAttentionLayer will use type_ids to know which projection to use
        neighbor_titles_padded = F.pad(title_embs, (0, 128), value=0.0)  # [num_papers, 1024] → [num_papers, 1152]
        cross_attn_vectors.append(neighbor_titles_padded)
        type_ids.extend([1] * num_papers)

        # Append theorem nodes (type_id=2) if present
        if extra_with_struct is not None:
            cross_attn_vectors.append(extra_with_struct)           # [P, 1152]
            type_ids.extend([2] * extra_with_struct.size(0))

        # Concatenate all vectors (summaries + titles + theorem nodes)
        # All vectors are now 1152-dim (titles are zero-padded, theorems are E5+struct)
        cross_attn_embeds = torch.cat(cross_attn_vectors, dim=0)  # [total_vectors, 1152]
        type_ids_tensor = torch.tensor(type_ids, dtype=torch.long, device=self.device)  # [total_vectors]

        # 🔥 GRAPH STRUCTURE LEARNING: Return node-level embeddings for edge prediction
        # Average of summary + title = one embedding per paper
        node_level_embeds = (summary_embs + title_embs) / 2.0  # [num_papers, 1024]

        # 🔥 GRAPH STRUCTURE LEARNING: Extract edge information
        edges = subgraph.get('edges', [])
        edge_list = [(e['source'], e['target']) for e in edges] if edges else []
        paper_ids = [p['paper_id'] for p in subgraph['papers']]

        # 🔥 COMMENTED OUT: Compute citation importance scores for each paper
        # Will be replaced with graph attention bias
        #     citation_importance_scores.append(importance)
        #
        # # Stack into tensor: [num_papers]

        return {
            'cross_attn_embeds': cross_attn_embeds,
            'type_ids': type_ids_tensor,
            'num_vectors': cross_attn_embeds.size(0),
            'num_paper_nodes': num_papers,
            # 🔥 NEW: For graph structure learning
            'node_embeds': node_level_embeds,  # [num_papers, 1024]
            'edge_list': edge_list,            # [(source_id, target_id), ...]
            'paper_ids': paper_ids,            # [paper_id1, paper_id2, ...]
            # 🔥 COMMENTED OUT: Citation importance scores - will use graph bias instead
            # 'citation_importance': citation_importance_scores  # [num_papers] in range [0.5, 2.5]
        }

class CrossAttentionLayer(nn.Module):
    """
    Numerically Safe Cross-attention layer for Mistral.
    Forces attention score calculation in Float32 to prevent NaN/Inf in Mixed Precision.

    🔥 NEW: Supports mixed-dimension encoder inputs (1152 for summaries, 1024 for titles)
    Applies *2 boost to summary attention scores (in decoder, not encoder)
    """
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        dropout: float = 0.1,
        summary_dim: int = 1152,  # 🔥 NEW: 1024 E5 + 128 structure
        title_dim: int = 1024     # 🔥 NEW: 1024 E5 only
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.scaling = self.head_dim ** -0.5

        # 🔥 NEW: Store dimensions
        self.summary_dim = summary_dim
        self.title_dim = title_dim

        # Learnable temperature (clamped in forward to prevent explosion)
        self.temperature = nn.Parameter(torch.tensor(0.1))

        # 🔥 NEW: Separate projection layers for summaries and titles
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_summary = nn.Linear(summary_dim, hidden_size, bias=False)  # 1152 → hidden
        self.v_proj_summary = nn.Linear(summary_dim, hidden_size, bias=False)  # 1152 → hidden
        self.k_proj_title = nn.Linear(title_dim, hidden_size, bias=False)      # 1024 → hidden
        self.v_proj_title = nn.Linear(title_dim, hidden_size, bias=False)      # 1024 → hidden
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Gating
        self.gate_graph = nn.Parameter(torch.zeros(1))

        # 🔥 REMOVED: Learnable summary_bias parameter
        # Now using fixed *2 boost in encode_subgraph() instead of learnable bias
        # Simpler and more interpretable!
        # self.summary_bias = nn.Parameter(torch.tensor(0.7))  # REMOVED

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 OLD: [num_papers] importance scores (not used)
        type_ids: Optional[torch.Tensor] = None  # 🔥 NEW: [num_vectors] type IDs (0=summary, 1=title)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # --- NaN TRAP: INPUT CHECK ---
        if torch.isnan(hidden_states).any():
            print("🚨 CRITICAL: Decoder input (hidden_states) contains NaNs!")
        if torch.isnan(encoder_hidden_states).any():
            print("🚨 CRITICAL: Encoder input (encoder_hidden_states) contains NaNs!")

        batch_size, text_len, _ = hidden_states.shape
        num_vectors = encoder_hidden_states.size(1)

        # 🔥 NEW: Project Q (from decoder hidden states) — cast to float32 to match proj weights
        Q = self.q_proj(hidden_states.float()).view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 🔥 NEW: Handle mixed dimensions - split encoder_hidden_states by type_ids
        if type_ids is not None and type_ids.size(0) == num_vectors:
            # Split by type: 0=summary (1152), 1=title (1024, zero-padded to 1152)
            enc_device = encoder_hidden_states.device
            summary_mask = (type_ids == 0).to(enc_device)  # [num_vectors]
            title_mask = (type_ids == 1).to(enc_device)    # [num_vectors]

            num_summaries = summary_mask.sum().item()
            num_titles = title_mask.sum().item()

            # Extract summaries and titles
            summaries = encoder_hidden_states[:, summary_mask, :]  # [batch, num_summaries, 1152]
            titles_padded = encoder_hidden_states[:, title_mask, :]  # [batch, num_titles, 1152] (with padding)

            # 🔥 NEW: Remove padding from titles (take first 1024 dims)
            titles = titles_padded[:, :, :self.title_dim]  # [batch, num_titles, 1024]

            # Ensure dtype alignment
            target_dtype = self.k_proj_summary.weight.dtype
            if summaries.dtype != target_dtype:
                summaries = summaries.to(target_dtype)
            if titles.dtype != target_dtype:
                titles = titles.to(target_dtype)

            # Project K, V separately for summaries and titles
            K_summary = self.k_proj_summary(summaries).view(batch_size, num_summaries, self.num_heads, self.head_dim).transpose(1, 2)
            V_summary = self.v_proj_summary(summaries).view(batch_size, num_summaries, self.num_heads, self.head_dim).transpose(1, 2)

            K_title = self.k_proj_title(titles).view(batch_size, num_titles, self.num_heads, self.head_dim).transpose(1, 2)
            V_title = self.v_proj_title(titles).view(batch_size, num_titles, self.num_heads, self.head_dim).transpose(1, 2)

            # Concatenate K and V in original order (preserve position)
            # We need to reconstruct the original order using type_ids
            K_list = []
            V_list = []
            summary_idx = 0
            title_idx = 0
            for tid in type_ids:
                if tid == 0:  # Summary
                    K_list.append(K_summary[:, :, summary_idx, :])  # [batch, num_heads, head_dim]
                    V_list.append(V_summary[:, :, summary_idx, :])
                    summary_idx += 1
                else:  # Title
                    K_list.append(K_title[:, :, title_idx, :])
                    V_list.append(V_title[:, :, title_idx, :])
                    title_idx += 1

            K = torch.stack(K_list, dim=2)  # [batch, num_heads, num_vectors, head_dim]
            V = torch.stack(V_list, dim=2)  # [batch, num_heads, num_vectors, head_dim]
        else:
            # Fallback: assume all vectors are same dimension (backward compatibility)
            # This shouldn't happen in new code, but keep for safety
            target_dtype = self.k_proj_summary.weight.dtype
            if encoder_hidden_states.dtype != target_dtype:
                encoder_hidden_states = encoder_hidden_states.to(target_dtype)

            # Use summary projections as default (1152 dim)
            K = self.k_proj_summary(encoder_hidden_states).view(batch_size, num_vectors, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj_summary(encoder_hidden_states).view(batch_size, num_vectors, self.num_heads, self.head_dim).transpose(1, 2)

        # --- NaN TRAP: PROJECTIONS ---
        if torch.isnan(Q).any(): print("🚨 CRITICAL: NaN in Q projection")
        if torch.isnan(K).any(): print("🚨 CRITICAL: NaN in K projection")

        # Force FP32 for Stability
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)

        # 2. Compute Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # --- NaN TRAP: SCORES ---
        if torch.isnan(scores).any():
            print("🚨 CRITICAL: NaN in Attention Scores (pre-softmax). Q or K might be too large.")
            # Debug values
            print(f"Max Q: {Q.max()}, Max K: {K.max()}")

        # 3. Apply Temperature
        temp = torch.clamp(self.temperature, min=0.01, max=1.0)
        scores = scores / temp

        # 4. 🔥 NEW: Apply *2 boost to SUMMARY scores (in decoder, not encoder!)
        # This is where the boost happens - summaries get 2x attention weight
        if type_ids is not None and type_ids.size(0) == num_vectors:
            # 🔥 ATTENTION REWEIGHTING: Control summary vs title influence
            # type_ids: 0 = summary (abstract + GNN), 1 = title
            scores_device = scores.device
            summary_mask = (type_ids == 0).to(scores_device)  # [num_vectors]
            title_mask = (type_ids == 1).to(scores_device)    # [num_vectors]

            # scores shape: [batch, num_heads, text_len, num_vectors]
            # Expand masks to match scores shape
            summary_mask_expanded = summary_mask.view(1, 1, 1, -1).expand_as(scores)
            title_mask_expanded = title_mask.view(1, 1, 1, -1).expand_as(scores)

            # 🔥 TUNABLE WEIGHTS (EASY TO ADJUST):
            # summary_weight: Multiplier for summary attention scores
            #   - 2.0 = default boost (summaries get 2x weight)
            #   - Higher (3.0, 4.0) = focus even more on abstracts
            # title_weight: Multiplier for title attention scores
            #   - 0.0 = completely ignore titles (prevent copying) ← START HERE
            #   - 0.1 = allow minimal title influence
            #   - 1.0 = equal weight to titles
            summary_weight = 2.0  # Boost summaries (abstracts + GNN structure)
            title_weight = 0.0    # Suppress titles (prevent copying)

            # Apply reweighting
            scores = torch.where(summary_mask_expanded, scores * summary_weight, scores)
            scores = torch.where(title_mask_expanded, scores * title_weight, scores)

            # # 🔥 DIAGNOSTIC: Monitor attention distribution every 500 forward passes
            # # NOTE: Assumes summary/title split — invalid in dual encoder setup (all type_id=0).
            #     CrossAttentionLayer._attn_monitor_counter = 0
            # CrossAttentionLayer._attn_monitor_counter += 1
            #     with torch.no_grad():

        # 5. Masking
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e4)

        # 6. Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # --- NaN TRAP: SOFTMAX ---
        if torch.isnan(attn_weights).any():
            print("🚨 CRITICAL: NaN in Softmax output. Scores likely contained Inf.")

        # Cast back to V dtype
        attn_weights = attn_weights.to(V.dtype)
        attn_weights = self.dropout(attn_weights)

        # 7. Weighted Sum
        attn_output = torch.matmul(attn_weights, V)

        # --- NaN TRAP: OUTPUT ---
        if torch.isnan(attn_output).any():
            print("🚨 CRITICAL: NaN in Attention Output (Weighted Sum).")

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, text_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        gate = torch.sigmoid(self.gate_graph)
        attn_output = attn_output * gate

        if return_attention_weights:
            return attn_output, attn_weights.mean(dim=1).mean(dim=1)
        
        return attn_output, None

class MistralDecoder(nn.Module):
    """
    Mistral-7B decoder with cross-attention to encoder outputs.

    Cross-attention is injected at specific layers [7, 15, 23, 31] to allow
    the decoder to learn from encoder node embeddings during generation.
    """
    def __init__(
        self,
        encoder_dim: int = 1024,
        # ✅ MODEL SELECTION - uncomment the one you want to use:
        # model_name: str = "Qwen/Qwen2-0.5B",  # 896-dim, 24 layers, 500M params (RECOMMENDED for short text)
        model_name: str = "mistralai/Mistral-7B-v0.1",  # 4096-dim, 32 layers, 7B params

        # ✅ CROSS-ATTENTION LAYERS - adjust based on model:
        # cross_attn_layers: List[int] = [4, 8, 12, 16],  # For Qwen2-0.5B (24 layers total)
        cross_attn_layers: List[int] = [6, 12, 18, 24, 30],  # For Mistral-7B (32 layers total) - 5 layers evenly spaced

        use_lora: bool = True,
        lora_rank: int = 32,  # Increased from 8 to 32 for more capacity
        lora_target_modules: list = None,  # Custom target modules for LoRA
        use_8bit: bool = False,
        device: str = "cuda",
        freeze_first_n_layers: int = 0,
        # ✅ NEW: Load pretrained LoRA checkpoint
        lora_checkpoint_path: str = None,  # e.g., "mistral_math_adapted/checkpoint-4500"

    ):
        super().__init__()
        self.device = 'cuda:0' if device == 'auto' else device
        self.encoder_dim = encoder_dim
        self.cross_attn_layer_indices = cross_attn_layers

        logger.info(f"Loading decoder model: {model_name}")

        # Load tokenizer
        # 🔥 MULTI-GPU FIX: use_fast=False to avoid "Already borrowed" error with DataParallel
        # Fast tokenizers (Rust-based) are not thread-safe across multiple GPU replicas
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # ✅ FIX: Qwen2 tokenizer needs special token configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            # Qwen2 doesn't have BOS token, use EOS token instead
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set bos_token_id to eos_token_id: {self.tokenizer.bos_token_id}")

        # 🔥 NEW: Add structure tokens for research idea generation
        special_tokens_dict = {'additional_special_tokens': ['<IDEA>', '</IDEA>']}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"✅ Added {num_added_tokens} special tokens for research idea generation: <IDEA>, </IDEA>")

        # Load model
        # ✅ Qwen2-0.5B works well with float32 (it's small enough)
        # ✅ Mistral-7B needs float16 to fit in memory
        if use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            # 🔥 FIX: Add use_safetensors=False for PyTorch 2.9 compatibility
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map={"": device},  # all on same device
                    torch_dtype=torch.float16,
                )
            except ValueError as e:
                if "could not determine the shape" in str(e):
                    logger.warning(f"Safetensors loading failed, falling back to PyTorch format: {e}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map={"": device},
                        torch_dtype=torch.float16,
                        use_safetensors=False,
                    )
                else:
                    raise
            # self.model.gradient_checkpointing_enable()
            
        else:
            # Determine dtype based on model size
            if "Qwen2-0.5B" in model_name or "Qwen2-1.5B" in model_name:
                dtype = torch.float32  # Qwen2 small models work fine with float32
            else:
                dtype = torch.float16  # Larger models need float16

            # Build architecture from config only — weights come from our checkpoint,
            # so downloading the base model's 14GB weights would be wasted.
            _hf_config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(_hf_config)
            self.model = self.model.to(dtype)
            _target_device = "cuda" if device == "auto" else device
            self.model = self.model.to(_target_device)
            logger.info(f"Loaded model architecture with dtype: {dtype}")

        # ✅ NEW: Load pretrained LoRA checkpoint if provided
        if lora_checkpoint_path is not None:
            from peft import PeftModel
            logger.info(f"Loading pretrained LoRA checkpoint from: {lora_checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint_path)
            logger.info("✅ Loaded pretrained LoRA weights")

            # 🔥 CRITICAL FIX: Enable training on LoRA adapters
            # The checkpoint has inference_mode=True which sets requires_grad=False
            # We need to unfreeze LoRA params so decoder can adapt to graph embeddings
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                    trainable_count += 1
            logger.info(f"🔓 Unfroze {trainable_count} LoRA parameters for fine-tuning on graph task")

        # 🔥 NEW: Resize model embeddings to accommodate new special tokens
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"✅ Resized model embeddings to {len(self.tokenizer)} tokens")

        # Get decoder hidden dimension (varies by model)
        # Qwen2-0.5B: 896, Qwen2-1.5B: 1536, Mistral-7B: 4096
        self.decoder_dim = self.model.config.hidden_size
        logger.info(f"Decoder hidden dimension: {self.decoder_dim}")

        # Project encoder outputs to decoder dimension
        # ✅ For Qwen2-0.5B (896): encoder (1024) → 896 (simpler projection)
        # ✅ For Mistral (4096): encoder (1024) → 2048 → 4096 (2-layer MLP)
        if self.decoder_dim <= 1024:
            # Small model: simple direct projection
            self.encoder_proj = nn.Sequential(
                nn.Linear(encoder_dim, self.decoder_dim, bias=False),
                nn.GELU(),
                nn.Dropout(0.1),
            ).to('cuda:0' if device == 'auto' else device).to(torch.float32)
            logger.info(f"Using simple projection: {encoder_dim} → {self.decoder_dim}")
        else:
            # Large model: 2-layer MLP for better semantic alignment
            self.encoder_proj = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim * 2, bias=False),  # 1024 → 2048
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(encoder_dim * 2, self.decoder_dim, bias=False)  # 2048 → decoder_dim
            ).to('cuda:0' if device == 'auto' else device).to(torch.float32)
            logger.info(f"Using 2-layer projection: {encoder_dim} → {encoder_dim * 2} → {self.decoder_dim}")

        # 🔥 FIX: Initialize with proper variance (xavier_normal_)
        # Default init was too small, causing tiny outputs that collapse after normalization
        for m in self.encoder_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # Apply LoRA (skip if already loaded from checkpoint)
        if use_lora and lora_checkpoint_path is None:
            # ✅ Adjust dropout based on model size
            # Smaller models (Qwen2) benefit from LOWER dropout (more overfitting = good for small datasets)
            # Larger models (Mistral) need HIGHER dropout (prevent overfitting)
            if self.decoder_dim <= 1024:
                lora_dropout_val = 0.05  # Lower dropout for small models
                logger.info(f"Using LOW LoRA dropout ({lora_dropout_val}) for small model (easier to fine-tune)")
            else:
                lora_dropout_val = 0.1  # Standard dropout for large models
                logger.info(f"Using STANDARD LoRA dropout ({lora_dropout_val}) for large model")

            logger.info(f"Applying LoRA with rank {lora_rank}")

            # Use custom target modules if provided, otherwise use default
            if lora_target_modules is not None:
                target_mods = lora_target_modules
                logger.info(f"Using custom LoRA target modules: {target_mods}")
            else:
                target_mods = [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (added k_proj)
                    "gate_proj", "up_proj", "down_proj",     # Feed-forward network
                    "lm_head"                                 # Output layer (CRITICAL!)
                ]
                logger.info(f"Using default LoRA target modules (with lm_head)")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=lora_dropout_val,  # ✅ Adaptive dropout
                target_modules=target_mods,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Gradient checkpointing DISABLED - incompatible with custom cross-attention injection
        # When we manually inject cross-attention in forward pass, gradient checkpointing
        # breaks because it tries to recompute activations but can't handle our custom hooks
        logger.info("Enabling gradient checkpointing for Mistral decoder")
        self.model.gradient_checkpointing_enable()

        # Inject cross-attention layers
        logger.info(f"Injecting cross-attention at layers: {cross_attn_layers}")

        # Clear GPU cache before injection
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

        self.cross_attn_layers = nn.ModuleDict()
        self.cross_attn_norms = nn.ModuleDict()

        base_model = self._get_base_model()

        gate_init_strategy = {
            3: 0.0,   # sigmoid(0.0) = 0.50 - all layers start neutral
            7: 0.0,   # sigmoid(0.0) = 0.50
            11: 0.0,  # sigmoid(0.0) = 0.50
            15: 0.0,  # sigmoid(0.0) = 0.50
            19: 0.0,  # sigmoid(0.0) = 0.50
            23: 0.0,  # sigmoid(0.0) = 0.50
            27: 0.0,  # sigmoid(0.0) = 0.50
            31: 0.0   # sigmoid(0.0) = 0.50
        }

        logger.info("="*80)
        logger.info("🎯 Neutral Gate Initialization (all gates = 0.5)")
        logger.info("="*80)

        for i, layer_idx in enumerate(cross_attn_layers):
            if layer_idx < len(base_model.layers):
                # Create and move cross-attention layer
                _init_device = 'cuda:0' if device == 'auto' else device
                cross_attn = CrossAttentionLayer(
                    hidden_size=self.decoder_dim,  # ✅ Use decoder_dim (works for both Qwen2 and Mistral)
                    num_heads=self.model.config.num_attention_heads,
                    dropout=0.1
                ).to(_init_device)

                # 🔥 OPTION D: Initialize gate based on layer depth
                gate_init_value = gate_init_strategy.get(layer_idx, 2.0)  # default 2.0
                cross_attn.gate_graph.data.fill_(gate_init_value)

                logger.info(f"  Layer {layer_idx:2d}: gate_init={gate_init_value:.1f} → sigmoid≈{torch.sigmoid(torch.tensor(gate_init_value)):.2f}")

                self.cross_attn_layers[str(layer_idx)] = cross_attn

                # Create and move RMSNorm layer
                norm = nn.RMSNorm(
                    self.decoder_dim,  # ✅ Use decoder_dim
                    eps=self.model.config.rms_norm_eps
                ).to(_init_device)
                self.cross_attn_norms[str(layer_idx)] = norm

        logger.info("="*80 + "\n")

        logger.info(f"Decoder ready with {len(self.cross_attn_layers)} cross-attention layers")

        # 🔧 CRITICAL FIX: Initialize cross-attention from self-attention weights
        # Instead of random initialization, copy pretrained self-attention weights
        # This makes cross-attention start with useful patterns instead of noise
        logger.info("Initializing cross-attention layers from self-attention weights...")
        self._initialize_cross_attention_from_self_attention(cross_attn_layers)

        # Sync cross_attn_norms and cross_attn_layers to match each transformer layer's device
        # (needed when device_map='auto' spreads layers across multiple GPUs)
        base_model = self._get_base_model()
        for layer_idx_str in self.cross_attn_layers:
            layer_idx = int(layer_idx_str)
            if layer_idx < len(base_model.layers):
                try:
                    layer_device = next(base_model.layers[layer_idx].parameters()).device
                except StopIteration:
                    continue
                self.cross_attn_layers[layer_idx_str].to(layer_device)
                self.cross_attn_norms[layer_idx_str].to(layer_device)

        if freeze_first_n_layers > 0:
            self._freeze_decoder_layers(freeze_first_n_layers)

    def _initialize_cross_attention_from_self_attention(self, cross_attn_layers: List[int]):
        """
        Initialize cross-attention weights by copying from self-attention.

        This gives cross-attention a head start with pretrained weights instead of
        random initialization, making it immediately useful to the decoder.
        """
        base_model = self._get_base_model()

        for layer_idx in cross_attn_layers:
            if layer_idx >= len(base_model.layers):
                continue

            # Get self-attention and cross-attention layers
            self_attn = base_model.layers[layer_idx].self_attn
            cross_attn = self.cross_attn_layers[str(layer_idx)]

            logger.info(f"  Layer {layer_idx}: Initializing cross-attn from self-attn")

            # Helper to get base weight
            def get_base_weight(module):
                if hasattr(module, 'base_layer'):
                    return module.base_layer.weight
                return module.weight

            # Q projection: decoder hidden → decoder hidden (same dims, can copy)
            cross_attn.q_proj.weight.data.copy_(get_base_weight(self_attn.q_proj).data)

            # 🔥 NEW: K, V projections now separate for summaries and titles
            # K_summary, V_summary: 1152 (E5 + structure) → decoder hidden (4096)
            # K_title, V_title: 1024 (E5 only) → decoder hidden (4096)
            # Self-attn: decoder hidden (4096) → decoder hidden (4096)
            # Dimensions don't match! Use Xavier initialization
            torch.nn.init.xavier_uniform_(cross_attn.k_proj_summary.weight)
            torch.nn.init.xavier_uniform_(cross_attn.v_proj_summary.weight)
            torch.nn.init.xavier_uniform_(cross_attn.k_proj_title.weight)
            torch.nn.init.xavier_uniform_(cross_attn.v_proj_title.weight)

            # O projection: same dims, can copy
            cross_attn.o_proj.weight.data.copy_(get_base_weight(self_attn.o_proj).data)

            logger.info(f"    Q: copied from self-attn")
            logger.info(f"    K_summary, V_summary (1152→hidden): Xavier init")
            logger.info(f"    K_title, V_title (1024→hidden): Xavier init")
            logger.info(f"    O: copied from self-attn")

        logger.info(f"✅ Cross-attention initialized from pretrained self-attention weights")

    def _freeze_decoder_layers(self, num_layers_to_freeze: int):
        """
        Freeze the first N layers of the Mistral decoder to force reliance on cross-attention.
        
        This prevents the pretrained LM from adapting freely in early layers, creating
        optimization pressure to use the encoder signal via cross-attention.
        """
        base_model = self._get_base_model()
        
        total_layers = len(base_model.layers)
        num_layers_to_freeze = min(num_layers_to_freeze, total_layers)
        
        logger.info(f"🔒 Freezing first {num_layers_to_freeze}/{total_layers} decoder layers...")
        
        frozen_params = 0
        for layer_idx in range(num_layers_to_freeze):
            for param in base_model.layers[layer_idx].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Count trainable params
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        
        logger.info(f"   Frozen params: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logger.info(f"   Trainable decoder params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"   + Cross-attention layers: {len(self.cross_attn_layers)} (always trainable)")

    def _get_base_model(self):
        """Get base model handling PEFT wrapping - returns the actual MistralModel"""
        # Navigate through PEFT wrapping to get to actual model layers
        # Structure: PeftModelForCausalLM -> LoraModel -> MistralForCausalLM -> MistralModel

        current = self.model
        max_depth = 10  # Safety limit to prevent infinite loops
        depth = 0

        # Unwrap PEFT layers until we find a model with 'layers' attribute
        while depth < max_depth:
            # Check if we've reached the model with layers
            if hasattr(current, 'layers'):
                return current

            # Continue unwrapping
            if hasattr(current, 'base_model'):
                current = current.base_model
                depth += 1
            elif hasattr(current, 'model'):
                current = current.model
                depth += 1
            else:
                break

        # Verify we have a model with layers
        if not hasattr(current, 'layers'):
            raise AttributeError(f"Could not find model with 'layers' attribute after unwrapping {depth} levels. Final type: {type(current).__name__}")

        return current

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        target_texts: Optional[List[str]] = None,
        max_length: int = 64,
        source_texts: Optional[List[str]] = None,
        root_indices: Optional[List[int]] = None,
        output_hidden_states: bool = False,  # 🔥 NEW: Return hidden states for contrastive loss
        phrases: Optional[List[str]] = None,  # 🔥 UPDATED: Phrase strings for self-attention
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 OLD: [batch, num_papers] importance scores (not used)
        type_ids: Optional[torch.Tensor] = None,  # 🔥 NEW: [batch, num_vectors] type IDs (0=summary, 1=title)
        **generation_kwargs  # 🔥 NEW: Capture generation parameters
    ) -> Tuple[Optional[torch.Tensor], Optional[List[str]], Optional[List[torch.Tensor]]]:
        """
        Forward pass through decoder.

        Args:
            encoder_hidden_states: [batch, num_nodes, encoder_dim] - encoder outputs
            encoder_attention_mask: [batch, num_nodes] - validity mask
            root_indices: List of root paper positions for diagnostic logging (optional)
            target_texts: Target strings for training (None for inference)
            max_length: Maximum generation length
            output_hidden_states: If True, return hidden states for contrastive loss
            phrases: Optional list of phrase strings to guide generation (via self-attention)
            type_ids: [batch, num_vectors] type IDs (0=summary, 1=title) for mixed-dimension handling

        Returns:
            (loss, generated_texts, hidden_states) - hidden_states only if output_hidden_states=True
        """

        # 🔥 REMOVED: encoder_proj can't handle mixed dimensions (1152 summaries, 1024 titles)
        # CrossAttentionLayer now handles projection directly with k_proj_summary/k_proj_title
        # Just pass encoder_hidden_states directly (no projection here)
        encoder_states = encoder_hidden_states

        # 🔍 DEBUG: Register hook to monitor gradients flowing back from decoder to encoder

        if encoder_attention_mask is not None:
            pass
        if target_texts is not None:
            # Training mode
            result = self._training_forward(
                encoder_states=encoder_states,
                encoder_mask=encoder_attention_mask,
                target_texts=target_texts,
                max_length=max_length,
                type_ids=type_ids,
                source_texts=source_texts,
                root_indices=root_indices,
                output_hidden_states=output_hidden_states,  # 🔥 NEW
                phrases=phrases,  # 🔥 UPDATED: Pass phrases for self-attention
                citation_importance_scores=citation_importance_scores  # 🔥 NEW: Pass citation importance
            )
            # _training_forward returns (loss, None) or (loss, None, hidden_states)
            if output_hidden_states:
                return result  # (loss, None, hidden_states)
            else:
                return result + (None,)  # (loss, None, None)
        else:
            # Inference mode - no hidden states needed
            loss, generated_texts, _ = self._inference_forward(
                encoder_states=encoder_states,
                encoder_mask=encoder_attention_mask,
                max_new_tokens=max_length,
                citation_importance_scores=citation_importance_scores,  # 🔥 NEW: Pass citation importance
                phrases=phrases,  # 🔥 UPDATED: Pass phrases (not source_texts)
                root_indices=root_indices,
                type_ids=type_ids,
                **generation_kwargs  # 🔥 NEW: Pass generation parameters
            )
            return loss, generated_texts, None

    def _training_forward(
        self,
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        target_texts: List[str],
        source_texts: List[str],
        max_length: int,
        type_ids: Optional[torch.Tensor] = None,  # 🔥 NEW: Type IDs for mixed-dimension handling
        root_indices: Optional[List[int]] = None,
        output_hidden_states: bool = False,  # 🔥 NEW
        phrases: Optional[List[str]] = None,  # 🔥 UPDATED: Phrase strings for self-attention
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 NEW: [batch, num_papers] importance scores
        **generation_kwargs  # 🔥 NEW: Catch unused params for compatibility
    ) -> Tuple[torch.Tensor, None, Optional[List[torch.Tensor]]]:
        """
        Training forward pass with Hybrid Text-Graph input.
        🔥 UPDATED: Now includes phrase in instruction prompt for self-attention guidance.

        Returns:
            (loss, None, hidden_states) if output_hidden_states=True
            (loss, None) otherwise
        """
        
        # 1. Format inputs: "[INST] Abstract: ... \n Title: [/INST] "
        # We assume target_texts are the Titles.
        # We prepend the instruction and abstract.
        if source_texts is None:
            source_texts = [""] * len(target_texts)

        # 🔥 FIX: Truncate abstracts to prevent title truncation
        # Reserve space for: instruction (~20 tokens) + title (~50 tokens) + safety margin (10)
        # Total reserved: ~80 tokens
        # If max_length=64, total budget is 320 tokens, so abstract gets 320-80=240 tokens
        max_abstract_tokens = (max_length + 256) - 80  # Typically 240 tokens for abstract
        flag = False # Whether to include abstract in prompt
        truncated_abstracts = []
        for abstract in source_texts:
            if len(abstract) > 0 and flag:
                # Tokenize and truncate abstract
                tokens = self.tokenizer(
                    abstract,
                    max_length=max_abstract_tokens,
                    truncation=True,
                    add_special_tokens=False  # Don't add BOS/EOS yet
                )
                # Decode back to text
                truncated = self.tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
                truncated_abstracts.append(truncated)
            else:
                truncated_abstracts.append("")

        # 🔥 UPDATED: Use phrases in prompts for self-attention guidance
        if phrases and len(phrases) == len(truncated_abstracts):
            # Phrase-guided prompt: instruction-following pattern
            prompts = [
                f"[INST] {phrase}\nUsing the provided paper context, predict the paper's main mathematical result. [/INST] "
                for phrase in phrases
            ]
        else:
            # Fallback: no phrase guidance
            prompts = [
                f"[INST] Using the provided papers and theorems, write a concise mathematical research abstract describing the main idea and key results. [/INST] "
                for _ in truncated_abstracts
            ]

        # 2. Tokenize Prompts first (to know where to mask)
        # We need to know exactly how long the prompt is so we can set labels to -100
        prompt_tokens = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            add_special_tokens=True 
        ).to(self.device)
        
        # Calculate length of each prompt (sum of attention mask)
        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)

        # 3. Tokenize Full Input (Prompt + Target Title)
        full_texts = [p + t for p, t in zip(prompts, target_texts)]
        
        # Use max_length + 256 to accommodate the abstract text + title
        inputs = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length + 256
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        # Standard attention mask from tokenizer (1 for content, 0 for padding)
        # Note: We don't use encoder_mask here because that's for the graph cross-attention
        
        # 4. Create Labels with Masking
        labels = input_ids.clone()
        
        # A. Mask out the prompt tokens (set to -100 to ignore in loss)
        # We iterate over the batch and mask indices 0 to prompt_length for each sample
        for i, prompt_len in enumerate(prompt_lengths):
            # The label mask stops exactly where the prompt ends
            # We use min() to ensure we don't crash if truncation cut off the prompt
            safe_len = min(prompt_len, input_ids.size(1))
            labels[i, :safe_len] = -100
            
        # B. Mask padding tokens as well
        labels[inputs["attention_mask"] == 0] = -100

        # 5. Forward Pass
        # The model reads the Abstract text AND attends to the Graph vectors
        # It computes CE loss ONLY on the Title tokens (where labels != -100)
        result = self._forward_with_cross_attention(
            input_ids=input_ids,
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            labels=labels,
            type_ids=type_ids,  # 🔥 NEW: Pass type IDs for mixed-dimension handling
            root_indices=root_indices,
            output_hidden_states=output_hidden_states,  # 🔥 NEW
            prompt_lengths=prompt_lengths,  # 🔥 NEW: Pass for contrastive target masking
            citation_importance_scores=citation_importance_scores,  # 🔥 NEW: Pass citation importance
            **generation_kwargs # Pass remaining kwargs to catch them
        )

        # Unpack result
        if output_hidden_states:
            loss, last4_hidden_states, prompt_lengths_out = result
            return loss, None, (last4_hidden_states, prompt_lengths_out)
        else:
            loss = result
            return loss, None

    def _inference_forward(
        self,
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        phrases: Optional[List[str]],  # 🔥 UPDATED: phrases instead of source_texts
        max_new_tokens: int = 45,  # 🔥 FIXED: Match training targets (30-40 tokens for novelty statements)
        root_indices: Optional[List[int]] = None,
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 NEW: [batch, num_papers] importance scores
        type_ids =None,
        **generation_kwargs  # 🔥 NEW: Capture generation parameters
    ) -> Tuple[None, List[str]]:

        batch_size = encoder_states.size(0)

        # 1. Prepare Prompt Inputs with phrases (SAME AS TRAINING!)
        if phrases is not None and len(phrases) == batch_size:
            # 🔥 UPDATED: Phrase-guided prompt (matches training)
            prompts = [
                f"[INST] {phrase}\nUsing the provided paper context, predict the paper's main mathematical result. [/INST] "
                for phrase in phrases
            ]
            # 🔍 DEBUG: Print first prompt

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs["input_ids"]
        else:
            # Fallback: no phrase guidance
            print(f"\n⚠️ WARNING - No phrases! Using fallback prompt")
            print(f"  phrases: {phrases}")
            print(f"  batch_size: {batch_size}")
            prompts = [f"[INST] Using the provided papers and theorems, write a concise mathematical research abstract describing the main idea and key results. [/INST] "] * batch_size
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs["input_ids"]

        # 2. Generate
        # Returns List[str] because we updated _generate_with_cross_attention
        generated_texts = self._generate_with_cross_attention(
            input_ids=input_ids,
            encoder_states=encoder_states,
            encoder_mask=encoder_mask,
            max_new_tokens=max_new_tokens,
            # Merge defaults with generation_kwargs (kwargs take precedence)
            **{
                'do_sample': False,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.5,
                'length_penalty': 1.0,
                'no_repeat_ngram_size': 3,
                **generation_kwargs
            },
            citation_importance_scores=citation_importance_scores, # 🔥 NEW: Pass citation importance
            type_ids=type_ids,
        )

        # 3. Return directly (No need to decode again!)
        # 🔥 FIX: Return 3 values to match GraphToTextModel.forward expectations
        return None, generated_texts, None

    def _forward_with_cross_attention(
        self,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        root_indices: Optional[List[int]] = None,
        type_ids=None,  # <--- ADD THIS ARGUMENT
        output_hidden_states: bool = False,  # 🔥 NEW
        prompt_lengths: Optional[torch.Tensor] = None,  # 🔥 NEW: [B] prompt token counts for contrastive masking
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 NEW: [num_papers] importance scores
        **generation_kwargs # 🔥 NEW: Catch unused kwargs in training forward
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward through Mistral with cross-attention injection.

        This is where the decoder learns from the encoder.

        Returns:
            loss if not output_hidden_states
            (loss, last4_hidden_states, prompt_lengths) if output_hidden_states
              last4_hidden_states: List of 4 tensors [B, seq, 4096] from layers 28-31
              prompt_lengths: [B] int tensor of prompt token counts
        """
        base_model = self._get_base_model()
        from torch.utils.checkpoint import checkpoint as grad_checkpoint

        # Get embeddings
        hidden_states = base_model.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        num_layers = len(base_model.layers)
        last4_hidden_states = []  # collect last 4 layers if output_hidden_states
        use_ckpt = self.training  # gradient checkpointing only during training
         # Forward through layers
        for layer_idx, layer in enumerate(base_model.layers):
            # Standard Mistral layer
            # More specific: use the input_layernorm’s parameter device
            # Robust device checking that handles LoRA/PEFT/Quantization
            try:
                if hasattr(layer, "input_layernorm"):
                    if hasattr(layer.input_layernorm, "weight"):
                        norm_device = layer.input_layernorm.weight.device
                    else:
                        norm_device = next(layer.input_layernorm.parameters()).device
                else:
                    # Fallback to general layer parameters
                    norm_device = next(layer.parameters()).device
            except StopIteration:
                norm_device = hidden_states.device

            hidden_states = hidden_states.to(norm_device)
            encoder_states = encoder_states.to(norm_device)
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(norm_device)
            if type_ids is not None:
                type_ids = type_ids.to(norm_device)

            if use_ckpt:
                def _layer_fn(h, _layer=layer, _pos=position_ids):
                    out = _layer(h, position_ids=_pos)
                    return out[0] if isinstance(out, tuple) else out
                hidden_states = grad_checkpoint(_layer_fn, hidden_states, use_reentrant=False)
            else:
                layer_outputs = layer(hidden_states, position_ids=position_ids)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    
            # Inject cross-attention
            if str(layer_idx) in self.cross_attn_layers:

                normed_states = self.cross_attn_norms[str(layer_idx)](hidden_states.to(torch.float32)).to(torch.float32)
                # 🔥 NEW: Extract type_ids for current batch sample
                current_type_ids = type_ids[0] if type_ids is not None and type_ids.size(0) > 0 else None
                cross_attn_output, attn_weights = self.cross_attn_layers[str(layer_idx)](
                    hidden_states=normed_states,
                    encoder_hidden_states=encoder_states,
                    attention_mask=encoder_mask,
                    return_attention_weights=False,  # attention weights not used in dual encoder setup
                    citation_importance_scores=citation_importance_scores,  # 🔥 OLD: Dynamic bias from citation contexts (not used)
                    type_ids=current_type_ids  # 🔥 NEW: Type IDs for mixed-dimension handling
                )  # Unpack tuple (output, attention_weights)

                # # 🔍 ANALYZE: Which nodes does the model prefer (summary vs title)?
                # # NOTE: This diagnostic assumes first half=summaries, second half=titles.
                # # In the dual encoder setup all vectors are type_id=0 (no title split), so this is WRONG.

                hidden_states_f32 = hidden_states.to(torch.float32)
                cross_attn_output_f32 = cross_attn_output.to(torch.float32)

                # 2. Add in high precision
                hidden_states_f32 = hidden_states_f32 + cross_attn_output_f32

                # 3. Cast back to original dtype (FP16)
                hidden_states = hidden_states_f32.to(hidden_states.dtype)

                # 🔬 DIAGNOSTIC: Measure cross-attention contribution vs residual

            # Collect last 4 pre-norm hidden states for contrastive loss
            if output_hidden_states and layer_idx >= num_layers - 4:
                last4_hidden_states.append(hidden_states)

        # Final norm and LM head
        hidden_states = base_model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 🔍 DIAGNOSTIC: Check for all-masked labels (causes NaN in CrossEntropyLoss)
        valid_labels_mask = (shift_labels.view(-1) != -100)
        num_valid = valid_labels_mask.sum().item()
        num_total = shift_labels.numel()

        if num_valid == 0:
            print(f"\n🚨 NaN SOURCE DETECTED: All labels are masked!")
            print(f"   Batch size: {input_ids.shape[0]}")
            print(f"   Sequence length: {input_ids.shape[1]}")
            print(f"   Valid labels: {num_valid}/{num_total}")
            print(f"   ⚠️  This means titles were completely truncated.")
            print(f"   Returning zero loss to prevent NaN.")
            # Return zero loss with gradient enabled to avoid breaking backward pass
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        # Log warning if very few valid labels (but not zero)
        if num_valid < num_total * 0.1:  # Less than 10% valid
            print(f"   ⚠️  WARNING: Only {num_valid}/{num_total} ({100*num_valid/num_total:.1f}%) labels are valid.")
            print(f"   Title may be partially truncated.")

        loss_fct = nn.CrossEntropyLoss()
        # 🔥 FIX: Cast logits to float32 before loss calculation; move labels to match logits device
        ce_loss = loss_fct(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.to(shift_logits.device).view(-1)
        )

        # 🔥 NEW: Add length penalty to encourage concise titles
        # Typical academic titles: 5-15 words (target ~10 words)
        # Count non-padding tokens (actual length)
        non_pad_mask = (labels != self.tokenizer.pad_token_id)
        actual_lengths = non_pad_mask.sum(dim=1).float()  # [batch_size]

        target_length = 300 # Target ~300 tokens (full abstract + key results)
        # Only penalize if TOO LONG (not too short) - asymmetric penalty
        length_overage = torch.clamp(actual_lengths - target_length, min=0)
        length_penalty = (length_overage / target_length).mean()

        # Combined loss: cross-entropy + small length penalty (only fires if >300 tokens)
        loss = ce_loss + 0.5 * length_penalty.to(ce_loss.device)

        # 🔥 NEW: Return hidden states if requested (for contrastive loss)
        if output_hidden_states:
            # last4_hidden_states: list of 4 tensors [B, seq, 4096] from layers 28-31 (pre-norm)
            # prompt_lengths: [B] int tensor so caller can mask to target tokens only
            return loss, last4_hidden_states, prompt_lengths
        else:
            return loss
    
    def _generate_with_cross_attention(
        self,
        input_ids: torch.Tensor,      # <--- CHANGED: Accepts the prompt (Abstract)
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_new_tokens: int = 32,
        temperature: float = 0.6,
        top_p: float = 0.92,
        top_k: int = 50,
        type_ids=None,  # <--- ADD THIS ARGUMENT

        repetition_penalty: float = 1.3,
        length_penalty: float = 1.2,  # 🔥 FIXED: 0.3 → 1.2 to penalize long outputs (> 1.0 = shorter)
        no_repeat_ngram_size: int = 3,
        
        citation_importance_scores: Optional[torch.Tensor] = None,  # 🔥 NEW: [num_papers] importance scores
        do_sample: bool = False,  # greedy by default
        **generation_kwargs  # 🔥 NEW: Capture other args
    ) -> List[str]:  # <--- CHANGED: Returns list of strings
        """
        Autoregressive generation starting from input_ids (the prompt).
        """
        base_model = self._get_base_model()
        self.model.eval()

        # 1. Initialize with the prompt instead of just BOS
        generated = input_ids.clone()
        prompt_length = generated.shape[1]
        batch_size = generated.shape[0]

        # N-gram helper (same as your code)
        def _get_ngram_blocked_tokens(generated_seq, ngram_size):
            if generated_seq.size(0) < ngram_size: return set()
            blocked = set()
            gen_list = generated_seq.tolist()
            for i in range(len(gen_list) - ngram_size + 1):
                ngram = tuple(gen_list[i:i + ngram_size - 1])
                for j in range(i):
                    if j + ngram_size - 1 < len(gen_list):
                        prev_ngram = tuple(gen_list[j:j + ngram_size - 1])
                        if prev_ngram == ngram: blocked.add(gen_list[j + ngram_size - 1])
            return blocked

        # Per-sequence EOS tracking
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 2. Generation Loop
        for step in range(max_new_tokens):
            if finished_sequences.all():
                break

            hidden_states = base_model.embed_tokens(generated)
            bsz, seq_len = generated.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)

            for layer_idx, layer in enumerate(base_model.layers):
                try:
                    layer_device = next(layer.parameters()).device
                    hidden_states = hidden_states.to(layer_device)
                    position_ids = position_ids.to(layer_device)
                    encoder_states = encoder_states.to(layer_device)
                    if encoder_mask is not None:
                        encoder_mask = encoder_mask.to(layer_device)
                    if type_ids is not None:
                        type_ids = type_ids.to(layer_device)
                except StopIteration:
                    pass

                # Match training exactly: no attention_mask, let HF handle causal masking internally
                layer_outputs = layer(hidden_states, position_ids=position_ids)
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

                if str(layer_idx) in self.cross_attn_layers:
                    normed_states = self.cross_attn_norms[str(layer_idx)](hidden_states)
                    # 🔥 NEW: Extract type_ids for current batch sample
                    current_type_ids = type_ids[0] if type_ids is not None and type_ids.size(0) > 0 else None
                    cross_attn_output, _ = self.cross_attn_layers[str(layer_idx)](
                        hidden_states=normed_states,
                        encoder_hidden_states=encoder_states,
                        attention_mask=encoder_mask,
                        return_attention_weights=False,  # Don't need weights during generation
                        citation_importance_scores=citation_importance_scores,  # 🔥 OLD: Dynamic bias from citation contexts (not used)
                        type_ids=current_type_ids  # 🔥 NEW: Type IDs for mixed-dimension handling
                    )
                    cross_attn_output = cross_attn_output.to(hidden_states.dtype)
                    cross_attn_output = torch.clamp(cross_attn_output, -65504, 65504)
                    hidden_states = hidden_states + cross_attn_output

            hidden_states = base_model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]

            # --- Sampling Logic ---
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            if no_repeat_ngram_size > 0:
                for i in range(batch_size):
                    # Only block based on new tokens, ignoring the prompt
                    gen_part = generated[i, prompt_length:]
                    if len(gen_part) > 0:
                        blocked_tokens = _get_ngram_blocked_tokens(gen_part, no_repeat_ngram_size)
                        for token_id in blocked_tokens:
                            next_token_logits[i, token_id] = float('-inf')

            if length_penalty < 1.0 and step > 5:
                progress_ratio = step / max_new_tokens
                eos_boost = progress_ratio * (1.0 - length_penalty) * 5.0
                next_token_logits[:, self.tokenizer.eos_token_id] += eos_boost

            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                top_k_logits, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(next_token_logits < min_top_k, torch.full_like(next_token_logits, float('-inf')), next_token_logits)

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 🔥 NEW: Mask tokens for finished sequences (use PAD instead of continuing)
            # 🔥 MULTI-GPU FIX: Move next_token back to self.device for bookkeeping
            # (next_token is on last GPU after lm_head; generated/finished_sequences live on self.device)
            next_token = next_token.to(self.device)
            next_token = torch.where(
                finished_sequences.unsqueeze(1),
                torch.full_like(next_token, self.tokenizer.pad_token_id),
                next_token
            )

            generated = torch.cat([generated, next_token], dim=1)

            # 🔥 NEW: Update finished status (mark sequences that just generated EOS)
            finished_sequences = finished_sequences | (next_token.squeeze(1) == self.tokenizer.eos_token_id)

            # Stop when ALL sequences are done
            if finished_sequences.all():
                break

        # 3. Decode and return text (stripping the prompt)
        generated_ids = generated[:, prompt_length:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 🔥 MULTI-GPU FIX: Ensure decoded is a proper list of strings
        if not isinstance(decoded, list):
            decoded = list(decoded)

        # Double-check each element is a string (not map object)
        result = []
        for idx, item in enumerate(decoded):
            if isinstance(item, str):
                result.append(item)
            else:
                # 🔥 DEBUG: Print what type we got
                print(f"\n⚠️ WARNING: decoded[{idx}] is not a string!")
                print(f"  Type: {type(item)}")
                print(f"  Item repr: {repr(item)}")

                # If item is iterator/map, it might be map(str, tokens) or similar
                # Try to materialize it: if it's iterable, join it
                try:
                    if hasattr(item, '__iter__'):
                        materialized = list(item)
                        print(f"  Materialized: {materialized[:50]}")  # First 50 elements

                        # If list of strings, join them
                        if materialized and isinstance(materialized[0], str):
                            text = ''.join(materialized)
                            print(f"  Joined text: {text}")
                            result.append(text)
                        else:
                            result.append(str(materialized))
                    else:
                        result.append(str(item))
                except Exception as e:
                    print(f"⚠️ Error materializing: {e}")
                    import traceback
                    traceback.print_exc()
                    result.append(f"[DECODE_ERROR: {type(item)}]")

        return result    # def _generate_with_cross_attention(

