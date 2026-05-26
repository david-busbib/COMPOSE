"""
Theorem dependency-graph encoder for COMPOSE (enc2).

Encodes a mathlib theorem subgraph (pre-loaded E5 node embeddings + edges)
into [N, 1152] vectors via a SimpleGNN + AttentionPool. Used as enc2 by
the COMPOSE dual-encoder pipeline in train_dual.py.

This file is the COMPOSE-arXiv slice of the original
`s2orc/LeanDojo/leandojo_benchmark_4/model_clean_theorem.py` — the
conjecture-generation decoder, dataset, trainer, and standalone loss
functions have been removed (saved ~3000 lines).
"""

import logging
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class DependencyMatrixHead(nn.Module):
    """
    Bilinear scoring head for dependency matrix prediction.
    score(i,j) = (h_i @ W1) @ (h_j @ W2)^T
    """
    def __init__(self, input_dim: int = 1152, hidden_dim: int = 256):
        super().__init__()
        self.proj_src = nn.Linear(input_dim, hidden_dim)
        self.proj_dst = nn.Linear(input_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, node_embeds: torch.Tensor) -> torch.Tensor:
        src = self.proj_src(node_embeds)  # [N, hidden_dim]
        dst = self.proj_dst(node_embeds)  # [N, hidden_dim]
        adj_logits = torch.matmul(src, dst.T) + self.bias  # [N, N]
        return adj_logits


class TheoremEncoder(nn.Module):
    """
    Encoder for theorem conjecture prediction.

    Uses PRE-LOADED embeddings from TheoremDataset instead of loading from file paths.
    This is much faster and simpler than the paper encoder.

    Architecture:
    1. Pre-loaded E5 embeddings (1024-dim) from numpy array
    2. GNN for graph structure encoding (adds 128-dim structural vectors)
    3. Output: [N, 1152] embeddings for cross-attention

    Note: For theorems, we only have one embedding per node (the statement embedding).
    We don't have separate summary/title like papers.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1024,
        struct_dim: int = 128,
        num_gnn_layers: int = 4,  # Changed from 2 → 4 for deeper graph encoding
        use_dependency_head: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.struct_dim = struct_dim
        self.use_dependency_head = use_dependency_head
        self.device = device

        # GNN for graph structure encoding
        self.gnn = SimpleGNN(
            hidden_dim=embed_dim,
            struct_dim=struct_dim,
            num_layers=num_gnn_layers
        ).to(device)

        # Projection to hidden_dim (optional, if embed_dim != hidden_dim)
        if embed_dim != hidden_dim:
            self.proj = nn.Linear(embed_dim, hidden_dim).to(device)
        else:
            self.proj = None

        # Attention pooling for graph-level representation
        self.graph_pool = AttentionPool(embed_dim + struct_dim).to(device)

        # Output projection to E5 space (for contrastive loss)
        self.output_proj = nn.Linear(embed_dim + struct_dim, 1024).to(device)

        # Dependency Matrix Head (Auxiliary Task)
        if use_dependency_head:
            self.dependency_head = DependencyMatrixHead(
                input_dim=embed_dim + struct_dim,
                hidden_dim=256
            ).to(device)
        else:
            self.dependency_head = None

        logger.info("\n" + "="*60)
        logger.info("THEOREM ENCODER")
        logger.info("="*60)
        logger.info(f"Input: Pre-loaded E5 embeddings [{embed_dim}]")
        logger.info(f"GNN: {num_gnn_layers} layers → structural vectors [{struct_dim}]")
        logger.info(f"Output: [{embed_dim + struct_dim}] (E5 + structure)")
        logger.info(f"Dependency Head: {'Enabled' if use_dependency_head else 'Disabled'}")
        logger.info("="*60 + "\n")

    def encode_subgraph(
        self,
        node_embeddings: torch.Tensor,
        edges: List[List[int]],
        target_uses: List[int] = None
    ) -> Dict:
        """
        Encode a theorem subgraph using pre-loaded embeddings.

        Args:
            node_embeddings: [N, 1024] pre-loaded E5 embeddings
            edges: [[i, j], ...] edge list (node indices)
            target_uses: List of indices the target directly uses (optional)

        Returns:
            Dict with:
                - 'cross_attn_embeds': [N, 1152] node embeddings with structure
                - 'type_ids': [N] type IDs (all 0 for theorems)
                - 'num_vectors': int
                - 'node_embeds': [N, 1024] original embeddings
                - 'graph_embed': [1152] pooled graph representation
                - 'predicted_embed': [1024] output for contrastive loss
        """
        node_embeddings = node_embeddings.to(self.device)
        num_nodes = node_embeddings.size(0)

        # Convert edges to tuple format for GNN
        edge_tuples = [(e[0], e[1]) for e in edges]


        # Run GNN to get structural vectors
        struct_vecs = self.gnn(node_embeddings, edge_tuples)  # [N, 128]

        # Concatenate E5 embeddings with structural vectors
        node_with_struct = torch.cat([node_embeddings, struct_vecs], dim=-1)  # [N, 1152]

        # Create mask for target_uses nodes (optional weighting)
        if target_uses is not None and len(target_uses) > 0:
            mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
        else:
            mask = None

        # Graph-level pooling
        graph_embed = self.graph_pool(node_with_struct, mask)  # [1, 1152]
        graph_embed = graph_embed.squeeze(0)  # [1152]

        # Project to E5 space for contrastive loss
        predicted_embed = self.output_proj(graph_embed)  # [1024]

        # Compute dependency logits if enabled
        adj_logits = None
        if self.use_dependency_head:
            adj_logits = self.dependency_head(node_with_struct)  # [N, N]

        # Type IDs (all 0 for theorems - single type)
        type_ids = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        return {
            'cross_attn_embeds': node_with_struct,  # [N, 1152]
            'type_ids': type_ids,                    # [N]
            'num_vectors': num_nodes,
            'node_embeds': node_embeddings,          # [N, 1024]
            'graph_embed': graph_embed,              # [1152]
            'predicted_embed': predicted_embed,      # [1024] for contrastive loss
            'edge_list': edge_tuples,
            'adj_logits': adj_logits,                # [N, N] or None
        }

    def forward(
        self,
        batch_node_embeddings: List[torch.Tensor],
        batch_edges: List[List[List[int]]],
        batch_target_uses: List[List[int]] = None
    ) -> Dict:
        """
        Encode a batch of theorem subgraphs.

        Args:
            batch_node_embeddings: List of [N_i, 1024] tensors
            batch_edges: List of edge lists [[[i,j], ...], ...]
            batch_target_uses: List of target_uses lists

        Returns:
            Dict with batched outputs
        """
        batch_size = len(batch_node_embeddings)

        results = {
            'cross_attn_embeds': [],
            'type_ids': [],
            'num_vectors': [],
            'node_embeds': [],
            'graph_embeds': [],
            'predicted_embeds': [],
            'adj_logits_list': [],
        }

        for i in range(batch_size):
            target_uses = batch_target_uses[i] if batch_target_uses else None

            encoded = self.encode_subgraph(
                node_embeddings=batch_node_embeddings[i],
                edges=batch_edges[i],
                target_uses=target_uses
            )

            results['cross_attn_embeds'].append(encoded['cross_attn_embeds'])
            results['type_ids'].append(encoded['type_ids'])
            results['num_vectors'].append(encoded['num_vectors'])
            results['node_embeds'].append(encoded['node_embeds'])
            results['graph_embeds'].append(encoded['graph_embed'])
            results['predicted_embeds'].append(encoded['predicted_embed'])
            results['adj_logits_list'].append(encoded['adj_logits'])

        # Stack graph-level outputs
        results['predicted_embeds'] = torch.stack(results['predicted_embeds'])  # [B, 1024]
        results['graph_embeds'] = torch.stack(results['graph_embeds'])  # [B, 1152]

        # NaN detection in encoder outputs
        if torch.isnan(results['predicted_embeds']).any() or torch.isinf(results['predicted_embeds']).any():
            logger.warning(f"🚨 NaN/Inf in ENCODER predicted_embeds! max={results['predicted_embeds'].abs().max().item():.4f}")
        if torch.isnan(results['graph_embeds']).any() or torch.isinf(results['graph_embeds']).any():
            logger.warning(f"🚨 NaN/Inf in ENCODER graph_embeds! max={results['graph_embeds'].abs().max().item():.4f}")

        return results

