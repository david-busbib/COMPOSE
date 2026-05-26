"""
Multi-Level Cross-Attention Layers for Scientific Paper Prediction

This module implements three strategic cross-attention layers:
1. GraphAwareCrossAttention: Injects citation graph structure into node features
2. HierarchicalCrossAttentionPooling: Better subgraph-level aggregation

Date: 2025
"""

import torch
import torch.nn as nn
import math


class GraphAwareCrossAttention(nn.Module):
    """
    Level 1 Cross-Attention: Graph-Aware Text Encoding

    Lets each node attend to its citation neighbors to inject graph structure
    into text features BEFORE UniGLM encoding.

    Key innovation: Uses edge_index to restrict attention to graph neighbors
    (not full N×N attention), making it efficient O(E × H × D).
    """

    def __init__(self, hidden_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge-aware bias (optional, for future edge features)
        self.edge_bias = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, hidden_dim] node text features
            edge_index: [2, E] citation edges (src, dst) where src cites dst
            edge_attr: [E, d] optional edge features (e.g., citation count)

        Returns:
            x_out: [N, hidden_dim] graph-aware features with residual connection
        """
        N = x.size(0)

        # Handle empty graph case
        if edge_index.size(1) == 0:
            return x

        # Project to Q, K, V
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # Initialize output
        output = torch.zeros_like(Q)

        # Extract source and destination nodes
        src, dst = edge_index[0], edge_index[1]

        # Gather K and V from source nodes (papers being cited)
        K_neighbors = K[src]  # [E, H, D]
        V_neighbors = V[src]  # [E, H, D]
        Q_targets = Q[dst]    # [E, H, D]

        # Compute attention scores: Q · K^T / sqrt(d)
        attn = (Q_targets * K_neighbors).sum(dim=-1) * self.scale  # [E, H]

        # Optional: Add edge-specific bias
        if not torch.is_tensor(edge_attr):
            edge_attr = torch.tensor(edge_attr, dtype=attn.dtype, device=attn.device)
        if edge_attr.dim() == 0:
            edge_attr = edge_attr.unsqueeze(0)  # scalar → [1]
        edge_bias = self.edge_bias * edge_attr.unsqueeze(-1)
        attn = attn + edge_bias

        # Softmax over incoming edges for each target node
        attn = self._edge_softmax(attn, dst, N)  # [E, H]
        attn = self.dropout(attn)

        # Weighted sum of values
        attn_expanded = attn.unsqueeze(-1)  # [E, H, 1]
        messages = attn_expanded * V_neighbors  # [E, H, D]

        # Aggregate messages to target nodes
        output.index_add_(0, dst, messages)

        # Reshape and project
        output = output.view(N, self.hidden_dim)
        output = self.out_proj(output)

        # Residual connection + layer norm
        return self.norm(x + output)

    def _edge_softmax(self, attn, dst, num_nodes):
        """
        Compute softmax over incoming edges for each node (numerically stable).

        Args:
            attn: [E, H] attention scores
            dst: [E] destination node indices
            num_nodes: total number of nodes

        Returns:
            attn_normalized: [E, H] normalized attention weights
        """
        # Find max for numerical stability
        max_vals = torch.full((num_nodes, attn.size(1)), float('-inf'),
                             device=attn.device, dtype=attn.dtype)
        max_vals.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(attn),
                                attn, reduce='amax', include_self=False)

        # Handle nodes with no incoming edges
        max_vals = torch.where(torch.isinf(max_vals), torch.zeros_like(max_vals), max_vals)

        # Subtract max and exp
        attn = attn - max_vals[dst]
        attn_exp = torch.exp(torch.clamp(attn, min=-20, max=20))

        # Sum over incoming edges
        attn_sum = torch.zeros(num_nodes, attn.size(1),
                              device=attn.device, dtype=attn.dtype)
        attn_sum.index_add_(0, dst, attn_exp)

        # Normalize (add epsilon for nodes with no edges)
        return attn_exp / (attn_sum[dst] + 1e-8)


class HierarchicalCrossAttentionPooling(nn.Module):
    """
    Level 2 Cross-Attention: Hierarchical Subgraph Pooling

    Replaces simple attention pooling with two-level cross-attention:
    1. Root-aware: Root node (target paper) attends to all context papers
    2. Global: Learnable query attends to all nodes (captures overall theme)

    Combines both for richer subgraph representation.
    """

    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Root-aware attention
        self.root_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Global attention with learnable query
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion layer: combines root-aware + global contexts
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_embeds, root_idx=0, mask=None):
        """
        Args:
            node_embeds: [N, hidden_dim] or [B, N, hidden_dim] node embeddings from UniGLM
            root_idx: index of root node (target paper, default 0)
            mask: [N] or [B, N] boolean mask (True = valid node, False = masked/padding)

        Returns:
            subgraph_embed: [hidden_dim] or [B, hidden_dim] aggregated subgraph embedding
        """
        # Handle both batched and unbatched input
        is_batched = len(node_embeds.shape) == 3
        if not is_batched:
            node_embeds = node_embeds.unsqueeze(0)  # [1, N, D]
            if mask is not None:
                mask = mask.unsqueeze(0)  # [1, N]

        B, N, D = node_embeds.shape

        # Create key_padding_mask for MultiheadAttention
        # (True = ignore this position, False = attend to this position)
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = ignore
        else:
            key_padding_mask = None

        # 1. Root-aware attention: Root attends to all nodes
        # 🔥 CRITICAL: If root_idx is None (masked node), skip root-aware attention!
        if root_idx is not None:
            root_embed = node_embeds[:, root_idx:root_idx+1, :]  # [B, 1, D]
            root_context, root_attn_weights = self.root_attn(
                query=root_embed,        # [B, 1, D]
                key=node_embeds,         # [B, N, D]
                value=node_embeds,       # [B, N, D]
                key_padding_mask=key_padding_mask,
                need_weights=True
            )  # [B, 1, D]
        else:
            # If root is masked, use zeros for root_context (no information leakage!)
            root_context = torch.zeros(B, 1, D, device=node_embeds.device)

        # 2. Global attention: Learnable query attends to all nodes
        global_query = self.global_query.expand(B, -1, -1)  # [B, 1, D]
        global_context, global_attn_weights = self.global_attn(
            query=global_query,      # [B, 1, D]
            key=node_embeds,         # [B, N, D]
            value=node_embeds,       # [B, N, D]
            key_padding_mask=key_padding_mask,
            need_weights=True
        )  # [B, 1, D]

        # 3. Fuse root-aware and global contexts
        combined = torch.cat([root_context, global_context], dim=-1)  # [B, 1, 2D]
        subgraph_embed = self.fusion(combined).squeeze(1)  # [B, D]

        # Remove batch dimension if input wasn't batched
        if not is_batched:
            subgraph_embed = subgraph_embed.squeeze(0)  # [D]

        return subgraph_embed




# Helper function for testing
