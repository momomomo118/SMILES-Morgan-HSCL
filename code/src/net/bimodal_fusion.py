"""Bimodal fusion layers for SMILES-Morgan HSCL.

This implements the paper-style fusion block:

1. Self-Attention over the target modality sequence.
2. Cross-Attention from context modality to target modality.
3. Position-wise Feed-Forward network.

For this project, SMILES is treated as the target/query modality and Morgan
fingerprint tokens are treated as the context/key-value modality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FusionOutput:
    sequence: torch.Tensor
    pooled: torch.Tensor
    self_attention_weights: Optional[torch.Tensor] = None
    cross_attention_weights: Optional[torch.Tensor] = None


def _to_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert 1/0 valid mask to PyTorch key_padding_mask.

    nn.MultiheadAttention expects True at positions that should be ignored.
    """

    if mask is None:
        return None
    return mask == 0


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.dropout(update))


class PositionwiseFeedForward(nn.Module):
    """Transformer-style feed-forward sublayer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BimodalCrossAttentionFusion(nn.Module):
    """Self-Attention + Cross-Attention + FFN bimodal fusion block."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        fusion_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "first_token",
    ) -> None:
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError("fusion_dim must be divisible by num_heads.")

        self.pooling = pooling
        self.query_proj = nn.Identity() if query_dim == fusion_dim else nn.Linear(query_dim, fusion_dim)
        self.context_proj = nn.Identity() if context_dim == fusion_dim else nn.Linear(context_dim, fusion_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_add_norm = AddNorm(fusion_dim, dropout=dropout)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_add_norm = AddNorm(fusion_dim, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            dim=fusion_dim,
            hidden_dim=feedforward_dim,
            dropout=dropout,
        )
        self.ffn_add_norm = AddNorm(fusion_dim, dropout=dropout)
        self.output_dim = fusion_dim

    def forward(
        self,
        query_sequence: torch.Tensor,
        context_sequence: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> FusionOutput:
        query = self.query_proj(query_sequence)
        context = self.context_proj(context_sequence)

        query_key_padding_mask = _to_key_padding_mask(query_mask)
        context_key_padding_mask = _to_key_padding_mask(context_mask)

        self_update, self_weights = self.self_attention(
            query=query,
            key=query,
            value=query,
            key_padding_mask=query_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        hidden = self.self_add_norm(query, self_update)

        cross_update, cross_weights = self.cross_attention(
            query=hidden,
            key=context,
            value=context,
            key_padding_mask=context_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        hidden = self.cross_add_norm(hidden, cross_update)

        ffn_update = self.feed_forward(hidden)
        fused_sequence = self.ffn_add_norm(hidden, ffn_update)
        fused_pooled = self._pool(fused_sequence, query_mask)

        return FusionOutput(
            sequence=fused_sequence,
            pooled=fused_pooled,
            self_attention_weights=self_weights if need_weights else None,
            cross_attention_weights=cross_weights if need_weights else None,
        )

    def _pool(self, sequence: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "first_token":
            return sequence[:, 0, :]
        if self.pooling == "mean":
            if mask is None:
                return sequence.mean(dim=1)
            mask = mask.unsqueeze(-1).to(sequence.dtype)
            return (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        raise ValueError(f"Unsupported pooling: {self.pooling}")
