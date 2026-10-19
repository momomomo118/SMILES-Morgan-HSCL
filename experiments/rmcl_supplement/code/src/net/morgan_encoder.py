"""Morgan fingerprint encoder for the multimodal HSCL model.

The baseline Morgan model predicted RT directly from graph/fingerprint features.
For multimodal HSCL, this module keeps the fingerprint branch as an encoder:

2048-bit Morgan vector -> token sequence -> Transformer encoder -> [CLS] vector

No RT regression head is included here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from .smiles_encoder import EncoderOutput
    from .projection_head import ProjectionHead
except ImportError:  # pragma: no cover - supports direct script imports
    from smiles_encoder import EncoderOutput
    from projection_head import ProjectionHead


@dataclass
class MorganEncoderConfig:
    n_bits: int = 2048
    chunk_size: int = 16
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.1


class MorganFingerprintEncoder(nn.Module):
    """Transformer encoder over fixed-length Morgan fingerprint chunks."""

    def __init__(self, config: MorganEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or MorganEncoderConfig()
        if self.config.n_bits % self.config.chunk_size != 0:
            raise ValueError("n_bits must be divisible by chunk_size.")

        self.num_tokens = self.config.n_bits // self.config.chunk_size
        self.input_proj = nn.Linear(self.config.chunk_size, self.config.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_tokens + 1, self.config.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)
        self.output_dim = self.config.hidden_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, morgan_fp: torch.Tensor, **kwargs) -> EncoderOutput:
        if morgan_fp.dim() != 2 or morgan_fp.size(1) != self.config.n_bits:
            raise ValueError(
                f"Expected morgan_fp shape [batch, {self.config.n_bits}], got {tuple(morgan_fp.shape)}"
            )

        batch_size = morgan_fp.size(0)
        tokens = morgan_fp.view(batch_size, self.num_tokens, self.config.chunk_size)
        tokens = self.input_proj(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.size(1), :]
        sequence = self.encoder(sequence)
        sequence = self.layer_norm(sequence)
        pooled = sequence[:, 0, :]
        mask = torch.ones(sequence.size()[:2], dtype=torch.long, device=sequence.device)
        return EncoderOutput(sequence=sequence, pooled=pooled, mask=mask)


class MorganProjectionHead(nn.Module):
    """Projection head for supervised contrastive learning."""

    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = ProjectionHead(input_dim=input_dim, projection_dim=projection_dim, dropout=dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class MorganMLPEncoder(nn.Module):
    """Lightweight fallback encoder for ablation experiments.

    It is useful for quick checks or for comparing Transformer encoding against
    a plain feed-forward fingerprint representation.
    """

    def __init__(self, n_bits: int = 2048, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(n_bits, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.GELU(),
        )

    def forward(self, morgan_fp: torch.Tensor, **kwargs) -> EncoderOutput:
        pooled = self.net(morgan_fp)
        sequence = pooled.unsqueeze(1)
        mask = torch.ones(sequence.size()[:2], dtype=torch.long, device=sequence.device)
        return EncoderOutput(sequence=sequence, pooled=pooled, mask=mask)
