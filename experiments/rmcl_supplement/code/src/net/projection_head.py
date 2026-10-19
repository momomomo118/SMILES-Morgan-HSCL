"""Projection heads for supervised contrastive learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head.

    The encoder pooled representation is kept for RT prediction, while this
    projected representation is used only for contrastive alignment. The output
    is L2-normalized by default because contrastive losses usually compare
    cosine/dot-product similarities.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        projected = self.net(pooled)
        if self.normalize:
            projected = F.normalize(projected, dim=-1)
        return projected


class IdentityProjectionHead(nn.Module):
    """A no-op projection head for ablations."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return F.normalize(pooled, dim=-1)
        return pooled
