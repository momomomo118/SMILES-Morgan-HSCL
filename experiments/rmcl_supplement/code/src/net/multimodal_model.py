"""Multimodal model skeleton built from SMILES and Morgan encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .bimodal_fusion import BimodalCrossAttentionFusion
from .morgan_encoder import MorganFingerprintEncoder, MorganProjectionHead
from .projection_head import ProjectionHead
from .smiles_encoder import SMILESEncoder, SMILESProjectionHead


@dataclass
class MultimodalForwardOutput:
    prediction: torch.Tensor
    smiles_pooled: torch.Tensor
    morgan_pooled: torch.Tensor
    fusion_pooled: Optional[torch.Tensor] = None
    z_smiles: Optional[torch.Tensor] = None
    z_morgan: Optional[torch.Tensor] = None
    z_fusion: Optional[torch.Tensor] = None


class ConcatRegressionHead(nn.Module):
    """Simple fusion head used as the first runnable multimodal baseline."""

    def __init__(self, smiles_dim: int, morgan_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(smiles_dim + morgan_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, smiles_pooled: torch.Tensor, morgan_pooled: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([smiles_pooled, morgan_pooled], dim=-1)
        return self.net(fused).squeeze(-1)


class SMILESMorganConcatModel(nn.Module):
    """First multimodal checkpoint: two encoders + projection heads + concat RT head."""

    def __init__(
        self,
        smiles_encoder: SMILESEncoder,
        morgan_encoder: MorganFingerprintEncoder,
        projection_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.morgan_encoder = morgan_encoder
        self.smiles_projection = SMILESProjectionHead(
            input_dim=smiles_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.morgan_projection = MorganProjectionHead(
            input_dim=morgan_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.regression_head = ConcatRegressionHead(
            smiles_dim=smiles_encoder.output_dim,
            morgan_dim=morgan_encoder.output_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morgan_fp: torch.Tensor,
    ) -> MultimodalForwardOutput:
        smiles_out = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask)
        morgan_out = self.morgan_encoder(morgan_fp=morgan_fp)
        prediction = self.regression_head(smiles_out.pooled, morgan_out.pooled)
        return MultimodalForwardOutput(
            prediction=prediction,
            smiles_pooled=smiles_out.pooled,
            morgan_pooled=morgan_out.pooled,
            z_smiles=self.smiles_projection(smiles_out.pooled),
            z_morgan=self.morgan_projection(morgan_out.pooled),
        )


class SMILESMorganCrossAttentionModel(nn.Module):
    """Paper-style bimodal fusion model.

    SMILES is the target/query modality. Morgan fingerprint tokens are the
    context/key-value modality.
    """

    def __init__(
        self,
        smiles_encoder: SMILESEncoder,
        morgan_encoder: MorganFingerprintEncoder,
        projection_dim: int = 256,
        fusion_dim: int = 256,
        fusion_heads: int = 8,
        fusion_feedforward_dim: int = 512,
        regression_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.morgan_encoder = morgan_encoder
        self.smiles_projection = SMILESProjectionHead(
            input_dim=smiles_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.morgan_projection = MorganProjectionHead(
            input_dim=morgan_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.fusion = BimodalCrossAttentionFusion(
            query_dim=smiles_encoder.output_dim,
            context_dim=morgan_encoder.output_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            feedforward_dim=fusion_feedforward_dim,
            dropout=dropout,
            pooling="first_token",
        )
        self.fusion_projection = ProjectionHead(
            input_dim=self.fusion.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.fusion.output_dim, regression_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, regression_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morgan_fp: torch.Tensor,
        need_attention_weights: bool = False,
    ) -> MultimodalForwardOutput:
        smiles_out = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask)
        morgan_out = self.morgan_encoder(morgan_fp=morgan_fp)
        fusion_out = self.fusion(
            query_sequence=smiles_out.sequence,
            context_sequence=morgan_out.sequence,
            query_mask=smiles_out.mask,
            context_mask=morgan_out.mask,
            need_weights=need_attention_weights,
        )
        prediction = self.regression_head(fusion_out.pooled).squeeze(-1)
        return MultimodalForwardOutput(
            prediction=prediction,
            smiles_pooled=smiles_out.pooled,
            morgan_pooled=morgan_out.pooled,
            fusion_pooled=fusion_out.pooled,
            z_smiles=self.smiles_projection(smiles_out.pooled),
            z_morgan=self.morgan_projection(morgan_out.pooled),
            z_fusion=self.fusion_projection(fusion_out.pooled),
        )


class SMILESMorganCrossAttentionResidualModel(nn.Module):
    """Cross-attention model with direct unimodal residual features."""

    def __init__(
        self,
        smiles_encoder: SMILESEncoder,
        morgan_encoder: MorganFingerprintEncoder,
        projection_dim: int = 256,
        fusion_dim: int = 256,
        fusion_heads: int = 8,
        fusion_feedforward_dim: int = 512,
        regression_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.morgan_encoder = morgan_encoder
        self.smiles_projection = SMILESProjectionHead(
            input_dim=smiles_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.morgan_projection = MorganProjectionHead(
            input_dim=morgan_encoder.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.fusion = BimodalCrossAttentionFusion(
            query_dim=smiles_encoder.output_dim,
            context_dim=morgan_encoder.output_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            feedforward_dim=fusion_feedforward_dim,
            dropout=dropout,
            pooling="first_token",
        )
        self.fusion_projection = ProjectionHead(
            input_dim=self.fusion.output_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        residual_dim = smiles_encoder.output_dim + self.fusion.output_dim + morgan_encoder.output_dim
        self.regression_head = nn.Sequential(
            nn.Linear(residual_dim, regression_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, regression_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morgan_fp: torch.Tensor,
        need_attention_weights: bool = False,
    ) -> MultimodalForwardOutput:
        smiles_out = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask)
        morgan_out = self.morgan_encoder(morgan_fp=morgan_fp)
        fusion_out = self.fusion(
            query_sequence=smiles_out.sequence,
            context_sequence=morgan_out.sequence,
            query_mask=smiles_out.mask,
            context_mask=morgan_out.mask,
            need_weights=need_attention_weights,
        )
        residual_feature = torch.cat([smiles_out.pooled, fusion_out.pooled, morgan_out.pooled], dim=-1)
        prediction = self.regression_head(residual_feature).squeeze(-1)
        return MultimodalForwardOutput(
            prediction=prediction,
            smiles_pooled=smiles_out.pooled,
            morgan_pooled=morgan_out.pooled,
            fusion_pooled=fusion_out.pooled,
            z_smiles=self.smiles_projection(smiles_out.pooled),
            z_morgan=self.morgan_projection(morgan_out.pooled),
            z_fusion=self.fusion_projection(fusion_out.pooled),
        )


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters of a module."""

    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters of a module."""

    for param in module.parameters():
        param.requires_grad = True
