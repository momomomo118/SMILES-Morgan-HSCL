"""SMILES encoder for the multimodal HSCL model.

This file re-implements the reproduced ChemBERTa/RoBERTa + BiLSTM baseline as
an encoder. It does not contain the RT regression head, so the output can be
used by multimodal fusion and contrastive learning modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import RobertaModel

try:
    from .projection_head import ProjectionHead
except ImportError:  # pragma: no cover - supports direct script imports
    from projection_head import ProjectionHead


@dataclass
class EncoderOutput:
    """Common encoder output used by fusion and contrastive modules."""

    sequence: torch.Tensor
    pooled: torch.Tensor
    mask: Optional[torch.Tensor] = None


def resolve_roberta_layers(num_hidden_layers: int, layers: Iterable[int]) -> List[int]:
    """Convert human layer ids to Hugging Face hidden-state indices.

    Hugging Face returns hidden_states as:
    index 0 = embedding output, index 1..N = transformer layer outputs.
    Therefore paper-style RoBERTa layers 2/4/6 map directly to hidden_states
    indices 2/4/6 for a 6-layer ChemBERTa/RoBERTa backbone.
    """

    resolved = []
    for layer in layers:
        if layer < 0:
            layer = num_hidden_layers + 1 + layer
        if layer < 0 or layer > num_hidden_layers:
            raise ValueError(
                f"Layer index {layer} is outside hidden_states range 0..{num_hidden_layers}."
            )
        resolved.append(layer)
    return resolved


class SMILESEncoder(nn.Module):
    """ChemBERTa/RoBERTa layer fusion followed by recurrent SMILES encoders."""

    def __init__(
        self,
        model_name_or_path: str,
        layers_to_use: Iterable[int] = (2, 4, 6),
        lstm_hidden_size: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pooling: str = "first_token",
        freeze_roberta: bool = False,
    ) -> None:
        super().__init__()
        try:
            self.roberta = RobertaModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'get'" not in str(exc):
                raise
            self.roberta = RobertaModel.from_pretrained(
                model_name_or_path,
                add_pooling_layer=False,
                use_safetensors=False,
            )
        self.layers_to_use = resolve_roberta_layers(self.roberta.config.num_hidden_layers, layers_to_use)
        self.pooling = pooling

        roberta_hidden = self.roberta.config.hidden_size
        fused_hidden = roberta_hidden * len(self.layers_to_use)
        self.lstm = nn.LSTM(
            input_size=fused_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.0,
        )
        self.output_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.dropout = nn.Dropout(dropout)

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        fused = torch.cat([outputs.hidden_states[i] for i in self.layers_to_use], dim=-1)
        sequence, _ = self.lstm(fused)
        sequence = self.dropout(sequence)
        pooled = self._pool(sequence, attention_mask)
        return EncoderOutput(sequence=sequence, pooled=pooled, mask=attention_mask)

    def _pool(self, sequence: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "first_token":
            return sequence[:, 0, :]
        if self.pooling == "last_token":
            if attention_mask is None:
                return sequence[:, -1, :]
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            return sequence[torch.arange(sequence.size(0), device=sequence.device), lengths]
        if self.pooling == "mean":
            if attention_mask is None:
                return sequence.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
            return (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        raise ValueError(f"Unsupported pooling: {self.pooling}")


class SMILESProjectionHead(nn.Module):
    """Projection head for supervised contrastive learning."""

    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = ProjectionHead(input_dim=input_dim, projection_dim=projection_dim, dropout=dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)
