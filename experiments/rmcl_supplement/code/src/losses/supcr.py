"""Supervised contrastive regression losses.

This module follows the HSCL paper formula:

    L = - 1/(2N) sum_i 1/(2N-1) sum_{j != i}
        exp(R_i dot R_j / tau)
        / sum_{k in A(i,j)} exp(R_i dot R_k / tau)

where A(i,j) = {k != i | |y_i - y_k| >= |y_i - y_j|}.

For RT prediction, y is the continuous retention-time label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlignmentLossOutput:
    total: torch.Tensor
    supcr: torch.Tensor
    similarity: torch.Tensor


class SupervisedContrastiveRegressionLoss(nn.Module):
    """Supervised contrastive regression for continuous RT labels.

    Args:
        temperature: Softmax temperature for dot-product similarities.
        normalize: Whether to L2-normalize features before similarity.
        use_log: If False, uses the paper equation exactly, i.e. negative
            probability ratio. If True, uses the common negative-log variant
            -log(exp/sum) for stronger gradients.
        max_pairs_per_anchor: Optional cap on candidates per anchor. When set,
            only the closest candidates in label distance are used. Leave as
            None for the full HSCL-style ranking objective.
        positive_mode: "rank" keeps the original HSCL ranking objective.
            "rt_bucket" treats near-RT samples as explicit positives.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
        use_log: bool = False,
        max_pairs_per_anchor: Optional[int] = None,
        positive_mode: str = "rank",
        strong_positive_threshold: float = 30.0,
        weak_positive_threshold: float = 60.0,
        weak_positive_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.normalize = normalize
        self.use_log = use_log
        self.max_pairs_per_anchor = max_pairs_per_anchor
        self.positive_mode = positive_mode
        self.strong_positive_threshold = strong_positive_threshold
        self.weak_positive_threshold = weak_positive_threshold
        self.weak_positive_weight = weak_positive_weight
        if self.positive_mode not in {"rank", "rt_bucket"}:
            raise ValueError("positive_mode must be one of: rank, rt_bucket")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCR on one feature set.

        Args:
            features: Tensor of shape [batch, dim].
            labels: Continuous RT labels of shape [batch].
        """

        if features.dim() != 2:
            raise ValueError(f"features must have shape [batch, dim], got {tuple(features.shape)}")
        labels = labels.view(-1).to(features.device)
        if labels.numel() != features.size(0):
            raise ValueError("labels length must match feature batch size.")
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        if self.normalize:
            features = F.normalize(features, dim=-1)

        similarities = features @ features.T / self.temperature
        label_distances = torch.abs(labels[:, None] - labels[None, :])
        self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)

        losses = []
        for anchor_idx in range(features.size(0)):
            valid = ~self_mask[anchor_idx]
            sim_i = similarities[anchor_idx, valid]
            dist_i = label_distances[anchor_idx, valid]

            if self.max_pairs_per_anchor is not None and self.max_pairs_per_anchor < sim_i.numel():
                keep = torch.argsort(dist_i, descending=False)[: self.max_pairs_per_anchor]
                sim_i = sim_i[keep]
                dist_i = dist_i[keep]

            if self.positive_mode == "rt_bucket":
                strong_positive = dist_i <= self.strong_positive_threshold
                weak_positive = (dist_i > self.strong_positive_threshold) & (
                    dist_i <= self.weak_positive_threshold
                )
                positive_mask = strong_positive | weak_positive
                if not positive_mask.any():
                    continue

                weights = torch.zeros_like(sim_i)
                weights = torch.where(strong_positive, torch.ones_like(weights), weights)
                weights = torch.where(
                    weak_positive,
                    torch.full_like(weights, self.weak_positive_weight),
                    weights,
                )
                log_prob = sim_i - torch.logsumexp(sim_i, dim=0)
                candidate_loss = -(weights[positive_mask] * log_prob[positive_mask])
                losses.append(candidate_loss.sum() / weights[positive_mask].sum().clamp_min(1e-12))
                continue

            # Formula denominator for each candidate j:
            # A(i,j) = {k != i | |y_i - y_k| >= |y_i - y_j|}.
            negative_or_equal_mask = dist_i[None, :] >= dist_i[:, None]

            if self.use_log:
                log_denominator = torch.logsumexp(
                    sim_i.unsqueeze(0).masked_fill(~negative_or_equal_mask, float("-inf")),
                    dim=1,
                )
                candidate_loss = -(sim_i - log_denominator)
            else:
                # Exact paper equation, implemented with a per-anchor max
                # subtraction for numerical stability.
                shifted_sim = sim_i - sim_i.max().detach()
                exp_sim = torch.exp(shifted_sim)
                denominator = (exp_sim.unsqueeze(0) * negative_or_equal_mask.to(exp_sim.dtype)).sum(dim=1)
                ratio = exp_sim / denominator.clamp_min(torch.finfo(exp_sim.dtype).eps)
                candidate_loss = -ratio

            losses.append(candidate_loss.mean())

        if not losses:
            return features.new_tensor(0.0)
        return torch.stack(losses).mean()


class CrossModalSupCRLoss(nn.Module):
    """SupCR alignment between two modality projections."""

    def __init__(
        self,
        temperature: float = 0.1,
        similarity_weight: float = 0.1,
        normalize: bool = True,
        use_log: bool = False,
        max_pairs_per_anchor: Optional[int] = None,
        positive_mode: str = "rank",
        strong_positive_threshold: float = 30.0,
        weak_positive_threshold: float = 60.0,
        weak_positive_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.supcr = SupervisedContrastiveRegressionLoss(
            temperature=temperature,
            normalize=normalize,
            use_log=use_log,
            max_pairs_per_anchor=max_pairs_per_anchor,
            positive_mode=positive_mode,
            strong_positive_threshold=strong_positive_threshold,
            weak_positive_threshold=weak_positive_threshold,
            weak_positive_weight=weak_positive_weight,
        )
        self.similarity_weight = similarity_weight

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, labels: torch.Tensor) -> AlignmentLossOutput:
        """Align two modalities with SupCR plus squared L2 similarity loss.

        Args:
            z_a: Projection from modality A, shape [batch, projection_dim].
            z_b: Projection from modality B, shape [batch, projection_dim].
            labels: RT labels, shape [batch].
        """

        if z_a.shape != z_b.shape:
            raise ValueError(f"z_a and z_b must have the same shape, got {z_a.shape} and {z_b.shape}")

        features = torch.cat([z_a, z_b], dim=0)
        repeated_labels = torch.cat([labels.view(-1), labels.view(-1)], dim=0)
        supcr_loss = self.supcr(features, repeated_labels)
        similarity_loss = F.mse_loss(z_a, z_b)
        total = supcr_loss + self.similarity_weight * similarity_loss
        return AlignmentLossOutput(total=total, supcr=supcr_loss, similarity=similarity_loss)


class FusionSupCRLoss(nn.Module):
    """SupCR loss for fused multimodal representations."""

    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
        use_log: bool = False,
        max_pairs_per_anchor: Optional[int] = None,
        positive_mode: str = "rank",
        strong_positive_threshold: float = 30.0,
        weak_positive_threshold: float = 60.0,
        weak_positive_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.supcr = SupervisedContrastiveRegressionLoss(
            temperature=temperature,
            normalize=normalize,
            use_log=use_log,
            max_pairs_per_anchor=max_pairs_per_anchor,
            positive_mode=positive_mode,
            strong_positive_threshold=strong_positive_threshold,
            weak_positive_threshold=weak_positive_threshold,
            weak_positive_weight=weak_positive_weight,
        )

    def forward(self, z_fusion: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.supcr(z_fusion, labels)
