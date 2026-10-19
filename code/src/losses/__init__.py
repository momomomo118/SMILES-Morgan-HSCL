"""Loss functions for SMILES-Morgan HSCL."""

from .supcr import (
    AlignmentLossOutput,
    CrossModalSupCRLoss,
    FusionSupCRLoss,
    SupervisedContrastiveRegressionLoss,
)

__all__ = [
    "AlignmentLossOutput",
    "CrossModalSupCRLoss",
    "FusionSupCRLoss",
    "SupervisedContrastiveRegressionLoss",
]
