"""Small smoke test for SupCR losses."""

from __future__ import annotations

import torch

from supcr import CrossModalSupCRLoss, FusionSupCRLoss, SupervisedContrastiveRegressionLoss


def main() -> None:
    torch.manual_seed(1234)
    batch_size = 8
    dim = 256
    labels = torch.linspace(300.0, 900.0, batch_size)
    z_smiles = torch.randn(batch_size, dim, requires_grad=True)
    z_morgan = torch.randn(batch_size, dim, requires_grad=True)

    supcr = SupervisedContrastiveRegressionLoss()
    unimodal_loss = supcr(z_smiles, labels)

    cross_modal = CrossModalSupCRLoss()
    alignment = cross_modal(z_smiles, z_morgan, labels)

    fusion = FusionSupCRLoss()
    fusion_loss = fusion((z_smiles + z_morgan) / 2, labels)

    total = unimodal_loss + alignment.total + fusion_loss
    total.backward()

    print("SupCR smoke test OK")
    print("unimodal:", float(unimodal_loss.detach()))
    print("alignment total:", float(alignment.total.detach()))
    print("fusion:", float(fusion_loss.detach()))


if __name__ == "__main__":
    main()
