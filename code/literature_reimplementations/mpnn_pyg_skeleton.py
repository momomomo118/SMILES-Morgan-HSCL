from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MPNN reproduction skeleton for Osipenko et al. This file records the "
            "expected interface. Implement PyTorch Geometric message passing here "
            "before using it for final reported metrics."
        )
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raise SystemExit(
        "MPNN skeleton only. No official code was found in the quick audit. "
        "Implement atom/bond featurization, message-passing layers, readout, "
        "and CSV export with columns sample_id,true_rt,pred_rt before training."
    )


if __name__ == "__main__":
    main()
