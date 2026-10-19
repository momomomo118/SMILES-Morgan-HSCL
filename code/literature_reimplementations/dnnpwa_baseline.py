from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DNNpwa reproduction starter. This scaffold is intentionally minimal: "
            "replace the descriptor/autoencoder block with the weighted-autoencoder "
            "pretraining details from Ju et al. before reporting final numbers."
        )
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Placeholder baseline for wiring only: predict the training-set mean.
    # The final DNNpwa reproduction should implement weighted autoencoder
    # pretraining + DNN transfer/fine-tuning from the original paper.
    with args.train.open(newline="", encoding="utf-8-sig") as f:
        train_rows = list(csv.DictReader(f))
    target_col = "rt" if "rt" in train_rows[0] else "true_rt"
    mean_rt = float(np.mean([float(row[target_col]) for row in train_rows]))

    with args.test.open(newline="", encoding="utf-8-sig") as f:
        test_rows = list(csv.DictReader(f))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "smiles", "true_rt", "pred_rt"])
        writer.writeheader()
        for i, row in enumerate(test_rows):
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", f"test_{i:06d}"),
                    "smiles": row.get("smiles", ""),
                    "true_rt": row.get("rt", row.get("true_rt")),
                    "pred_rt": mean_rt,
                }
            )


if __name__ == "__main__":
    main()
