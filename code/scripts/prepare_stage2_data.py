"""Prepare timestamped Stage 2 validation CSV inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    data_dir = args.stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.project_root / "data" / "unified" / "train.csv"
    train = pd.read_csv(train_path)

    sanity = train.sample(n=20, random_state=args.seed).sort_index().reset_index(drop=True)
    sanity.to_csv(data_dir / "sanity_20.csv", index=False)

    shuffled = train.copy()
    shuffled["rt"] = shuffled["rt"].sample(frac=1.0, random_state=args.seed).to_numpy()
    shuffled.to_csv(data_dir / "train_label_shuffled.csv", index=False)

    print(f"Wrote {data_dir / 'sanity_20.csv'}")
    print(f"Wrote {data_dir / 'train_label_shuffled.csv'}")


if __name__ == "__main__":
    main()
