"""Create repeated train/val/test splits for PredRet systems.

The Transformer-LSTM paper reports mean +/- std across repeated transfer
experiments. This script creates one split folder per system and seed so the
same training command can be run repeatedly and summarized later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_SYSTEMS = [
    "FEM_Lipids",
    "FEM_long",
    "FEM_short",
    "LIFE_new",
    "LIFE_old",
    "Eawag_XBridgeC18",
    "IPB_Halle",
    "UniToyama_Atlantis",
    "MTBLS87",
    "Cao_HILIC",
]


def split_one(df: pd.DataFrame, system: str, seed: int, out_dir: Path) -> dict[str, int]:
    g = df[df["system"].eq(system)].copy()
    if g.empty:
        raise ValueError(f"No rows found for system={system}")

    g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_val, test = train_test_split(g, test_size=0.10, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.10, random_state=seed)

    parts = {"train": train, "val": val, "test": test}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, part in parts.items():
        part = part.copy().reset_index(drop=True)
        part["split"] = split
        part["sample_id"] = [f"{system}_seed{seed}_{split}_{i:06d}" for i in range(len(part))]
        part.to_csv(out_dir / f"{split}.csv", index=False)
        written[split] = len(part)

    all_df = pd.concat([parts["train"], parts["val"], parts["test"]], ignore_index=True)
    all_df.to_csv(out_dir / "all.csv", index=False)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/external_predret/transfer_10_systems/all.csv")
    ap.add_argument("--output-root", default="data/external_predret/by_system_repeated")
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[2021, 2022, 2023, 2024, 2025])
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_root = Path(args.output_root)

    rows = []
    for system in args.systems:
        for seed in args.seeds:
            out_dir = out_root / system / f"seed_{seed}"
            counts = split_one(df, system, seed, out_dir)
            row = {"system": system, "seed": seed, **counts, "path": str(out_dir)}
            rows.append(row)
            print(row, flush=True)

    summary = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "split_summary.csv", index=False)
    print(f"Wrote split summary: {out_root / 'split_summary.csv'}")


if __name__ == "__main__":
    main()
