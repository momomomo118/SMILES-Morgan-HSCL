"""Prepare one shared SMILES/RT split for multimodal training.

This script converts the already reproduced SMILES baseline splits into a
single canonical input format used by both the SMILES branch and the Morgan
fingerprint branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMILES_DATA_DIR = (
    PROJECT_ROOT
    / "smiles_baseline"
    / "RT-TR-main"
    / "RT-TR-main"
    / "data"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "unified"


def read_split(path: Path, split: str) -> pd.DataFrame:
    """Read one baseline split and normalize columns."""
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", header=None, names=["smiles", "rt"])
    else:
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

    required = {"smiles", "rt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.loc[:, ["smiles", "rt"]].copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["smiles", "rt"])
    df = df[df["smiles"] != ""]
    df.insert(0, "sample_id", [f"{split}_{i:06d}" for i in range(len(df))])
    df["split"] = split
    return df


def build_unified_dataset(source_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create train/val/test/all CSV files in the unified schema."""
    split_files = {
        "train": source_dir / "train.csv",
        "val": source_dir / "val.csv",
        "test": source_dir / "test.csv",
    }
    frames = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, path in split_files.items():
        if not path.exists():
            fallback = source_dir / f"{split}_fixed.tsv"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing both {path} and {fallback}")
            path = fallback

        df = read_split(path, split)
        df.to_csv(output_dir / f"{split}.csv", index=False)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(output_dir / "all.csv", index=False)

    summary = (
        all_df.groupby("split")
        .agg(samples=("sample_id", "count"), rt_min=("rt", "min"), rt_max=("rt", "max"))
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return all_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared multimodal SMILES/RT input files.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SMILES_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    all_df = build_unified_dataset(args.source_dir, args.output_dir)
    print(f"Wrote unified data to: {args.output_dir}")
    print(f"Total samples: {len(all_df)}")
    print(all_df.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
