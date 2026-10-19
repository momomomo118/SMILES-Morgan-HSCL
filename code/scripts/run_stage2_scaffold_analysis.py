"""Analyze Bemis-Murcko scaffold overlap and create a scaffold-disjoint split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover
    raise SystemExit("RDKit is required for this script.") from exc


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)


def rt_summary(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    return (
        df.groupby(split_col)
        .agg(
            rows=("sample_id", "size"),
            scaffolds=("scaffold", "nunique"),
            rt_min=("rt", "min"),
            rt_q1=("rt", lambda x: x.quantile(0.25)),
            rt_median=("rt", "median"),
            rt_mean=("rt", "mean"),
            rt_q3=("rt", lambda x: x.quantile(0.75)),
            rt_max=("rt", "max"),
            rt_std=("rt", "std"),
        )
        .reset_index()
    )


def make_scaffold_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = []
    for scaffold, group in df.groupby("scaffold"):
        groups.append((scaffold, len(group), float(group["rt"].mean())))
    groups.sort(key=lambda x: x[1], reverse=True)

    target = {"train": 0.81 * len(df), "val": 0.09 * len(df), "test": 0.10 * len(df)}
    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for scaffold, size, _ in groups:
        ratios = {split: counts[split] / target[split] for split in counts}
        split = min(ratios, key=ratios.get)
        assignment[scaffold] = split
        counts[split] += size

    out = df.copy()
    out["split"] = out["scaffold"].map(assignment)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = args.stage_dir / "scaffold_analysis"
    data_out = args.stage_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    data_root = args.project_root / "data" / "unified"
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(data_root / f"{split}.csv")
        df["original_split"] = split
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["scaffold"] = all_df["smiles"].map(scaffold_smiles)
    all_df.loc[all_df["scaffold"] == "", "scaffold"] = "NO_SCAFFOLD_" + all_df["smiles"].astype(str)

    sets = {split: set(all_df.loc[all_df["original_split"] == split, "scaffold"]) for split in ["train", "val", "test"]}
    overlap = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
        "train_scaffolds": len(sets["train"]),
        "val_scaffolds": len(sets["val"]),
        "test_scaffolds": len(sets["test"]),
        "total_scaffolds": int(all_df["scaffold"].nunique()),
    }
    (out_dir / "scaffold_overlap_summary.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    rt_summary(all_df, "original_split").to_csv(out_dir / "scaffold_split_summary.csv", index=False)
    all_df.loc[all_df["original_split"] == "train", "scaffold"].value_counts().head(100).rename_axis("scaffold").reset_index(name="count").to_csv(out_dir / "top_scaffolds_train.csv", index=False)
    all_df.loc[all_df["original_split"] == "test", "scaffold"].value_counts().head(100).rename_axis("scaffold").reset_index(name="count").to_csv(out_dir / "top_scaffolds_test.csv", index=False)

    split_df = make_scaffold_split(all_df.drop(columns=["original_split"]), args.seed)
    for split in ["train", "val", "test"]:
        cols = ["sample_id", "smiles", "rt", "split"]
        split_df.loc[split_df["split"] == split, cols].to_csv(data_out / f"scaffold_{split}.csv", index=False)
    rt_summary(split_df, "split").to_csv(out_dir / "generated_scaffold_split_rt_summary.csv", index=False)

    scaffold_sets = {split: set(split_df.loc[split_df["split"] == split, "scaffold"]) for split in ["train", "val", "test"]}
    generated_overlap = {
        "train_val": len(scaffold_sets["train"] & scaffold_sets["val"]),
        "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
        "val_test": len(scaffold_sets["val"] & scaffold_sets["test"]),
    }
    (out_dir / "generated_scaffold_split_overlap.json").write_text(json.dumps(generated_overlap, indent=2), encoding="utf-8")
    print(json.dumps({"original_overlap": overlap, "generated_overlap": generated_overlap}, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
