"""Create condition-aware PredRet evaluation protocols.

Outputs:
- leave_one_system_out: train on all other systems, validate/test on held-out system
- same_system_scaffold: scaffold-based train/val/test split inside each system
- few_shot_target: source systems plus 8/16/32 labeled target examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SYSTEMS = [
    "Eawag_XBridgeC18",
    "FEM_Lipids",
    "FEM_long",
    "FEM_short",
    "IPB_Halle",
    "LIFE_new",
    "LIFE_old",
    "UniToyama_Atlantis",
    "Cao_HILIC",
    "MTBLS87",
]


def smiles_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold:
                return scaffold
    except Exception:
        pass
    return hashlib.sha1(str(smiles).encode("utf-8")).hexdigest()[:12]


def split_target_for_val_test(target: pd.DataFrame, seed: int, val_frac: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = target.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(target) * val_frac)))
    if len(target) > 1:
        n_val = min(n_val, len(target) - 1)
    val = target.iloc[:n_val].copy()
    test = target.iloc[n_val:].copy()
    if test.empty:
        test = val.copy()
    return val, test


def write_split_dir(split_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train.assign(split="train").to_csv(split_dir / "train.csv", index=False)
    val.assign(split="val").to_csv(split_dir / "val.csv", index=False)
    test.assign(split="test").to_csv(split_dir / "test.csv", index=False)
    return {"train": len(train), "val": len(val), "test": len(test)}


def make_leave_one_system_out(df: pd.DataFrame, out: Path, systems: list[str], seed: int) -> list[dict[str, object]]:
    root = out / "leave_one_system_out"
    records = []
    for system in systems:
        target = df[df["system"].eq(system)].copy()
        source = df[~df["system"].eq(system)].copy()
        val, test = split_target_for_val_test(target, seed)
        counts = write_split_dir(root / system, source, val, test)
        records.append({"protocol": "leave_one_system_out", "system": system, **counts})
    return records


def make_same_system_scaffold(df: pd.DataFrame, out: Path, systems: list[str], seed: int) -> list[dict[str, object]]:
    root = out / "same_system_scaffold"
    records = []
    for system in systems:
        g = df[df["system"].eq(system)].copy()
        g["scaffold"] = g["smiles"].map(smiles_scaffold)
        scaffold_sizes = g.groupby("scaffold").size().sort_values(ascending=False)
        scaffold_order = list(scaffold_sizes.index)
        rng = np.random.default_rng(seed)
        rng.shuffle(scaffold_order)

        n = len(g)
        target_test = max(1, int(round(0.10 * n)))
        target_val = max(1, int(round(0.10 * n)))
        buckets = {"test": [], "val": [], "train": []}
        counts = {"test": 0, "val": 0, "train": 0}
        for scaffold in scaffold_order:
            size = int(scaffold_sizes.loc[scaffold])
            if counts["test"] < target_test:
                bucket = "test"
            elif counts["val"] < target_val:
                bucket = "val"
            else:
                bucket = "train"
            buckets[bucket].append(scaffold)
            counts[bucket] += size

        train = g[g["scaffold"].isin(buckets["train"])].drop(columns=["scaffold"])
        val = g[g["scaffold"].isin(buckets["val"])].drop(columns=["scaffold"])
        test = g[g["scaffold"].isin(buckets["test"])].drop(columns=["scaffold"])
        if train.empty:
            train = g.iloc[max(len(val) + len(test), 1) :].drop(columns=["scaffold"])
        counts_out = write_split_dir(root / system / "fold_00", train, val, test)
        records.append({"protocol": "same_system_scaffold", "system": system, **counts_out})
    return records


def make_few_shot(df: pd.DataFrame, out: Path, systems: list[str], shots: list[int], seed: int) -> list[dict[str, object]]:
    root = out / "few_shot_target"
    records = []
    for system in systems:
        source = df[~df["system"].eq(system)].copy()
        target = df[df["system"].eq(system)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for shot in shots:
            n_shot = min(shot, max(len(target) - 2, 1))
            target_train = target.iloc[:n_shot].copy()
            remainder = target.iloc[n_shot:].copy()
            val, test = split_target_for_val_test(remainder, seed + shot, val_frac=0.25)
            train = pd.concat([source, target_train], ignore_index=True)
            counts = write_split_dir(root / system / f"{shot}_shot", train, val, test)
            records.append({"protocol": "few_shot_target", "system": system, "shots": shot, **counts})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/external_predret/condition_metadata/row_conditions.csv")
    parser.add_argument("--output-dir", default="data/external_predret/condition_protocols")
    parser.add_argument("--systems", nargs="*", default=TARGET_SYSTEMS)
    parser.add_argument("--shots", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = df[df["system"].isin(args.systems)].copy()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    records.extend(make_leave_one_system_out(df, out, args.systems, args.seed))
    records.extend(make_same_system_scaffold(df, out, args.systems, args.seed))
    records.extend(make_few_shot(df, out, args.systems, args.shots, args.seed))

    summary = pd.DataFrame(records)
    summary.to_csv(out / "protocol_summary.csv", index=False)
    payload = {
        "input": args.input,
        "output_dir": str(out),
        "seed": args.seed,
        "systems": args.systems,
        "shots": args.shots,
        "protocols": sorted(summary["protocol"].unique()),
    }
    (out / "protocol_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
