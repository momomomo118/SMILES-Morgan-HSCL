"""Create a leakage-free PredRet 10-system / 10-fold protocol.

The cleaned PredRet rows are not changed.  A canonical SMILES column is created
only for grouped splitting and audit files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - optional environment dependency
    Chem = None

try:
    from sklearn.model_selection import GroupKFold
except Exception:  # pragma: no cover - optional environment dependency
    GroupKFold = None


SYSTEMS = [
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


def canonicalize(smiles: str) -> tuple[str, str]:
    if Chem is None:
        return str(smiles), "rdkit_unavailable_used_input_smiles"
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles), "rdkit_parse_failed_used_input_smiles"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), "rdkit_canonical"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fallback_group_kfold(df: pd.DataFrame, group_col: str, n_splits: int, seed: int):
    rng = random.Random(seed)
    group_sizes = df.groupby(group_col, sort=True).size().to_dict()
    groups = list(group_sizes)
    rng.shuffle(groups)
    buckets = [{"groups": [], "n": 0} for _ in range(n_splits)]
    for group in sorted(groups, key=lambda g: group_sizes[g], reverse=True):
        target = min(buckets, key=lambda b: b["n"])
        target["groups"].append(group)
        target["n"] += group_sizes[group]
    all_idx = set(df.index.tolist())
    for bucket in buckets:
        test_groups = set(bucket["groups"])
        test_idx = set(df.index[df[group_col].isin(test_groups)].tolist())
        train_idx = sorted(all_idx - test_idx)
        yield np.asarray(train_idx, dtype=int), np.asarray(sorted(test_idx), dtype=int)


def group_kfold(df: pd.DataFrame, group_col: str, n_splits: int, seed: int):
    groups = df[group_col].to_numpy()
    if GroupKFold is not None:
        yield from GroupKFold(n_splits=n_splits).split(df, groups=groups)
    else:
        yield from fallback_group_kfold(df, group_col, n_splits, seed)


def inner_validation_groups(train_df: pd.DataFrame, group_col: str, val_fraction: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    group_sizes = train_df.groupby(group_col, sort=True).size().to_dict()
    groups = list(group_sizes)
    rng.shuffle(groups)
    target_n = max(1, int(round(len(train_df) * val_fraction)))
    selected: list[str] = []
    selected_n = 0
    for group in sorted(groups, key=lambda g: (group_sizes[g], rng.random())):
        if selected and selected_n >= target_n:
            break
        selected.append(group)
        selected_n += group_sizes[group]
    if len(selected) == len(groups) and len(groups) > 1:
        selected = selected[:-1]
    return set(selected)


def split_overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column].astype(str)).intersection(set(right[column].astype(str))))


def write_split(df: pd.DataFrame, path: Path, split_name: str) -> None:
    out = df.copy()
    out["split"] = split_name
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def audit_fold(system: str, fold: int, fold_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"
    row = {
        "system": system,
        "fold": fold,
        "inner_train_rows": len(train),
        "inner_val_rows": len(val),
        "outer_test_rows": len(test),
        "inner_train_groups": train["canonical_smiles"].nunique(),
        "inner_val_groups": val["canonical_smiles"].nunique(),
        "outer_test_groups": test["canonical_smiles"].nunique(),
        "train_val_sample_overlap": split_overlap(train, val, "sample_id"),
        "train_test_sample_overlap": split_overlap(train, test, "sample_id"),
        "val_test_sample_overlap": split_overlap(val, test, "sample_id"),
        "train_val_canonical_overlap": split_overlap(train, val, "canonical_smiles"),
        "train_test_canonical_overlap": split_overlap(train, test, "canonical_smiles"),
        "val_test_canonical_overlap": split_overlap(val, test, "canonical_smiles"),
        "train_sha256": file_sha256(train_csv),
        "val_sha256": file_sha256(val_csv),
        "test_sha256": file_sha256(test_csv),
    }
    row["val_test_byte_identical"] = row["val_sha256"] == row["test_sha256"]
    row["passed"] = (
        row["train_val_sample_overlap"] == 0
        and row["train_test_sample_overlap"] == 0
        and row["val_test_sample_overlap"] == 0
        and row["train_val_canonical_overlap"] == 0
        and row["train_test_canonical_overlap"] == 0
        and row["val_test_canonical_overlap"] == 0
        and not row["val_test_byte_identical"]
        and len(train) > 0
        and len(val) > 0
        and len(test) > 0
    )
    return row


def build_protocol(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    required = {"system", "smiles", "rt", "sample_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{args.input} missing columns: {sorted(missing)}")

    if args.output_dir.resolve() == Path("server_results/predret_compare").resolve():
        raise ValueError("Refusing to write strict protocol into historical result directory.")

    can_rows = []
    canonical_values = []
    statuses = []
    for smiles in df["smiles"]:
        canonical, status = canonicalize(smiles)
        canonical_values.append(canonical)
        statuses.append(status)
        can_rows.append({"smiles": smiles, "canonical_smiles": canonical, "canonicalization_status": status})
    df = df.copy()
    df["canonical_smiles"] = canonical_values
    df["canonicalization_status"] = statuses

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(can_rows).drop_duplicates().to_csv(args.output_dir / "canonical_smiles_mapping.csv", index=False)

    audit_rows = []
    by_root = args.output_dir / "by_system_10fold"
    systems = args.systems or SYSTEMS
    for system in systems:
        sdf = df[df["system"] == system].reset_index(drop=True)
        if sdf.empty:
            raise ValueError(f"No rows for system {system}")
        if sdf["canonical_smiles"].nunique() < args.outer_folds:
            raise ValueError(f"{system} has fewer canonical groups than outer folds")

        for fold, (outer_train_idx, outer_test_idx) in enumerate(group_kfold(sdf, "canonical_smiles", args.outer_folds, args.seed)):
            outer_train = sdf.iloc[outer_train_idx].copy().reset_index(drop=True)
            outer_test = sdf.iloc[outer_test_idx].copy().reset_index(drop=True)
            val_groups = inner_validation_groups(
                outer_train,
                "canonical_smiles",
                val_fraction=args.val_fraction,
                seed=args.seed + 1009 * fold + 9173 * (systems.index(system) + 1),
            )
            inner_val = outer_train[outer_train["canonical_smiles"].isin(val_groups)].copy().reset_index(drop=True)
            inner_train = outer_train[~outer_train["canonical_smiles"].isin(val_groups)].copy().reset_index(drop=True)

            fold_dir = by_root / system / f"fold_{fold:02d}"
            write_split(outer_train, fold_dir / "outer_train.csv", "outer_train")
            write_split(outer_test, fold_dir / "outer_test.csv", "outer_test")
            write_split(inner_train, fold_dir / "train.csv", "train")
            write_split(inner_val, fold_dir / "val.csv", "val")
            write_split(outer_test, fold_dir / "test.csv", "test")
            audit_rows.append(audit_fold(system, fold, fold_dir, inner_train, inner_val, outer_test))

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "fold_audit.csv", index=False)
    summary = (
        audit.groupby("system")
        .agg(
            folds=("fold", "count"),
            rows_train_mean=("inner_train_rows", "mean"),
            rows_val_mean=("inner_val_rows", "mean"),
            rows_test_mean=("outer_test_rows", "mean"),
            canonical_train_test_overlap_max=("train_test_canonical_overlap", "max"),
            canonical_val_test_overlap_max=("val_test_canonical_overlap", "max"),
            val_test_identical_count=("val_test_byte_identical", "sum"),
            passed_all=("passed", "all"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "summary_strict_10fold.csv", index=False)
    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "cleaned_rows": int(len(df)),
        "systems": systems,
        "outer_folds": args.outer_folds,
        "inner_validation_fraction": args.val_fraction,
        "seed": args.seed,
        "canonicalization_counts": df["canonicalization_status"].value_counts().to_dict(),
        "splitter": "sklearn.model_selection.GroupKFold" if GroupKFold is not None else "deterministic_size_balanced_group_kfold_fallback",
        "passed": bool(audit["passed"].all()),
    }
    (args.output_dir / "strict_protocol.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], ["system", "fold"]].to_dict("records")
        raise RuntimeError(f"Strict protocol audit failed: {failed}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned/all_cleaned.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold"))
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction should be in (0, 0.5)")
    audit = build_protocol(args)
    print(f"Strict protocol passed for {len(audit)} folds.")


if __name__ == "__main__":
    main()
