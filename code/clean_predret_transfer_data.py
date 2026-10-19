"""Clean the 10 PredRet transfer systems and create paper-style 10-fold splits.

Cleaning policy:
- keep only the ten systems used in the Transformer-LSTM transfer table
- drop nonfinite RT and RT <= 0
- collapse duplicate (system, smiles) rows to the median RT
- preserve representative metadata for inspection
- create 10 folds per system: 90% train and 10% test/validation
"""

from __future__ import annotations

import argparse
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

ALIASES = {
    "FEM_lipids": "FEM_Lipids",
    "FEM_Lipids": "FEM_Lipids",
    "UniToyama": "UniToyama_Atlantis",
    "UniToyama_Atlantis": "UniToyama_Atlantis",
}


def alias_system(value: object) -> str:
    text = str(value)
    return ALIASES.get(text, text)


def join_unique(values: pd.Series, limit: int = 8) -> str:
    items = []
    for value in values.dropna().astype(str):
        if value not in items:
            items.append(value)
        if len(items) >= limit:
            break
    return " | ".join(items)


def qbin_for_stratification(rt: pd.Series, n_bins: int = 10) -> pd.Series:
    """Return robust quantile bins, backing off when a system has few unique RTs."""
    clean = pd.to_numeric(rt, errors="coerce")
    for bins in range(min(n_bins, clean.nunique()), 1, -1):
        try:
            labels = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
            counts = labels.value_counts()
            if len(counts) >= 2 and counts.min() >= 2:
                return labels.astype(int)
        except Exception:
            continue
    return pd.Series(np.zeros(len(clean), dtype=int), index=rt.index)


def split_one_system(g: pd.DataFrame, seed: int) -> pd.DataFrame:
    g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_splits = min(10, len(g))
    if n_splits < 2:
        g["fold"] = 0
        g["split"] = "train"
        return g

    bins = qbin_for_stratification(g["rt"], n_bins=10)
    rng = np.random.default_rng(seed)
    test_indices_by_fold = [[] for _ in range(n_splits)]
    if bins.nunique() >= 2:
        # Round-robin each RT quantile bin into folds. This approximates
        # stratified K-fold without requiring scikit-learn.
        for _, idx_values in pd.Series(g.index, index=bins).groupby(level=0):
            idx = np.array(idx_values.tolist(), dtype=int)
            rng.shuffle(idx)
            for pos, value in enumerate(idx):
                test_indices_by_fold[pos % n_splits].append(int(value))
    else:
        idx = np.arange(len(g))
        rng.shuffle(idx)
        for pos, value in enumerate(idx):
            test_indices_by_fold[pos % n_splits].append(int(value))

    parts = []
    all_idx = set(range(len(g)))
    for fold, test_idx_list in enumerate(test_indices_by_fold):
        test_idx = sorted(test_idx_list)
        train_idx = sorted(all_idx.difference(test_idx))
        train = g.iloc[train_idx].copy()
        test = g.iloc[test_idx].copy()
        train["fold"] = fold
        test["fold"] = fold
        train["split"] = "train"
        test["split"] = "test"
        parts.extend([train, test])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean PredRet transfer data.")
    parser.add_argument("--input", default="data/external_predret/transfer_10_systems/all.csv")
    parser.add_argument("--output-dir", default="data/external_predret/transfer_10_systems_cleaned")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-systems", nargs="*", default=TARGET_SYSTEMS)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repeated = out / "by_system_10fold"
    repeated.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    df["system"] = df["system"].map(alias_system)
    df = df[df["system"].isin(args.target_systems)].copy()
    before_rows = len(df)
    before_counts = df["system"].value_counts().sort_index().to_dict()

    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    if "rt_min" in df.columns:
        df["rt_min"] = pd.to_numeric(df["rt_min"], errors="coerce")
    invalid = df[(~np.isfinite(df["rt"])) | (df["rt"] <= 0)].copy()
    invalid.to_csv(out / "removed_invalid_rt_rows.csv", index=False)
    df = df[np.isfinite(df["rt"]) & (df["rt"] > 0)].copy()

    grouped_rows = []
    duplicate_groups = []
    meta_cols = [c for c in ["sample_id", "compound_name", "inchi", "source_row", "pubchem", "username", "time"] if c in df.columns]
    for (system, smiles), g in df.groupby(["system", "smiles"], sort=False):
        rt_values = g["rt"].astype(float)
        rt_median = float(rt_values.median())
        row = {
            "system": system,
            "smiles": smiles,
            "rt": rt_median,
            "rt_min": rt_median / 60.0,
            "n_merged": int(len(g)),
            "rt_source_min": float(rt_values.min()),
            "rt_source_max": float(rt_values.max()),
            "rt_source_std": float(rt_values.std()) if len(g) > 1 else 0.0,
        }
        for col in meta_cols:
            if col == "source_row":
                row[col] = join_unique(g[col].astype(str), limit=20)
            else:
                row[col] = join_unique(g[col])
        grouped_rows.append(row)
        if len(g) > 1 or rt_values.nunique() > 1:
            duplicate_groups.append(row.copy())

    clean = pd.DataFrame(grouped_rows)
    clean = clean.sort_values(["system", "rt", "smiles"]).reset_index(drop=True)
    clean["sample_id"] = [f"predret_clean_{i + 1:06d}" for i in range(len(clean))]
    clean["split"] = "all"
    clean.to_csv(out / "all_cleaned.csv", index=False)
    pd.DataFrame(duplicate_groups).to_csv(out / "merged_duplicate_smiles_groups.csv", index=False)

    split_all_parts = []
    for system in args.target_systems:
        g = clean[clean["system"] == system].copy()
        folded = split_one_system(g, args.seed)
        sys_dir = repeated / system
        sys_dir.mkdir(parents=True, exist_ok=True)
        folded.to_csv(sys_dir / "all_folds.csv", index=False)
        for fold, fg in folded.groupby("fold"):
            fold_dir = sys_dir / f"fold_{int(fold):02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fg[fg["split"] == "train"].to_csv(fold_dir / "train.csv", index=False)
            test = fg[fg["split"] == "test"].copy()
            test.to_csv(fold_dir / "test.csv", index=False)
            test.to_csv(fold_dir / "val.csv", index=False)
        split_all_parts.append(folded)

    all_folds = pd.concat(split_all_parts, ignore_index=True)
    all_folds.to_csv(out / "all_cleaned_10fold_long.csv", index=False)

    summary = []
    for system in args.target_systems:
        before = before_counts.get(system, 0)
        after = int((clean["system"] == system).sum())
        removed_invalid = int((invalid["system"] == system).sum())
        merged = int(((clean["system"] == system) & (clean["n_merged"] > 1)).sum())
        rt = clean.loc[clean["system"] == system, "rt"]
        summary.append(
            {
                "system": system,
                "before_rows": before,
                "after_rows": after,
                "removed_invalid_rt_rows": removed_invalid,
                "merged_smiles_groups": merged,
                "rt_min": rt.min(),
                "rt_max": rt.max(),
                "rt_mean": rt.mean(),
                "rt_std": rt.std(),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "cleaning_summary.csv", index=False)

    payload = {
        "input": str(inp),
        "output_dir": str(out),
        "seed": args.seed,
        "policy": [
            "drop nonfinite RT and RT <= 0",
            "collapse duplicate (system, smiles) rows to median RT",
            "create per-system 10-fold 90/10 splits; test is also copied to val for train_multimodal compatibility",
        ],
        "before_rows": before_rows,
        "after_rows": int(len(clean)),
        "removed_invalid_rt_rows": int(len(invalid)),
        "merged_smiles_groups": int((clean["n_merged"] > 1).sum()),
        "systems": summary,
    }
    (out / "cleaning_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# Cleaned PredRet Transfer Data",
        "",
        f"- input rows: {before_rows}",
        f"- output rows: {len(clean)}",
        f"- removed invalid RT rows: {len(invalid)}",
        f"- merged duplicate smiles groups: {(clean['n_merged'] > 1).sum()}",
        f"- output: `{out}`",
        "",
        "## Summary",
        "",
        summary_df.to_string(index=False),
    ]
    (out / "README_cleaned.md").write_text("\n".join(report), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Wrote: {out / 'all_cleaned.csv'}")
    print(f"Wrote: {out / 'by_system_10fold'}")
    print(f"Wrote: {out / 'cleaning_summary.csv'}")


if __name__ == "__main__":
    main()
