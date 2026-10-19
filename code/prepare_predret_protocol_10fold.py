"""Create paper-protocol PredRet 10-fold splits without cleaning.

This is the main comparison data for the Transformer-LSTM PredRet transfer
table: keep the prepared PredRet rows as-is, only normalize system aliases, and
create per-system 90/10 folds. Cleaned data should be treated as an ablation.
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


def split_one_system(g: pd.DataFrame, seed: int, n_folds: int = 10) -> pd.DataFrame:
    g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_splits = min(n_folds, len(g))
    rng = np.random.default_rng(seed)

    idx = np.arange(len(g))
    rng.shuffle(idx)
    fold_ids = np.arange(len(g)) % n_splits
    rng.shuffle(fold_ids)

    parts = []
    for fold in range(n_splits):
        test_idx = set(idx[fold_ids == fold].tolist())
        train_mask = [i not in test_idx for i in range(len(g))]
        test_mask = [i in test_idx for i in range(len(g))]
        train = g.loc[train_mask].copy()
        test = g.loc[test_mask].copy()
        train["fold"] = fold
        test["fold"] = fold
        train["split"] = "train"
        test["split"] = "test"
        parts.extend([train, test])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/external_predret/transfer_10_systems/all.csv")
    parser.add_argument("--output-dir", default="data/external_predret/transfer_10_systems_protocol_10fold")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--systems", nargs="*", default=TARGET_SYSTEMS)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    repeated = out / "by_system_10fold"
    repeated.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    df["system"] = df["system"].map(alias_system)
    df = df[df["system"].isin(args.systems)].copy()
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    all_parts = []
    summary = []
    for system in args.systems:
        g = df[df["system"].eq(system)].copy()
        if g.empty:
            raise ValueError(f"No rows found for system={system}")
        folded = split_one_system(g, seed=args.seed, n_folds=10)
        sys_dir = repeated / system
        sys_dir.mkdir(parents=True, exist_ok=True)
        folded.to_csv(sys_dir / "all_folds.csv", index=False)

        for fold, fg in folded.groupby("fold"):
            fold_dir = sys_dir / f"fold_{int(fold):02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train = fg[fg["split"].eq("train")].copy()
            test = fg[fg["split"].eq("test")].copy()
            train.to_csv(fold_dir / "train.csv", index=False)
            # The paper describes 90/10 CV; train_multimodal needs a validation
            # file, so use the held-out fold for model selection and final eval.
            test.to_csv(fold_dir / "val.csv", index=False)
            test.to_csv(fold_dir / "test.csv", index=False)

        rt = g["rt"]
        summary.append(
            {
                "system": system,
                "rows": len(g),
                "rt_min": float(rt.min()),
                "rt_max": float(rt.max()),
                "rt_mean": float(rt.mean()),
                "rt_std": float(rt.std()),
                "nonpositive_rt_rows": int((rt <= 0).sum()),
                "nonfinite_rt_rows": int((~np.isfinite(rt)).sum()),
            }
        )
        all_parts.append(folded)

    all_folds = pd.concat(all_parts, ignore_index=True)
    all_folds.to_csv(out / "all_protocol_10fold_long.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "summary_protocol_10fold.csv", index=False)

    payload = {
        "input": str(inp),
        "output_dir": str(out),
        "seed": args.seed,
        "policy": "unfiltered prepared PredRet rows; per-system 10-fold 90/10; held-out fold copied to val and test",
        "systems": summary,
    }
    (out / "summary_protocol_10fold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
