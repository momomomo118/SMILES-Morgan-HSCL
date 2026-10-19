"""Decide whether PredRet transfer data needs cleaning before training.

The script reads the prepared PredRet 10-system table and optional audit outputs,
then writes a concise per-system decision report.
"""

from __future__ import annotations

import argparse
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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def ks_test(train: pd.Series, test: pd.Series) -> tuple[float, float]:
    try:
        from scipy.stats import ks_2samp
    except Exception:
        return np.nan, np.nan
    if len(train) == 0 or len(test) == 0:
        return np.nan, np.nan
    result = ks_2samp(train, test)
    return float(result.statistic), float(result.pvalue)


def severity_from_reasons(reasons: list[str]) -> str:
    hard_prefixes = ("rt<=0", "rt_nonfinite", "rt_mismatch", "empty_split")
    if any(any(reason.startswith(prefix) for prefix in hard_prefixes) for reason in reasons):
        return "MUST_CLEAN"
    if reasons:
        return "SHOULD_CLEAN_OR_USE_ROBUST_PROTOCOL"
    return "OK"


def frame_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as markdown without requiring tabulate."""
    if df.empty:
        return "_empty_"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check if PredRet data needs cleaning.")
    parser.add_argument(
        "--data",
        default="data/external_predret/transfer_10_systems/all.csv",
        help="Prepared PredRet all.csv with system/split/rt columns.",
    )
    parser.add_argument(
        "--dropped-rows",
        default="data/external_predret/dropped_rows.csv",
        help="Optional dropped_rows.csv from prepare_predret_data.py.",
    )
    parser.add_argument(
        "--audit-dir",
        default="outputs/predret_data_audit",
        help="Optional directory from audit_predret_data.py.",
    )
    parser.add_argument("--output-dir", default="outputs/predret_cleaning_need")
    parser.add_argument("--max-zero-rt", type=int, default=0)
    parser.add_argument(
        "--max-rt-mismatch",
        type=float,
        default=1e-3,
        help="Allowed absolute difference between rt and rt_min*60 in seconds.",
    )
    parser.add_argument(
        "--multi-rt-group-frac-threshold",
        type=float,
        default=0.03,
        help="Flag same-smiles multi-RT if fraction of groups exceeds this value.",
    )
    parser.add_argument(
        "--split-mean-diff-frac-threshold",
        type=float,
        default=0.20,
        help="Flag split drift if abs(test-train mean) / full std exceeds this value.",
    )
    parser.add_argument("--ks-p-threshold", type=float, default=0.05)
    parser.add_argument(
        "--target-systems",
        nargs="*",
        default=TARGET_SYSTEMS,
        help="Systems to check. Defaults to the paper's 10 PredRet transfer systems.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    df["system"] = df["system"].map(alias_system)
    df = df[df["system"].isin(args.target_systems)].copy()
    df["rt_num"] = pd.to_numeric(df["rt"], errors="coerce")
    if "rt_min" in df.columns:
        df["rt_min_num"] = pd.to_numeric(df["rt_min"], errors="coerce")
        df["rt_from_min"] = df["rt_min_num"] * 60.0
        df["rt_mismatch_abs"] = (df["rt_num"] - df["rt_from_min"]).abs()
    else:
        df["rt_mismatch_abs"] = np.nan

    drop_df = read_csv_if_exists(Path(args.dropped_rows))
    if len(drop_df) and "system" in drop_df.columns:
        drop_df["system_alias"] = drop_df["system"].map(alias_system)

    rows = []
    for system in args.target_systems:
        g = df[df["system"] == system].copy()
        reasons: list[str] = []
        actions: list[str] = []

        split_counts = g["split"].value_counts().to_dict() if "split" in g.columns else {}
        for split in ("train", "test"):
            if split_counts.get(split, 0) == 0:
                reasons.append(f"empty_split:{split}")
                actions.append("regenerate_split")

        rt = g["rt_num"]
        zero_rt = int((rt <= 0).sum())
        nonfinite_rt = int((~np.isfinite(rt)).sum())
        rt_mismatch = int((g["rt_mismatch_abs"] > args.max_rt_mismatch).sum())
        if zero_rt > args.max_zero_rt:
            reasons.append(f"rt<=0:{zero_rt}")
            actions.append("drop_rt_le_0")
        if nonfinite_rt:
            reasons.append(f"rt_nonfinite:{nonfinite_rt}")
            actions.append("drop_nonfinite_rt")
        if rt_mismatch:
            reasons.append(f"rt_mismatch_rt_min_x60:{rt_mismatch}")
            actions.append("fix_rt_unit_or_recompute_rt")

        dropped_rows = 0
        if len(drop_df) and "system_alias" in drop_df.columns:
            dropped_rows = int((drop_df["system_alias"] == system).sum())
            if dropped_rows:
                reasons.append(f"dropped_rows_from_conversion:{dropped_rows}")
                actions.append("inspect_dropped_rows")

        smiles_groups = int(g["smiles"].nunique()) if "smiles" in g.columns else 0
        same_smiles_multi_rt = 0
        same_inchi_multi_rt = 0
        if "smiles" in g.columns:
            same_smiles_multi_rt = int(g.groupby("smiles")["rt_num"].nunique().gt(1).sum())
            frac = same_smiles_multi_rt / max(smiles_groups, 1)
            if frac > args.multi_rt_group_frac_threshold:
                reasons.append(f"same_smiles_multi_rt:{same_smiles_multi_rt}({frac:.1%})")
                actions.append("collapse_system_smiles_rt_by_median")
        if "inchi" in g.columns:
            same_inchi_multi_rt = int(g.groupby("inchi")["rt_num"].nunique().gt(1).sum())

        train = g.loc[g.get("split", "") == "train", "rt_num"].dropna()
        test = g.loc[g.get("split", "") == "test", "rt_num"].dropna()
        full_std = float(rt.std()) if len(rt.dropna()) > 1 else np.nan
        mean_diff = float(test.mean() - train.mean()) if len(train) and len(test) else np.nan
        mean_diff_frac = abs(mean_diff) / full_std if np.isfinite(full_std) and full_std > 0 else np.nan
        ks_stat, ks_p = ks_test(train, test)
        if np.isfinite(mean_diff_frac) and mean_diff_frac > args.split_mean_diff_frac_threshold:
            reasons.append(f"split_mean_shift:{mean_diff:.3f}s/{mean_diff_frac:.1%}std")
            actions.append("use_10fold_cv_or_rt_stratified_split")
        if np.isfinite(ks_p) and ks_p < args.ks_p_threshold:
            reasons.append(f"split_ks_p<{args.ks_p_threshold}:{ks_p:.4g}")
            actions.append("use_10fold_cv_or_rt_stratified_split")

        if same_smiles_multi_rt or zero_rt:
            actions.append("rerun_model_after_cleaning")

        actions = list(dict.fromkeys(actions))
        severity = severity_from_reasons(reasons)
        rows.append(
            {
                "system": system,
                "decision": severity,
                "n": len(g),
                "train_n": split_counts.get("train", 0),
                "val_n": split_counts.get("val", 0),
                "test_n": split_counts.get("test", 0),
                "rt_min": rt.min(),
                "rt_max": rt.max(),
                "rt_mean": rt.mean(),
                "rt_std": rt.std(),
                "zero_or_neg_rt": zero_rt,
                "nonfinite_rt": nonfinite_rt,
                "rt_mismatch_rt_min_x60": rt_mismatch,
                "dropped_rows": dropped_rows,
                "same_smiles_multi_rt_groups": same_smiles_multi_rt,
                "same_inchi_multi_rt_groups": same_inchi_multi_rt,
                "test_minus_train_mean": mean_diff,
                "abs_mean_shift_over_full_std": mean_diff_frac,
                "ks_stat_train_vs_test": ks_stat,
                "ks_p_train_vs_test": ks_p,
                "reasons": "; ".join(reasons) if reasons else "none",
                "recommended_actions": "; ".join(actions) if actions else "none",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "cleaning_need_by_system.csv", index=False)

    must = int((result["decision"] == "MUST_CLEAN").sum())
    should = int((result["decision"] == "SHOULD_CLEAN_OR_USE_ROBUST_PROTOCOL").sum())
    ok = int((result["decision"] == "OK").sum())
    overall = "MUST_CLEAN" if must else ("SHOULD_CLEAN_OR_USE_ROBUST_PROTOCOL" if should else "OK")

    md = [
        "# PredRet Cleaning Need Report",
        "",
        f"- input: `{data_path}`",
        f"- systems checked: {len(result)}",
        f"- overall decision: **{overall}**",
        f"- MUST_CLEAN systems: {must}",
        f"- SHOULD_CLEAN_OR_USE_ROBUST_PROTOCOL systems: {should}",
        f"- OK systems: {ok}",
        "",
        "## Decision Table",
        "",
        frame_to_markdown(result[
            [
                "system",
                "decision",
                "n",
                "zero_or_neg_rt",
                "same_smiles_multi_rt_groups",
                "test_minus_train_mean",
                "ks_p_train_vs_test",
                "reasons",
                "recommended_actions",
            ]
        ]),
        "",
        "## Interpretation",
        "",
        "- `MUST_CLEAN`: contains hard label problems such as RT <= 0, nonfinite RT, RT unit mismatch, or empty required split.",
        "- `SHOULD_CLEAN_OR_USE_ROBUST_PROTOCOL`: labels are not strictly invalid, but duplicate labels or split drift can make the comparison unstable.",
        "- For paper comparison, prefer cleaning plus paper-style 10-fold CV instead of a fixed 81/9/10 split.",
    ]
    (out_dir / "cleaning_need_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== PredRet cleaning need ===")
    print(f"overall decision: {overall}")
    print(
        result[
            [
                "system",
                "decision",
                "n",
                "zero_or_neg_rt",
                "same_smiles_multi_rt_groups",
                "test_minus_train_mean",
                "ks_p_train_vs_test",
                "recommended_actions",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote: {out_dir / 'cleaning_need_by_system.csv'}")
    print(f"Wrote: {out_dir / 'cleaning_need_report.md'}")


if __name__ == "__main__":
    main()
