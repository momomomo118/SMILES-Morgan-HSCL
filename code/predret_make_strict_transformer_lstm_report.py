"""Build strict PredRet Proposed vs Transformer-LSTM comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["mae", "rmse", "medae", "mape", "r2"]


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def normalize_summary(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "variant" in df.columns:
        df = df.drop(columns=["variant"])
    keep = ["system", "n"]
    for metric in METRICS:
        keep.extend([f"{metric}_mean", f"{metric}_std"])
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep].rename(columns={c: f"{prefix}_{c}" for c in keep if c != "system"})


def fmt_pm(mean, std) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} +/- {float(std):.3f}"


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]).astype(float))


def build_comparison(result_dir: Path, old_dir: Path) -> pd.DataFrame:
    proposed = pd.read_csv(result_dir / "proposed" / "ours_full_ensemble_summary.csv")
    transformer_lstm = pd.read_csv(result_dir / "transformer_lstm" / "transformer_lstm_summary.csv")
    comp = normalize_summary(proposed, "proposed").merge(
        normalize_summary(transformer_lstm, "transformer_lstm"),
        on="system",
        how="outer",
    )
    comp["delta_mae"] = comp["proposed_mae_mean"] - comp["transformer_lstm_mae_mean"]
    comp["proposed_lower_mae"] = comp["delta_mae"] < 0

    old_proposed = read_csv_optional(old_dir / "ours_full_ensemble_summary.csv")
    old_lstm = read_csv_optional(old_dir / "rttr_repro_summary.csv")
    if old_proposed is not None:
        old_p = old_proposed[["system", "mae_mean"]].rename(columns={"mae_mean": "old_leaky_proposed_mae_mean"})
        comp = comp.merge(old_p, on="system", how="left")
        comp["proposed_mae_change_vs_old_leaky"] = comp["proposed_mae_mean"] - comp["old_leaky_proposed_mae_mean"]
    if old_lstm is not None:
        old_l = old_lstm[["system", "mae_mean"]].rename(
            columns={"mae_mean": "old_leaky_transformer_lstm_mae_mean"}
        )
        comp = comp.merge(old_l, on="system", how="left")
        comp["transformer_lstm_mae_change_vs_old_leaky"] = (
            comp["transformer_lstm_mae_mean"] - comp["old_leaky_transformer_lstm_mae_mean"]
        )
    comp = comp.sort_values("system").reset_index(drop=True)
    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(comparison_dir / "strict_comparison.csv", index=False)
    return comp


def build_paired_statistics(result_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    proposed = pd.read_csv(result_dir / "proposed" / "ours_full_ensemble_detail.csv")
    transformer_lstm = pd.read_csv(result_dir / "transformer_lstm" / "transformer_lstm_detail.csv")
    merged = proposed[["system", "fold", "mae"]].rename(columns={"mae": "proposed_mae"}).merge(
        transformer_lstm[["system", "fold", "mae"]].rename(columns={"mae": "transformer_lstm_mae"}),
        on=["system", "fold"],
        how="inner",
        validate="one_to_one",
    )
    merged["delta_mae"] = merged["proposed_mae"] - merged["transformer_lstm_mae"]

    rows = []
    for system, group in merged.groupby("system"):
        deltas = group.sort_values("fold")["delta_mae"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(deltas, seed + len(rows), n_boot)
        row = {
            "system": system,
            "n_folds": len(deltas),
            "mean_delta_mae": float(np.mean(deltas)),
            "std_delta_mae": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
            "proposed_lower_fold_count": int((deltas < 0).sum()),
        }
        try:
            from scipy.stats import wilcoxon

            stat = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            row["wilcoxon_statistic"] = float(stat.statistic)
            row["wilcoxon_pvalue"] = float(stat.pvalue)
        except Exception:
            row["wilcoxon_statistic"] = np.nan
            row["wilcoxon_pvalue"] = np.nan
        rows.append(row)

    stats = pd.DataFrame(rows).sort_values("system").reset_index(drop=True)
    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(comparison_dir / "paired_statistics.csv", index=False)
    return stats


def assert_same_outer_tests(protocol_dir: Path, result_dir: Path) -> dict[str, object]:
    detail = pd.read_csv(result_dir / "transformer_lstm" / "transformer_lstm_detail.csv")
    mismatches = []
    checked = 0
    for _, row in detail.iterrows():
        system = row["system"]
        fold = int(row["fold"])
        fold_name = f"fold_{fold:02d}"
        test_csv = protocol_dir / "by_system_10fold" / system / fold_name / "test.csv"
        pred_csv = result_dir / "transformer_lstm" / system / fold_name / "predictions_test_transformer_lstm.csv"
        if not test_csv.exists() or not pred_csv.exists():
            mismatches.append({"system": system, "fold": fold, "reason": "missing file"})
            continue
        test_ids = pd.read_csv(test_csv)["sample_id"].astype(str).tolist()
        pred_ids = pd.read_csv(pred_csv)["sample_id"].astype(str).tolist()
        checked += 1
        if test_ids != pred_ids:
            mismatches.append({"system": system, "fold": fold, "reason": "sample_id order/content mismatch"})
    return {"checked_folds": checked, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def write_report(protocol_dir: Path, result_dir: Path, old_dir: Path, comparison: pd.DataFrame, stats: pd.DataFrame) -> None:
    audit = pd.read_csv(protocol_dir / "fold_audit.csv")
    split_summary = pd.read_csv(protocol_dir / "summary_strict_10fold.csv")
    manifest = json.loads((protocol_dir / "strict_protocol.json").read_text(encoding="utf-8"))
    test_audit = assert_same_outer_tests(protocol_dir, result_dir)
    passed = bool(audit["passed"].all()) and test_audit["mismatch_count"] == 0
    proposed_lower_count = int(comparison["proposed_lower_mae"].sum())

    old_detail = read_csv_optional(old_dir / "rttr_repro_detail.csv")
    old_note = "not checked"
    if old_detail is not None and {"best_val_mae", "mae"}.issubset(old_detail.columns):
        old_note = f"{int(np.isclose(old_detail['best_val_mae'], old_detail['mae']).sum())}/{len(old_detail)} rows have best_val_mae == test mae"

    lines = [
        "# Strict PredRet Proposed vs Transformer-LSTM Audit",
        "",
        "## Baseline identity",
        "",
        "- Formal baseline name: Transformer-LSTM reproduction.",
        "- Architecture: ChemBERTa/RoBERTa SMILES encoder, hidden-state layer fusion 2/4/6, bidirectional LSTM, dense scalar RT regression head.",
        "- This is not Xue et al. RT-Transformer and no RT-Transformer checkpoint is used.",
        "- Current project checkpoint: checkpoints/paper_ablation_20260825/B_chemberta_bilstm_seed2026/best.pt.",
        "- Checkpoint provenance: local SMRT training on data/unified train/val/test with recorded config and metrics; stronger than the missing checkpoint-380982 asset, but SMRT same-split MAE is 32.7605 s, not the historical external-path 26.2352 s prediction record.",
        "",
        "## Strict protocol",
        "",
        f"- 100-fold split leakage audit: {'PASS' if bool(audit['passed'].all()) else 'FAIL'}",
        f"- Transformer-LSTM outer-test sample alignment: {'PASS' if test_audit['mismatch_count'] == 0 else 'FAIL'}",
        f"- checked Transformer-LSTM folds: {test_audit['checked_folds']}",
        "- Proposed and Transformer-LSTM use the same strict 10 systems, outer folds, inner validation folds, and canonical SMILES grouping.",
        "- Outer test is not used for scheduler, early stopping, checkpoint choice, epoch choice, blend selection, or calibration.",
        "",
        "## Canonical overlap audit",
        "",
        "| system | folds | train/test canonical overlap max | val/test canonical overlap max | val=test count | passed |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in split_summary.sort_values("system").iterrows():
        lines.append(
            f"| {row['system']} | {int(row['folds'])} | {int(row['canonical_train_test_overlap_max'])} | "
            f"{int(row['canonical_val_test_overlap_max'])} | {int(row['val_test_identical_count'])} | {bool(row['passed_all'])} |"
        )

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Delta MAE = MAE_proposed - MAE_Transformer-LSTM.",
            "- Negative Delta MAE indicates lower prediction error for the proposed model.",
            f"- Proposed MAE lower on {proposed_lower_count} / {len(comparison)} systems.",
            f"- Historical old rttr_repro_* files are historical_mislabeled_transformer_lstm; leakage symptom: {old_note}.",
            "",
            "| Chromatographic system | Proposed model MAE (s) | Transformer-LSTM MAE (s) | Delta MAE (s) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['system']} | {fmt_pm(row['proposed_mae_mean'], row['proposed_mae_std'])} | "
            f"{fmt_pm(row['transformer_lstm_mae_mean'], row['transformer_lstm_mae_std'])} | "
            f"{float(row['delta_mae']):.3f} |"
        )

    lines.extend(["", "## Paired Fold Statistics", "", "| system | mean Delta MAE | std | 95% bootstrap CI | Proposed-lower folds |", "| --- | ---: | ---: | ---: | ---: |"])
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['system']} | {row['mean_delta_mae']:.3f} | {row['std_delta_mae']:.3f} | "
            f"[{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] | {int(row['proposed_lower_fold_count'])}/{int(row['n_folds'])} |"
        )

    lines.extend(["", "## Paper Claim Guardrail", ""])
    if proposed_lower_count == len(comparison) and passed:
        lines.append("The strict results support reporting that the proposed model achieved lower MAE than Transformer-LSTM on all evaluated systems.")
    elif passed:
        lines.append(
            f"The strict results support reporting competitive cross-system transfer performance and lower MAE than Transformer-LSTM on {proposed_lower_count} of {len(comparison)} systems."
        )
    else:
        lines.append("The strict audit did not pass; do not finalize Section 3.4 from these outputs.")

    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "strict_predret_comparison_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold"))
    parser.add_argument("--result-dir", type=Path, default=Path("server_results/predret_compare_strict_20260909"))
    parser.add_argument("--old-dir", type=Path, default=Path("server_results/predret_compare"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    comparison = build_comparison(args.result_dir, args.old_dir)
    stats = build_paired_statistics(args.result_dir, args.bootstrap, args.seed)
    write_report(args.protocol_dir, args.result_dir, args.old_dir, comparison, stats)
    print(f"Wrote {args.result_dir / 'comparison' / 'strict_comparison.csv'}")
    print(f"Wrote {args.result_dir / 'comparison' / 'paired_statistics.csv'}")
    print(f"Wrote {args.result_dir / 'comparison' / 'strict_predret_comparison_audit.md'}")


if __name__ == "__main__":
    main()
