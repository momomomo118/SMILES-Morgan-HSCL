"""Build the strict PredRet comparison table and Chinese audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def normalize_summary(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "variant" in df.columns:
        df = df.drop(columns=["variant"])
    keep = [
        "system",
        "n",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "medae_mean",
        "mape_mean",
        "r2_mean",
    ]
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


def build_comparison(result_dir: Path, old_dir: Path) -> pd.DataFrame:
    proposed = pd.read_csv(result_dir / "ours_full_ensemble_summary.csv")
    rttr = pd.read_csv(result_dir / "rttr_repro_summary.csv")
    comp = normalize_summary(proposed, "proposed").merge(normalize_summary(rttr, "rttr"), on="system", how="outer")
    comp["delta_mae"] = comp["proposed_mae_mean"] - comp["rttr_mae_mean"]
    comp["proposed_lower_mae"] = comp["delta_mae"] < 0

    old_proposed = read_csv_optional(old_dir / "ours_full_ensemble_summary.csv")
    old_rttr = read_csv_optional(old_dir / "rttr_repro_summary.csv")
    if old_proposed is not None:
        old_p = old_proposed[["system", "mae_mean"]].rename(columns={"mae_mean": "old_leaky_proposed_mae_mean"})
        comp = comp.merge(old_p, on="system", how="left")
        comp["proposed_mae_change_vs_old_leaky"] = comp["proposed_mae_mean"] - comp["old_leaky_proposed_mae_mean"]
    if old_rttr is not None:
        old_r = old_rttr[["system", "mae_mean"]].rename(columns={"mae_mean": "old_leaky_rttr_mae_mean"})
        comp = comp.merge(old_r, on="system", how="left")
        comp["rttr_mae_change_vs_old_leaky"] = comp["rttr_mae_mean"] - comp["old_leaky_rttr_mae_mean"]
    comp = comp.sort_values("system").reset_index(drop=True)
    comp.to_csv(result_dir / "strict_comparison.csv", index=False)
    return comp


def write_report(protocol_dir: Path, result_dir: Path, comparison: pd.DataFrame) -> None:
    audit = pd.read_csv(protocol_dir / "fold_audit.csv")
    summary = pd.read_csv(protocol_dir / "summary_strict_10fold.csv")
    manifest = json.loads((protocol_dir / "strict_protocol.json").read_text(encoding="utf-8"))

    total_rows = int(manifest["cleaned_rows"])
    systems = manifest["systems"]
    passed = bool(audit["passed"].all())
    proposed_lower_count = int(comparison["proposed_lower_mae"].sum())

    lines = [
        "# Strict PredRet 10-system / 10-fold leakage audit",
        "",
        "## 结论",
        "",
        f"- 新 protocol 泄漏状态：{'通过，无 sample 或 canonical SMILES overlap，且 val.csv 与 test.csv 不相同。' if passed else '未通过，请停止训练并检查 fold_audit.csv。'}",
        "- 外层 test 只用于最终 metrics；scheduler、early stopping、best checkpoint 和 blend weight selection 均限定为 inner-validation。",
        "- full ensemble 的 convex blend weights 在 inner-validation predictions / true_rt 上选择，冻结后应用到 outer-test。",
        "",
        "## 数据与 split",
        "",
        f"- cleaned rows: {total_rows}",
        f"- chromatographic systems: {len(systems)}",
        f"- canonicalization: {manifest.get('canonicalization_counts', {})}",
        f"- outer splitter: {manifest.get('splitter')}",
        "- inner validation: outer-train 内按 canonical SMILES group 约 90/10 划分。",
        "",
        "## Canonical overlap audit",
        "",
        "| system | folds | train/test canonical overlap max | val/test canonical overlap max | val=test count | passed |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.sort_values("system").iterrows():
        lines.append(
            f"| {row['system']} | {int(row['folds'])} | {int(row['canonical_train_test_overlap_max'])} | "
            f"{int(row['canonical_val_test_overlap_max'])} | {int(row['val_test_identical_count'])} | {bool(row['passed_all'])} |"
        )

    lines.extend(
        [
            "",
            "## Strict comparison",
            "",
            f"- Delta MAE 定义：MAE_proposed - MAE_RTTransformer；负数表示 proposed MAE 更低。",
            f"- proposed MAE 低于 RT-Transformer 的 system 数：{proposed_lower_count} / {len(comparison)}",
            "",
            "| system | proposed MAE | RTTransformer MAE | delta MAE | proposed lower | proposed MAE change vs old leaky |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for _, row in comparison.iterrows():
        change = row.get("proposed_mae_change_vs_old_leaky", pd.NA)
        change_text = "NA" if pd.isna(change) else f"{float(change):.3f}"
        lines.append(
            f"| {row['system']} | {fmt_pm(row['proposed_mae_mean'], row['proposed_mae_std'])} | "
            f"{fmt_pm(row['rttr_mae_mean'], row['rttr_mae_std'])} | {float(row['delta_mae']):.3f} | "
            f"{bool(row['proposed_lower_mae'])} | {change_text} |"
        )

    lines.extend(
        [
            "",
            "## Exact new Table 3 data",
            "",
            "| system | method | MAE mean +/- std | RMSE mean | MedAE mean | MAPE mean | R2 mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['system']} | Proposed full ensemble | {fmt_pm(row['proposed_mae_mean'], row['proposed_mae_std'])} | "
            f"{float(row['proposed_rmse_mean']):.3f} | {float(row['proposed_medae_mean']):.3f} | "
            f"{float(row['proposed_mape_mean']):.3f} | {float(row['proposed_r2_mean']):.3f} |"
        )
        lines.append(
            f"| {row['system']} | RT-Transformer | {fmt_pm(row['rttr_mae_mean'], row['rttr_mae_std'])} | "
            f"{float(row['rttr_rmse_mean']):.3f} | {float(row['rttr_medae_mean']):.3f} | "
            f"{float(row['rttr_mape_mean']):.3f} | {float(row['rttr_r2_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Generalization claim",
            "",
            "旧 CASE C 结果只能作为 historical/exploratory result。正式论文结论应基于本目录的 strict protocol 结果；若 strict 结果较旧结果变差，应报告为真实外推表现下降，不能回调 protocol 或使用 outer-test 重新选择权重。",
        ]
    )
    (result_dir / "strict_predret_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold"))
    parser.add_argument("--result-dir", type=Path, default=Path("server_results/predret_compare_strict_20260909"))
    parser.add_argument("--old-dir", type=Path, default=Path("server_results/predret_compare"))
    args = parser.parse_args()
    comparison = build_comparison(args.result_dir, args.old_dir)
    write_report(args.protocol_dir, args.result_dir, comparison)
    print(f"Wrote {args.result_dir / 'strict_comparison.csv'}")
    print(f"Wrote {args.result_dir / 'strict_predret_leakage_audit.md'}")


if __name__ == "__main__":
    main()
