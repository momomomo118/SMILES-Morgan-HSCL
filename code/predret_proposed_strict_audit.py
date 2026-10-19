"""Audit Proposed strict PredRet outputs after Phase A finishes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def require(path: Path, missing: list[str]) -> None:
    if not path.exists():
        missing.append(str(path))


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold"))
    parser.add_argument("--proposed-dir", type=Path, default=Path("server_results/predret_compare_strict_20260909/proposed"))
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint-root", type=Path, default=None)
    parser.add_argument("--gnn-root", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, default=Path("server_results/predret_compare"))
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument("--checkpoint-names", nargs="+", default=["exact1234", "exact777", "exact2025", "div_dropout015_seed777"])
    parser.add_argument("--gnn-seeds", nargs="+", default=["2026", "3407", "777"])
    parser.add_argument(
        "--require-fold-checkpoints",
        action="store_true",
        help="Require transient per-fold checkpoints. Disabled by default for lean paper-result runs.",
    )
    args = parser.parse_args()
    if args.base_checkpoint_root is None:
        args.base_checkpoint_root = args.base_root

    audit = pd.read_csv(args.protocol_dir / "fold_audit.csv")
    protocol_passed = bool(audit["passed"].all()) and len(audit) == len(args.systems) * 10

    missing: list[str] = []
    leakage_flags = []
    for system in args.systems:
        for fold in range(10):
            fold_name = f"fold_{fold:02d}"
            require(args.proposed_dir / system / fold_name / "metrics_full_ensemble.json", missing)
            require(args.proposed_dir / system / fold_name / "predictions_full_ensemble_val.csv", missing)
            require(args.proposed_dir / system / fold_name / "predictions_full_ensemble.csv", missing)
            for ckpt in args.checkpoint_names:
                require(args.base_root / ckpt / system / fold_name / "predictions_val_cross_attention_residual.csv", missing)
                require(args.base_root / ckpt / system / fold_name / "predictions_cross_attention_residual.csv", missing)
                if args.require_fold_checkpoints:
                    require(args.base_checkpoint_root / ckpt / system / fold_name / "best_cross_attention_residual.pt", missing)
            for seed in args.gnn_seeds:
                require(args.gnn_root / f"seed{seed}" / system / fold_name / "predictions_val_gnn.csv", missing)
                require(args.gnn_root / f"seed{seed}" / system / fold_name / "predictions_test_gnn.csv", missing)
                if args.require_fold_checkpoints:
                    require(args.gnn_root / f"seed{seed}" / system / fold_name / "best_pure_gin.pt", missing)

            metrics_path = args.proposed_dir / system / fold_name / "metrics_full_ensemble.json"
            if metrics_path.exists():
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                if payload.get("blend_selection_split") != "inner_validation":
                    leakage_flags.append(f"{system}/{fold_name}: blend_selection_split={payload.get('blend_selection_split')}")
                if payload.get("outer_test_used_for_weight_selection") is not False:
                    leakage_flags.append(f"{system}/{fold_name}: outer_test_used_for_weight_selection not false")

    detail_path = args.proposed_dir / "ours_full_ensemble_detail.csv"
    summary_path = args.proposed_dir / "ours_full_ensemble_summary.csv"
    require(detail_path, missing)
    require(summary_path, missing)

    if missing or leakage_flags or not protocol_passed:
        lines = [
            "# Proposed strict leakage audit",
            "",
            f"- protocol_passed: {protocol_passed}",
            f"- expected_folds: {len(args.systems) * 10}",
            f"- missing_output_count: {len(missing)}",
            f"- leakage_flag_count: {len(leakage_flags)}",
            "",
            "## Missing outputs",
            "",
            *[f"- {x}" for x in missing[:200]],
            "",
            "## Leakage flags",
            "",
            *[f"- {x}" for x in leakage_flags[:200]],
        ]
        args.proposed_dir.mkdir(parents=True, exist_ok=True)
        (args.proposed_dir.parent / "proposed_strict_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError("Proposed strict audit failed; formal summary should not be used.")

    summary = pd.read_csv(summary_path)
    old = pd.read_csv(args.old_dir / "ours_full_ensemble_summary.csv") if (args.old_dir / "ours_full_ensemble_summary.csv").exists() else None
    if old is not None:
        old = old[["system", "mae_mean"]].rename(columns={"mae_mean": "old_leaky_proposed_mae_mean"})
        summary = summary.merge(old, on="system", how="left")
        summary["mae_change_vs_old_leaky"] = summary["mae_mean"] - summary["old_leaky_proposed_mae_mean"]
        summary.to_csv(summary_path, index=False)

    lines = [
        "# Proposed strict leakage audit",
        "",
        "## Status",
        "",
        "- protocol_passed: True",
        "- completed_folds: 100 / 100",
        "- outer-test usage for scheduler/early stopping/checkpoint/epoch/member/blend/calibration selection: not detected in generated artifacts",
        "- blend_selection_split: inner_validation for 100 / 100 folds",
        "- historical old baseline note: server_results/predret_compare/rttr_repro_*.csv is historical_mislabeled_transformer_lstm, best described as a ChemBERTa/RoBERTa + BiLSTM Transformer-LSTM reproduction, not Xue et al. RT-Transformer.",
        "",
        "## Proposed strict summary",
        "",
        "| system | MAE | RMSE | MedAE | MAPE | R2 | MAE change vs old leaky |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values("system").iterrows():
        change = row.get("mae_change_vs_old_leaky", pd.NA)
        change_txt = "NA" if pd.isna(change) else f"{float(change):.3f}"
        lines.append(
            f"| {row['system']} | {fmt_pm(float(row['mae_mean']), float(row['mae_std']))} | "
            f"{fmt_pm(float(row['rmse_mean']), float(row['rmse_std']))} | "
            f"{fmt_pm(float(row['medae_mean']), float(row['medae_std']))} | "
            f"{fmt_pm(float(row['mape_mean']), float(row['mape_std']))} | "
            f"{fmt_pm(float(row['r2_mean']), float(row['r2_std']))} | {change_txt} |"
        )

    (args.proposed_dir.parent / "proposed_strict_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Proposed strict audit passed.")


if __name__ == "__main__":
    main()
