from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "top6_pool_target_power_ensemble"
FINAL_SOURCE = ROOT / "outputs" / "final_fusion_ccrl_gnn"
OUTPUT = ROOT / "deliverables" / "paper_experiment_package_2026-07-25" / "07_strict_ablation"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260725

STAGES = [
    (
        "Top6 neural ensemble",
        "Top6 神经主干集成",
        SOURCE / "predictions_test_calibrated.csv",
        "pred_ensemble",
        "六个 SMILES-Morgan-SupCR 主干模型的验证集加权集成",
    ),
    (
        "+ OOF residual",
        "+ OOF 残差校正",
        SOURCE / "predictions_test_oof_residual.csv",
        "pred_rt",
        "只增加 OOF 残差校正",
    ),
    (
        "+ RDKit/GBDT blend",
        "+ RDKit/GBDT 融合",
        SOURCE / "predictions_test_gbdt_blend.csv",
        "pred_rt",
        "只增加验证集选择的 RDKit/GBDT 预测融合",
    ),
    (
        "+ val-selected PureGIN",
        "+ 验证集选择 PureGIN",
        SOURCE / "predictions_test_val_selected_gnn_blend.csv",
        "pred_rt",
        "只增加验证集选择权重的 PureGIN 图分支",
    ),
    (
        "+ GNN residual",
        "+ GNN 后残差校正",
        SOURCE / "predictions_test_gnn_blend_residual.csv",
        "pred_rt",
        "只增加图融合后的残差校正",
    ),
    (
        "Final CCRL ensemble + GNN PTDF",
        "Final CCRL ensemble + GNN PTDF",
        FINAL_SOURCE / "predictions_test_final.csv",
        "pred_final",
        "Scheme A: CCRL Top6 pred_ensemble plus g2026/g3407/g777 validation-optimized decision fusion",
    ),
]


def load_stage(path: Path, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "true_rt", prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    result = frame[["sample_id", "true_rt", prediction_column]].rename(
        columns={prediction_column: "prediction"}
    )
    if result["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {path}")
    return result.sort_values("sample_id").reset_index(drop=True)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    absolute = np.abs(error)
    return {
        "n": int(len(y)),
        "mae_s": float(absolute.mean()),
        "mape_pct": float(np.mean(absolute / np.abs(y)) * 100),
        "medae_s": float(np.median(absolute)),
        "rmse_s": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)),
        "p95_abs_error_s": float(np.quantile(absolute, 0.95)),
        "mean_signed_error_s": float(error.mean()),
    }


def paired_bootstrap(
    previous_absolute_error: np.ndarray,
    current_absolute_error: np.ndarray,
    seed: int,
) -> dict[str, float | bool]:
    # Positive values mean that the current stage reduces MAE.
    improvement = previous_absolute_error - current_absolute_error
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(improvement), size=len(improvement))
        sampled_means[index] = improvement[sample].mean()
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "mae_reduction_s": float(improvement.mean()),
        "paired_bootstrap_95ci_low_s": float(low),
        "paired_bootstrap_95ci_high_s": float(high),
        "statistically_clear_improvement_95pct": bool(low > 0),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for english, chinese, path, column, note in STAGES:
        frame = load_stage(path, column)
        loaded.append((english, chinese, path, note, frame))

    reference_ids = loaded[0][4]["sample_id"].to_numpy()
    reference_y = loaded[0][4]["true_rt"].to_numpy(dtype=float)
    for english, _, path, _, frame in loaded[1:]:
        if not np.array_equal(reference_ids, frame["sample_id"].to_numpy()):
            raise ValueError(f"Sample alignment differs at {english}: {path}")
        target_difference = np.max(
            np.abs(reference_y - frame["true_rt"].to_numpy(dtype=float))
        )
        if target_difference > 0.1:
            raise ValueError(f"Target mismatch {target_difference:.6f}s at {path}")

    result_rows = []
    predictions = []
    for index, (english, chinese, path, note, frame) in enumerate(loaded):
        prediction = frame["prediction"].to_numpy(dtype=float)
        predictions.append(prediction)
        row = {
            "stage_order": index + 1,
            "stage_en": english,
            "stage_zh": chinese,
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "controlled_change": note,
            **metrics(reference_y, prediction),
        }
        if index == 0:
            row.update(
                {
                    "mae_reduction_vs_previous_s": np.nan,
                    "previous_step_95ci_low_s": np.nan,
                    "previous_step_95ci_high_s": np.nan,
                    "previous_step_clear_95pct": np.nan,
                    "mae_reduction_vs_base_s": 0.0,
                    "base_95ci_low_s": 0.0,
                    "base_95ci_high_s": 0.0,
                    "base_clear_95pct": False,
                }
            )
        else:
            previous = paired_bootstrap(
                np.abs(predictions[index - 1] - reference_y),
                np.abs(prediction - reference_y),
                BOOTSTRAP_SEED + index,
            )
            versus_base = paired_bootstrap(
                np.abs(predictions[0] - reference_y),
                np.abs(prediction - reference_y),
                BOOTSTRAP_SEED + 100 + index,
            )
            row.update(
                {
                    "mae_reduction_vs_previous_s": previous["mae_reduction_s"],
                    "previous_step_95ci_low_s": previous[
                        "paired_bootstrap_95ci_low_s"
                    ],
                    "previous_step_95ci_high_s": previous[
                        "paired_bootstrap_95ci_high_s"
                    ],
                    "previous_step_clear_95pct": previous[
                        "statistically_clear_improvement_95pct"
                    ],
                    "mae_reduction_vs_base_s": versus_base["mae_reduction_s"],
                    "base_95ci_low_s": versus_base["paired_bootstrap_95ci_low_s"],
                    "base_95ci_high_s": versus_base[
                        "paired_bootstrap_95ci_high_s"
                    ],
                    "base_clear_95pct": versus_base[
                        "statistically_clear_improvement_95pct"
                    ],
                }
            )
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT / "strict_pipeline_ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT / "strict_pipeline_ablation_results.json").write_text(
        json.dumps(result_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if plt is None:
        print(f"Wrote {OUTPUT / 'strict_pipeline_ablation_results.csv'}")
        print(f"Wrote {OUTPUT / 'strict_pipeline_ablation_results.json'}")
        print("matplotlib is not installed; skipped strict ablation figure generation.")
        return

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#a7bed3", "#91b8d5", "#76a9d1", "#5f96c4", "#477eae", "#315f91"]
    labels = results["stage_zh"].tolist()
    mae = results["mae_s"].to_numpy()
    improvement = results["mae_reduction_vs_previous_s"].to_numpy()
    low = results["previous_step_95ci_low_s"].to_numpy()
    high = results["previous_step_95ci_high_s"].to_numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.7),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    y_positions = np.arange(len(labels))
    bars = axes[0].barh(y_positions, mae, color=colors, height=0.68)
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("测试集 MAE / 秒（越低越好）")
    axes[0].set_title("严格流水线增量消融：绝对性能", fontweight="bold")
    axes[0].set_xlim(min(mae) - 0.8, max(mae) + 1.4)
    axes[0].grid(axis="x", color="#e5eaf0", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, mae):
        axes[0].text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9.5,
        )

    step_labels = labels[1:]
    step_improvement = improvement[1:]
    xerr = np.vstack(
        [
            step_improvement - low[1:],
            high[1:] - step_improvement,
        ]
    )
    clear = results["previous_step_clear_95pct"].iloc[1:].astype(bool).to_numpy()
    point_colors = ["#315f91" if flag else "#9a6a5b" for flag in clear]
    axes[1].errorbar(
        step_improvement,
        np.arange(len(step_labels)),
        xerr=xerr,
        fmt="none",
        ecolor="#64748b",
        elinewidth=1.5,
        capsize=4,
    )
    axes[1].scatter(
        step_improvement,
        np.arange(len(step_labels)),
        c=point_colors,
        s=58,
        zorder=3,
    )
    axes[1].axvline(0, color="#475569", linewidth=1.2)
    axes[1].set_yticks(np.arange(len(step_labels)), step_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("相对前一步的 MAE 降低 / 秒（95% 配对 bootstrap CI）")
    axes[1].set_title("每个新增模块的边际贡献", fontweight="bold")
    axes[1].grid(axis="x", color="#e5eaf0", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for index, value in enumerate(step_improvement):
        axes[1].text(
            max(high[index + 1], value) + 0.04,
            index,
            f"{value:+.3f}",
            va="center",
            fontsize=9.3,
        )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "SMRT 最终预测流水线严格消融实验",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "同一固定测试集 N=7,798；每次只增加一个验证集已锁定的模块；蓝点表示 95% CI 下边界大于 0。",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.055, 0.99, 0.95])
    fig.savefig(OUTPUT / "strict_pipeline_ablation_zh.svg", bbox_inches="tight")
    fig.savefig(
        OUTPUT / "strict_pipeline_ablation_zh.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    markdown = [
        "# SMRT 最终预测流水线严格消融",
        "",
        "所有阶段使用同一固定测试集（N=7,798）和同一评价函数。每一步只增加一个已经在验证集上锁定的模块；测试标签没有参与权重或超参数选择。",
        "",
        "| 阶段 | MAE (s) | RMSE (s) | R² | 相对前一步 MAE 降低 (s) | 95% CI | 95% 明确改善 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in results.iterrows():
        if index == 0:
            delta, ci, clear_text = "—", "—", "—"
        else:
            delta = f"{row['mae_reduction_vs_previous_s']:.4f}"
            ci = (
                f"[{row['previous_step_95ci_low_s']:.4f}, "
                f"{row['previous_step_95ci_high_s']:.4f}]"
            )
            clear_text = "是" if bool(row["previous_step_clear_95pct"]) else "否"
        markdown.append(
            f"| {row['stage_zh']} | {row['mae_s']:.4f} | {row['rmse_s']:.4f} | "
            f"{row['r2']:.4f} | {delta} | {ci} | {clear_text} |"
        )
    markdown.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是最终预测流水线的严格增量消融，可用于证明 OOF 残差、RDKit/GBDT、PureGIN、GNN 后残差和多 GNN 集成的边际贡献。",
            "",
            "它不能单独证明 SMILES、Morgan、cross-attention 或 SupCR 的训练期因果贡献；这些内部表征模块仍需要同一训练预算和同一随机种子的重新训练消融。",
        ]
    )
    (OUTPUT / "strict_pipeline_ablation_report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

