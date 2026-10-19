from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "audit_outputs" / "final_ablation_visuals"
FIG_DIR = ROOT / "src" / "figures" / "final_ablation_cn"
DATA_DIR = ROOT / "data" / "unified"

SERVER_RESULTS = ROOT / "server_results"
MULTIMODAL_DIR = ROOT / "src" / "outputs" / "multimodal"
TUNING_DIR = ROOT / "src" / "outputs" / "supcr_tuning"
BASELINE_DIR = ROOT / "audit_outputs" / "20260522_220337_stage2_validation" / "baselines"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


MODEL_SOURCES = [
    {
        "model": "SMILES单模态",
        "ablation_group": "single_modal",
        "source_path": SERVER_RESULTS / "test_metrics_smiles_only.json",
    },
    {
        "model": "Morgan单模态",
        "ablation_group": "single_modal",
        "source_path": SERVER_RESULTS / "test_metrics_morgan_only.json",
    },
    {
        "model": "拼接融合",
        "ablation_group": "fusion_structure",
        "source_path": MULTIMODAL_DIR / "test_metrics_concat.json",
    },
    {
        "model": "交叉注意力融合",
        "ablation_group": "fusion_structure",
        "source_path": MULTIMODAL_DIR / "test_metrics_cross_attention.json",
    },
    {
        "model": "残差交叉注意力",
        "ablation_group": "fusion_structure",
        "source_path": SERVER_RESULTS / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "残差交叉注意力 + SupCR",
        "ablation_group": "contrastive_loss",
        "source_path": TUNING_DIR / "exact_cool_light" / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "残差交叉注意力 + SupCR长训练",
        "ablation_group": "training_schedule",
        "source_path": TUNING_DIR / "exact_cool_light_long_200e" / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "残差交叉注意力 + SupCR TTA8",
        "ablation_group": "test_time_augmentation",
        "source_path": TUNING_DIR
        / "exact_cool_light_long_200e_tta"
        / "test_metrics_cross_attention_residual_tta8.json",
    },
    {
        "model": "残差交叉注意力 + SupCR集成",
        "ablation_group": "ensemble",
        "source_path": TUNING_DIR / "ensemble_aug035_huber_5seed" / "test_metrics_ensemble_best.json",
    },
    {
        "model": "目标变换 + SupCR seed777",
        "ablation_group": "target_transform",
        "source_path": TUNING_DIR / "target_power_seed777" / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "目标变换 + SupCR seed2025",
        "ablation_group": "target_transform",
        "source_path": TUNING_DIR / "target_power_seed2025" / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "目标变换 + SupCR seed1234",
        "ablation_group": "target_transform",
        "source_path": TUNING_DIR / "target_power_v1" / "test_metrics_cross_attention_residual.json",
    },
    {
        "model": "目标变换 + SupCR三种子集成",
        "ablation_group": "target_transform_ensemble",
        "source_path": TUNING_DIR / "target_power_3seed_ensemble" / "test_metrics_ensemble_equal.json",
    },
]


ORDER = [
    "拼接融合",
    "交叉注意力融合",
    "Morgan单模态",
    "SMILES单模态",
    "残差交叉注意力",
    "残差交叉注意力 + SupCR",
    "残差交叉注意力 + SupCR长训练",
    "残差交叉注意力 + SupCR TTA8",
    "残差交叉注意力 + SupCR集成",
    "目标变换 + SupCR seed1234",
    "目标变换 + SupCR seed2025",
    "目标变换 + SupCR seed777",
    "目标变换 + SupCR三种子集成",
]


GRADIENT_COLORS = ["#33539E", "#7FACD6", "#BFB8DA", "#E8B7D4", "#A5678E"]
COLORS = {
    "blue": "#33539E",
    "sky": "#7FACD6",
    "lavender": "#BFB8DA",
    "pink": "#E8B7D4",
    "mauve": "#A5678E",
    "light": "#F8F7FB",
    "dark": "#2D3446",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.family": "sans-serif",
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.edgecolor": "#D6DDE7",
            "axes.linewidth": 0.9,
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.frameon": False,
            "grid.color": "#E8EDF4",
            "grid.linewidth": 0.8,
            "svg.fonttype": "path",
        }
    )


def savefig(name: str) -> None:
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight", facecolor="white")
    plt.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close()


def load_model_results() -> pd.DataFrame:
    rows = []
    for item in MODEL_SOURCES:
        path = item["source_path"]
        if not path.exists():
            continue
        metrics = read_json(path)
        row = {
            "model": item["model"],
            "ablation_group": item["ablation_group"],
            "source_path": str(path.relative_to(ROOT)),
        }
        for key in ["mae", "rmse", "medae", "r2", "pearson", "within_30s", "within_60s", "within_120s"]:
            row[key] = metrics.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    order_map = {name: i for i, name in enumerate(ORDER)}
    return df.sort_values("model", key=lambda s: s.map(order_map)).reset_index(drop=True)


def load_supcr_sweep() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(TUNING_DIR.glob("*/test_metrics*.json")):
        name = metrics_path.parent.name
        if "ensemble" in name or "tta" in name:
            continue
        metrics = read_json(metrics_path)
        history_path = metrics_path.parent / "history_cross_attention_residual.json"
        best_epoch = np.nan
        best_val_mae = np.nan
        if history_path.exists():
            history = read_json(history_path)
            if history:
                best = min(history, key=lambda row: row.get("val_mae", float("inf")))
                best_epoch = best.get("epoch", np.nan)
                best_val_mae = best.get("val_mae", np.nan)
        rows.append(
            {
                "config": name,
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "test_mae": metrics.get("mae"),
                "test_rmse": metrics.get("rmse"),
                "test_r2": metrics.get("r2"),
                "test_pearson": metrics.get("pearson"),
                "within_120s": metrics.get("within_120s"),
                "source_path": str(metrics_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows).sort_values(["best_val_mae", "test_mae"]).reset_index(drop=True)


def load_external_baselines() -> pd.DataFrame:
    path = BASELINE_DIR / "baseline_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["split"].eq("test")].copy().sort_values("mae")


def load_predictions() -> tuple[pd.DataFrame, str]:
    candidates = [
        (
            TUNING_DIR / "target_power_3seed_ensemble" / "predictions_ensemble_equal.csv",
            "目标变换 + SupCR三种子集成",
        ),
        (
            TUNING_DIR / "ensemble_aug035_huber_5seed" / "predictions_ensemble_best.csv",
            "残差交叉注意力 + SupCR集成",
        ),
        (
            TUNING_DIR / "exact_cool_light_long_200e" / "predictions_cross_attention_residual.csv",
            "残差交叉注意力 + SupCR长训练",
        ),
        (
            TUNING_DIR / "exact_cool_light" / "predictions_cross_attention_residual.csv",
            "残差交叉注意力 + SupCR",
        ),
    ]
    for path, label in candidates:
        if path.exists():
            return pd.read_csv(path), label
    raise FileNotFoundError("No prediction CSV found for final visualizations.")


def plot_overall_metrics(results: pd.DataFrame) -> None:
    df = results.copy()
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    specs = [
        ("mae", "MAE (s)", "{:.1f}", None),
        ("rmse", "RMSE (s)", "{:.1f}", None),
        ("r2", "R2", "{:.3f}", (0.25, 0.95)),
        ("within_120s", "Within 120 s", "{:.1%}", (0.55, 1.0)),
    ]
    palette = [GRADIENT_COLORS[i % len(GRADIENT_COLORS)] for i in range(len(df))]
    for ax, (col, title, fmt, ylim) in zip(axes.ravel(), specs):
        bars = ax.bar(x, df[col], color=palette[: len(df)], width=0.64)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=28, ha="right")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, df[col].max() * 1.16)
        for bar, value in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("统一测试集消融实验结果", fontsize=15, fontweight="bold", y=1.02)
    savefig("01_总体消融指标对比")


def plot_incremental_improvement(results: pd.DataFrame) -> None:
    df = results.set_index("model")
    milestones = [
        "Morgan单模态",
        "SMILES单模态",
        "残差交叉注意力",
        "残差交叉注意力 + SupCR",
        "残差交叉注意力 + SupCR长训练",
        "残差交叉注意力 + SupCR集成",
        "目标变换 + SupCR seed777",
        "目标变换 + SupCR三种子集成",
    ]
    rows = []
    for name in milestones:
        if name in df.index:
            rows.append({"model": name, "mae": df.loc[name, "mae"], "within_120s": df.loc[name, "within_120s"]})
    m = pd.DataFrame(rows)

    best_single = min(
        [df.loc[name, "mae"] for name in ["Morgan单模态", "SMILES单模态"] if name in df.index],
        default=np.nan,
    )
    m["mae_reduction_vs_best_single"] = best_single - m["mae"]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.5))
    x = np.arange(len(m))
    bars = ax1.bar(x, m["mae"], color=[GRADIENT_COLORS[i % len(GRADIENT_COLORS)] for i in range(len(m))], width=0.58)
    ax1.set_ylabel("MAE (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(m["model"], rotation=22, ha="right")
    ax1.set_ylim(0, m["mae"].max() * 1.16)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)
    for bar, value in zip(bars, m["mae"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}", ha="center", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(x, m["mae_reduction_vs_best_single"], color=COLORS["mauve"], marker="o", linewidth=2.0, label="相对最佳单模态 MAE 降低")
    ax2.axhline(0, color="#AAB3BF", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("MAE reduction (s)")
    ax2.legend(loc="upper right")
    ax1.set_title("增量消融：从单模态到集成模型")
    savefig("02_增量消融MAE收益")


def plot_supcr_sweep(sweep: pd.DataFrame) -> None:
    df = sweep.dropna(subset=["best_val_mae", "test_mae"]).head(14).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), gridspec_kw={"width_ratios": [1.25, 1.0]})

    colors = [COLORS["mauve"] if name == "exact_cool_light" else GRADIENT_COLORS[i % len(GRADIENT_COLORS)] for i, name in enumerate(df["config"])]
    bars = axes[0].barh(df["config"], df["best_val_mae"], color=colors, alpha=0.9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Best validation MAE (s)")
    axes[0].set_title("SupCR 参数消融：按验证集 MAE 排序")
    axes[0].grid(axis="x")
    for bar, value in zip(bars, df["best_val_mae"]):
        axes[0].text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)

    axes[1].scatter(df["best_val_mae"], df["test_mae"], s=72, color=COLORS["mauve"], alpha=0.82)
    for _, row in df.iterrows():
        axes[1].text(row["best_val_mae"] + 0.07, row["test_mae"], row["config"], fontsize=7.2)
    axes[1].set_xlabel("Best validation MAE (s)")
    axes[1].set_ylabel("Test MAE (s)")
    axes[1].set_title("验证集与测试集表现关系")
    axes[1].grid(True)
    savefig("03_SupCR参数消融")


def plot_predictions(pred: pd.DataFrame, label: str, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), gridspec_kw={"width_ratios": [1.08, 1.0]})
    ax = axes[0]
    ax.scatter(pred["true_rt"], pred["pred_rt"], s=12, color=COLORS["blue"], alpha=0.42, edgecolors="none")
    lo = min(pred["true_rt"].min(), pred["pred_rt"].min())
    hi = max(pred["true_rt"].max(), pred["pred_rt"].max())
    ax.plot([lo, hi], [lo, hi], color=COLORS["mauve"], linewidth=1.6)
    ax.set_xlabel("True RT (s)")
    ax.set_ylabel("Predicted RT (s)")
    ax.set_title(f"{label}：预测值与真实值")
    ax.grid(True)
    ax.text(
        0.03,
        0.97,
        f"MAE {metrics['mae']:.2f} s\nRMSE {metrics['rmse']:.2f} s\nR2 {metrics['r2']:.4f}\nPearson {metrics['pearson']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D6DDE7", alpha=0.95),
    )

    abs_error = np.sort(pred["abs_error"].to_numpy())
    coverage = np.arange(1, len(abs_error) + 1) / len(abs_error) * 100
    axes[1].plot(abs_error, coverage, color=COLORS["sky"], linewidth=2.2)
    for threshold, key in [(30, "within_30s"), (60, "within_60s"), (120, "within_120s")]:
        value = metrics[key] * 100
        axes[1].axvline(threshold, color=COLORS["mauve"], linestyle="--", linewidth=1.1)
        axes[1].scatter([threshold], [value], color=COLORS["mauve"], s=45, zorder=5)
        axes[1].text(threshold + 3, value - 3, f"{value:.1f}%", fontsize=8.5, color=COLORS["mauve"])
    axes[1].set_xlim(0, min(260, abs_error.max()))
    axes[1].set_ylim(0, 101)
    axes[1].set_xlabel("Absolute error threshold (s)")
    axes[1].set_ylabel("Covered samples (%)")
    axes[1].set_title("误差覆盖率曲线")
    axes[1].grid(True)
    savefig("04_预测散点与误差覆盖")


def plot_error_diagnostics(pred: pd.DataFrame) -> None:
    pred = pred.copy()
    pred["rt_bin"] = pd.cut(pred["true_rt"], bins=np.linspace(300, 1500, 13), include_lowest=True)
    by_bin = pred.groupby("rt_bin", observed=True)["abs_error"].agg(["mean", "median", "count"]).reset_index()
    labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in by_bin["rt_bin"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1))
    axes[0].hist(pred["error"], bins=60, color=COLORS["blue"], alpha=0.82, edgecolor="white")
    axes[0].axvline(0, color=COLORS["dark"], linestyle="--", linewidth=1.2)
    axes[0].set_title("有符号误差分布")
    axes[0].set_xlabel("Prediction error (s)")
    axes[0].set_ylabel("Sample count")
    axes[0].grid(axis="y")

    axes[1].plot(labels, by_bin["median"], marker="o", color=COLORS["mauve"], linewidth=2.0, label="绝对误差中位数")
    axes[1].plot(labels, by_bin["mean"], marker="s", color=COLORS["sky"], linewidth=1.8, label="绝对误差均值")
    axes[1].set_title("不同保留时间区间的误差")
    axes[1].set_xlabel("True RT bin (s)")
    axes[1].set_ylabel("Absolute error (s)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True)
    axes[1].legend()
    savefig("05_误差分布与保留时间区间")

    by_bin.to_csv(OUT_DIR / "error_by_rt_bin.csv", index=False)


def plot_training_curves() -> None:
    candidates = [
        ("残差交叉注意力 + SupCR长训练", TUNING_DIR / "exact_cool_light_long_200e" / "history_cross_attention_residual.json"),
        ("残差交叉注意力 + SupCR", TUNING_DIR / "exact_cool_light" / "history_cross_attention_residual.json"),
        ("残差交叉注意力基线", SERVER_RESULTS / "history_cross_attention_residual.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    line_colors = [COLORS["blue"], COLORS["sky"], COLORS["mauve"]]
    for (label, path), color in zip(candidates, line_colors):
        if not path.exists():
            continue
        h = pd.DataFrame(read_json(path))
        if h.empty:
            continue
        axes[0].plot(h["epoch"], h["val_mae"], marker="o", markersize=3, linewidth=1.9, color=color, label=label)
        train_col = "train_task" if "train_task" in h else "train_mae_loss"
        if train_col in h:
            axes[1].plot(h["epoch"], h[train_col], marker="o", markersize=3, linewidth=1.7, color=color, label=label)
    axes[0].set_title("训练过程中的验证集 MAE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MAE (s)")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)
    axes[1].set_title("训练目标变化")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)
    savefig("06_训练曲线")


def plot_dataset_and_baselines(baselines: pd.DataFrame) -> None:
    split_frames = []
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["rt"])
            df["split"] = split
            split_frames.append(df)
    data = pd.concat(split_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1.0]})
    for split, color in zip(["train", "val", "test"], [COLORS["blue"], COLORS["sky"], COLORS["mauve"]]):
        values = data.loc[data["split"].eq(split), "rt"]
        axes[0].hist(values, bins=np.linspace(300, 1500, 50), density=True, alpha=0.36, color=color, label=f"{split} ({len(values):,})")
    axes[0].set_title("不同数据划分的保留时间分布")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")
    axes[0].grid(axis="y")
    axes[0].legend()

    if not baselines.empty:
        b = baselines.sort_values("mae", ascending=True)
        bars = axes[1].barh(b["baseline"], b["mae"], color=COLORS["lavender"], alpha=0.95)
        axes[1].invert_yaxis()
        axes[1].set_title("传统基线模型测试集结果")
        axes[1].set_xlabel("MAE (s)")
        axes[1].grid(axis="x")
        for bar, value in zip(bars, b["mae"]):
            axes[1].text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8.5)
    else:
        axes[1].axis("off")
    savefig("07_数据分布与传统基线")


def plot_architecture_summary() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.8))
    ax.axis("off")
    boxes = [
        ("SMILES\nChemBERTa 编码器", 0.08, 0.64, COLORS["blue"]),
        ("Morgan 指纹\nTransformer 编码器", 0.08, 0.24, COLORS["sky"]),
        ("跨模态\nSupCR 对齐", 0.37, 0.44, COLORS["lavender"]),
        ("残差交叉注意力\n融合", 0.63, 0.44, COLORS["pink"]),
        ("保留时间\n回归头", 0.86, 0.44, COLORS["mauve"]),
    ]
    for text, x, y, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.12", facecolor=color, edgecolor="none"),
            transform=ax.transAxes,
        )
    arrows = [
        ((0.19, 0.64), (0.31, 0.49)),
        ((0.19, 0.24), (0.31, 0.39)),
        ((0.45, 0.44), (0.55, 0.44)),
        ((0.72, 0.44), (0.80, 0.44)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2, color=COLORS["dark"]))
    ax.text(0.5, 0.93, "SMILES-Morgan HSCL 模型流程", ha="center", va="center", fontsize=15, fontweight="bold", color=COLORS["dark"], transform=ax.transAxes)
    savefig("08_模型流程图")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda value: format(value, floatfmt) if pd.notna(value) else "")
        else:
            text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(text_df.columns)
    rows = text_df.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), sep, *(fmt_row(row) for row in rows)])


def write_report(results: pd.DataFrame, sweep: pd.DataFrame, baselines: pd.DataFrame, pred_label: str, pred_metrics: dict) -> None:
    best = results.sort_values("mae").iloc[0]
    residual = results[results["model"].eq("残差交叉注意力")]
    smiles = results[results["model"].eq("SMILES单模态")]
    morgan = results[results["model"].eq("Morgan单模态")]
    best_single_mae = min(smiles["mae"].iloc[0], morgan["mae"].iloc[0])
    residual_mae = residual["mae"].iloc[0] if not residual.empty else np.nan

    lines = [
        "# 消融实验与可视化汇总",
        "",
        "本次整理复用了项目中已经完成的训练与测试产物，没有重新覆盖训练代码或模型权重。",
        "",
        "## 关键结论",
        f"- 当前最优结果为 **{best['model']}**：MAE={best['mae']:.2f}s，RMSE={best['rmse']:.2f}s，R2={best['r2']:.4f}，Within 120s={best['within_120s']:.2%}。",
        f"- 最优单模态 MAE 为 {best_single_mae:.2f}s；最终集成模型相对最优单模态降低 MAE {best_single_mae - best['mae']:.2f}s。",
        f"- 残差交叉注意力基线 MAE 为 {residual_mae:.2f}s；最终集成模型相对该基线降低 MAE {residual_mae - best['mae']:.2f}s。",
        f"- 用于预测散点图和误差诊断的明细结果来自 **{pred_label}**。",
        "",
        "## 输出文件",
        "- `ablation_results.csv`：单模态、融合结构、SupCR、TTA、集成的统一消融表。",
        "- `supcr_sweep_results.csv`：SupCR 超参数/训练配置扫参表，按验证 MAE 排序。",
        "- `external_baseline_results.csv`：传统 Morgan/RDKit 基线结果。",
        "- `figures/`：项目相关可视化图，PNG 和 SVG 各一份。",
        "",
        "## 图表清单",
    ]
    for path in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## 主要消融表",
            markdown_table(results[["model", "mae", "rmse", "medae", "r2", "pearson", "within_30s", "within_60s", "within_120s"]]),
            "",
            "## SupCR 扫参前 10",
            markdown_table(sweep.head(10)[["config", "best_epoch", "best_val_mae", "test_mae", "test_rmse", "test_r2", "within_120s"]]),
        ]
    )
    if not baselines.empty:
        lines.extend(["", "## 传统基线", markdown_table(baselines[["baseline", "mae", "rmse", "r2", "pearson"]])])

    (OUT_DIR / "ablation_visual_report.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "best_model": best.to_dict(),
        "prediction_detail_model": pred_label,
        "prediction_detail_metrics": pred_metrics,
        "figure_files": [str(path.relative_to(ROOT)) for path in sorted(FIG_DIR.glob("*.png"))],
    }
    (OUT_DIR / "visual_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    results = load_model_results()
    sweep = load_supcr_sweep()
    baselines = load_external_baselines()
    pred, pred_label = load_predictions()

    results.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    sweep.to_csv(OUT_DIR / "supcr_sweep_results.csv", index=False)
    baselines.to_csv(OUT_DIR / "external_baseline_results.csv", index=False)

    metric_row = results[results["model"].eq(pred_label)]
    if metric_row.empty and pred_label == "残差交叉注意力 + SupCR集成":
        pred_metrics = read_json(TUNING_DIR / "ensemble_aug035_huber_5seed" / "test_metrics_ensemble_best.json")
    else:
        pred_metrics = metric_row.iloc[0].to_dict()

    plot_overall_metrics(results)
    plot_incremental_improvement(results)
    plot_supcr_sweep(sweep)
    plot_predictions(pred, pred_label, pred_metrics)
    plot_error_diagnostics(pred)
    plot_training_curves()
    plot_dataset_and_baselines(baselines)
    plot_architecture_summary()
    write_report(results, sweep, baselines, pred_label, pred_metrics)

    print(f"Wrote final ablation package to {OUT_DIR}")


if __name__ == "__main__":
    main()
