from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"D:/SMILES_Morgan_HSCL_REPRO_26p05")
OUT = ROOT / "figures_vector_bilingual"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "blue": "#33539E",
    "sky": "#7FACD6",
    "lavender": "#BFB8DA",
    "pink": "#E8B7D4",
    "mauve": "#A5678E",
}
COLORS = list(PALETTE.values())

mpl.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFC7D5")
    ax.spines["bottom"].set_color("#BFC7D5")
    ax.grid(axis="y", color="#E8EDF5", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#334155", labelsize=9)


def savefig(name: str):
    path = OUT / name
    plt.savefig(path, format="svg")
    plt.close()
    print(path)


def annotate_bars(ax, bars, fmt="{:.2f}", dy=0.35):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1F2937",
        )


def load_ablation():
    path = ROOT / "outputs" / "ablation_lowcost_26p05" / "ablation_lowcost_valid_only.csv"
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError("Run code/run_lowcost_ablation.py first.")


def fig_model_performance(df: pd.DataFrame):
    names = [
        "SMILES-only",
        "Morgan-only",
        "Top6 calibrated ensemble",
        "+ RDKit/GBDT blend",
        "PureGIN seed777",
        "Final CCRL ensemble + GNN PTDF",
    ]
    labels = [
        "SMILES-only\nSMILES单模态",
        "Morgan-only\nMorgan单模态",
        "Top6 ensemble\nTop6主干",
        "RDKit/GBDT\n描述符校正",
        "PureGIN\n图分支",
        "Final PTDF\n最终模型",
    ]
    vals = []
    for n in names:
        sub = df[df["name"] == n]
        vals.append(float(sub.iloc[0]["mae"]) if len(sub) else np.nan)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    bars = ax.bar(range(len(vals)), vals, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4], COLORS[0]], width=0.66)
    bars[-1].set_edgecolor("#172554")
    bars[-1].set_linewidth(1.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("MAE / 秒 (s)", fontsize=10, color="#1F2937")
    ax.set_title("模型性能对比 / Model Performance Comparison", fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.set_ylim(0, max(vals) * 1.18)
    annotate_bars(ax, bars, dy=max(vals) * 0.015)
    clean_axes(ax)
    fig.text(0.5, 0.015, "数值越低越好 / Lower is better", ha="center", fontsize=8.5, color="#64748B")
    fig.tight_layout(rect=[0.03, 0.05, 0.99, 0.95])
    savefig("01_model_performance_comparison.svg")


def fig_pipeline(df: pd.DataFrame):
    names = [
        "previous best ensemble (reported)",
        "Top6 calibrated ensemble",
        "+ OOF residual",
        "+ RDKit/GBDT blend",
        "+ GNN residual",
        "Final CCRL ensemble + GNN PTDF",
    ]
    labels = [
        "Previous best\n旧最佳",
        "Top6 ensemble\nTop6主干",
        "OOF residual\nOOF残差",
        "RDKit/GBDT\n描述符融合",
        "GNN residual\n图残差",
        "Final\n最终",
    ]
    vals = [float(df[df["name"] == n].iloc[0]["mae"]) for n in names if len(df[df["name"] == n])]
    labels = labels[: len(vals)]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(vals))
    ax.plot(x, vals, color=PALETTE["blue"], linewidth=2.4, marker="o", markersize=7)
    ax.fill_between(x, vals, [max(vals) + 0.7] * len(vals), color=PALETTE["sky"], alpha=0.16)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.16, f"{v:.2f}", ha="center", va="top", fontsize=8.5, color="#1F2937")
    ax.annotate(
        "降低 4.24 s / -4.24 s",
        xy=(len(vals) - 1, vals[-1]),
        xytext=(len(vals) - 2.35, vals[-1] + 2.2),
        arrowprops=dict(arrowstyle="->", color=PALETTE["mauve"], lw=1.6),
        fontsize=9.5,
        color=PALETTE["mauve"],
        ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE / 秒 (s)", fontsize=10)
    ax.set_title("主线增量提升 / Incremental Improvement", fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.set_ylim(min(vals) - 1.0, max(vals) + 3.0)
    clean_axes(ax)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.95])
    savefig("02_incremental_improvement.svg")


def fig_modality_ablation(df: pd.DataFrame):
    names = [
        "SMILES-only",
        "Morgan-only",
        "SMILES+Morgan residual cross-attention",
        "+ RDKit/GBDT blend",
        "PureGIN seed777",
        "Final CCRL ensemble + GNN PTDF",
    ]
    labels = [
        "SMILES\n序列",
        "Morgan\n指纹",
        "SMILES+Morgan\n多模态主干",
        "+ RDKit\n描述符",
        "Graph-only\n图分支",
        "Full model\n完整模型",
    ]
    vals = [float(df[df["name"] == n].iloc[0]["mae"]) for n in names if len(df[df["name"] == n])]
    labels = labels[: len(vals)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=[PALETTE["blue"], PALETTE["sky"], PALETTE["lavender"], PALETTE["pink"], PALETTE["mauve"], PALETTE["blue"]], height=0.62)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("MAE / 秒 (s)", fontsize=10)
    ax.set_title("输入模态消融 / Input Modality Ablation", fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    for b in bars:
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=8.8, color="#1F2937")
    ax.set_xlim(0, max(vals) * 1.16)
    clean_axes(ax)
    ax.grid(axis="x", color="#E8EDF5", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.95])
    savefig("03_input_modality_ablation.svg")


def fig_prediction_scatter():
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    y = pred["true_rt"].to_numpy(float)
    p = pred["pred_final"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(y, p, s=8, color=PALETTE["blue"], alpha=0.28, linewidths=0)
    lim_min = min(y.min(), p.min()) - 30
    lim_max = max(y.max(), p.max()) + 30
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=PALETTE["mauve"], lw=1.8, label="理想线 / Ideal y=x")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("真实 RT / True RT (s)", fontsize=10)
    ax.set_ylabel("预测 RT / Predicted RT (s)", fontsize=10)
    ax.set_title("最终模型预测散点 / Final Prediction Scatter", fontsize=13.5, weight="bold", color=PALETTE["blue"], pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8.8)
    clean_axes(ax)
    fig.text(0.58, 0.12, "MAE = 25.89 s\nR² = 0.9124", fontsize=10, color="#1F2937", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["lavender"], alpha=0.92))
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.95])
    savefig("04_prediction_scatter.svg")


def fig_error_distribution():
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    ae = np.abs(pred["pred_final"].to_numpy(float) - pred["true_rt"].to_numpy(float))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bins = np.linspace(0, np.quantile(ae, 0.995), 42)
    ax.hist(ae, bins=bins, color=PALETTE["sky"], edgecolor="white", linewidth=0.6)
    median = float(np.median(ae))
    p95 = float(np.quantile(ae, 0.95))
    ax.axvline(median, color=PALETTE["blue"], lw=2.0, label=f"MedAE 中位误差 = {median:.2f}s")
    ax.axvline(p95, color=PALETTE["mauve"], lw=2.0, linestyle="--", label=f"P95 = {p95:.02f}s")
    ax.set_xlabel("绝对误差 / Absolute Error (s)", fontsize=10)
    ax.set_ylabel("样本数 / Count", fontsize=10)
    ax.set_title("绝对误差分布 / Absolute Error Distribution", fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=8.8)
    clean_axes(ax)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.95])
    savefig("05_absolute_error_distribution.svg")


def fig_rt_bin_error():
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    pred["abs_error"] = (pred["pred_final"] - pred["true_rt"]).abs()
    pred["rt_bin"] = pd.qcut(pred["true_rt"], 10, duplicates="drop")
    g = pred.groupby("rt_bin", observed=False).agg(mae=("abs_error", "mean"), rt=("true_rt", "mean"), n=("abs_error", "size")).reset_index()
    labels = [f"{int(i.left)}-{int(i.right)}" for i in g["rt_bin"]]
    vals = g["mae"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    bars = ax.bar(np.arange(len(vals)), vals, color=[COLORS[i % len(COLORS)] for i in range(len(vals))], width=0.68)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MAE / 秒 (s)", fontsize=10)
    ax.set_xlabel("真实 RT 区间 / True RT bins (s)", fontsize=10)
    ax.set_title("不同 RT 区间误差 / Error by RT Range", fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    annotate_bars(ax, bars, dy=max(vals) * 0.015)
    ax.set_ylim(0, max(vals) * 1.22)
    clean_axes(ax)
    fig.tight_layout(rect=[0.04, 0.08, 0.99, 0.95])
    savefig("06_error_by_rt_bin.svg")


def fig_mechanism():
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("模型整体机制 / Overall Model Mechanism", fontsize=15, weight="bold", color=PALETTE["blue"], pad=10)

    def box(x, y, w, h, text, fc, ec=None):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color="#111827", linespacing=1.25)
        return patch

    def arrow(x1, y1, x2, y2, color=PALETTE["blue"]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=color))

    box(0.35, 2.55, 1.25, 0.72, "输入 SMILES\nInput", "#F8FAFC", PALETTE["blue"])
    box(2.15, 4.55, 1.65, 0.78, "SMILES 序列\nChemBERTa", PALETTE["sky"])
    box(2.15, 3.35, 1.65, 0.78, "Morgan 指纹\nFingerprint", PALETTE["lavender"])
    box(2.15, 2.15, 1.65, 0.78, "RDKit 描述符\nDescriptors", PALETTE["pink"])
    box(2.15, 0.95, 1.65, 0.78, "分子图\nMolecular graph", PALETTE["mauve"])

    box(4.55, 3.75, 1.75, 0.88, "SupCR 多模态主干\nSMILES-Morgan", "#EEF4FB", PALETTE["blue"])
    box(4.55, 2.25, 1.75, 0.88, "GBDT 残差校正\nRDKit residual", "#FBEFF6", PALETTE["mauve"])
    box(4.55, 0.72, 1.75, 0.88, "PureGIN 三个 seed\nGraph branch", "#F2EFF9", PALETTE["mauve"])

    box(7.15, 2.35, 1.65, 0.92, "堆叠集成\nStacked ensemble", "#F8FAFC", PALETTE["blue"])
    box(8.95, 2.35, 0.85, 0.92, "RT 预测\n25.89s", PALETTE["blue"], PALETTE["blue"])
    ax.text(9.375, 2.81, "RT 预测\n25.89s", ha="center", va="center", fontsize=9.4, color="white", linespacing=1.25)

    arrow(1.6, 2.91, 2.15, 4.94)
    arrow(1.6, 2.91, 2.15, 3.74)
    arrow(1.6, 2.91, 2.15, 2.54)
    arrow(1.6, 2.91, 2.15, 1.34)
    arrow(3.8, 4.94, 4.55, 4.19)
    arrow(3.8, 3.74, 4.55, 4.05)
    arrow(6.3, 4.19, 7.15, 2.98)
    arrow(3.8, 2.54, 4.55, 2.69, PALETTE["mauve"])
    arrow(6.3, 2.69, 7.15, 2.77, PALETTE["mauve"])
    arrow(3.8, 1.34, 4.55, 1.16, PALETTE["mauve"])
    arrow(6.3, 1.16, 7.15, 2.52, PALETTE["mauve"])
    arrow(8.8, 2.81, 8.95, 2.81)

    ax.text(5.45, 5.32, "监督对比学习保留在主干中\nSupCR remains in backbone", ha="center", fontsize=8.6, color="#475569")
    ax.text(5.45, 0.28, "Graph Transformer 未纳入最终模型\nGraph Transformer not included", ha="center", fontsize=8.2, color="#64748B")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.95])
    savefig("07_model_mechanism.svg")


def main():
    df = load_ablation()
    fig_model_performance(df)
    fig_pipeline(df)
    fig_modality_ablation(df)
    fig_prediction_scatter()
    fig_error_distribution()
    fig_rt_bin_error()
    fig_mechanism()
    with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "description"])
        for p in sorted(OUT.glob("*.svg")):
            writer.writerow([p.name, p.stem])


if __name__ == "__main__":
    main()


