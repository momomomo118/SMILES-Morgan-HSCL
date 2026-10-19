from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"D:/SMILES_Morgan_HSCL_REPRO_26p05")
OUT = ROOT / "figures_vector_split_language"
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
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8.5, color="#1F2937")


def load_ablation():
    return pd.read_csv(ROOT / "outputs" / "ablation_lowcost_26p05" / "ablation_lowcost_valid_only.csv")


def value(df, name):
    sub = df[df["name"] == name]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["mae"])


def labels(lang, zh, en):
    return zh if lang == "zh" else en


def fig_model_performance(df, lang):
    names = [
        "SMILES-only", "Morgan-only", "Top6 calibrated ensemble",
        "+ RDKit/GBDT blend", "PureGIN seed777", "Final CCRL ensemble + GNN PTDF",
    ]
    zh = ["SMILES\n单模态", "Morgan\n单模态", "Top6\n主干集成", "RDKit/GBDT\n校正", "PureGIN\n图分支", "最终\n模型"]
    en = ["SMILES-\nonly", "Morgan-\nonly", "Top6\nensemble", "RDKit/GBDT\nblend", "PureGIN\nbranch", "Final\nmodel"]
    vals = [value(df, n) for n in names]
    fig, ax = plt.subplots(figsize=(9.0, 5.35))
    bars = ax.bar(range(len(vals)), vals, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4], COLORS[0]], width=0.64)
    bars[-1].set_edgecolor("#172554")
    bars[-1].set_linewidth(1.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels(lang, zh, en), rotation=0, ha="center")
    ax.set_ylabel(labels(lang, "MAE / 秒", "MAE / s"), fontsize=10)
    ax.set_title(labels(lang, "模型性能对比", "Model Performance Comparison"), fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.set_ylim(0, max(vals) * 1.18)
    annotate_bars(ax, bars, dy=max(vals) * 0.015)
    clean_axes(ax)
    fig.text(0.5, 0.018, labels(lang, "数值越低越好", "Lower is better"), ha="center", fontsize=8.5, color="#64748B")
    fig.tight_layout(rect=[0.04, 0.09, 0.99, 0.95])
    savefig(f"01_model_performance_comparison_{lang}.svg")


def fig_pipeline(df, lang):
    names = [
        "previous best ensemble (reported)", "Top6 calibrated ensemble", "+ OOF residual",
        "+ RDKit/GBDT blend", "+ GNN residual", "Final CCRL ensemble + GNN PTDF",
    ]
    zh = ["旧最佳", "Top6主干", "OOF残差", "RDKit/GBDT", "GNN残差", "最终模型"]
    en = ["Previous best", "Top6 ensemble", "OOF residual", "RDKit/GBDT", "GNN residual", "Final model"]
    vals = [value(df, n) for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(vals))
    ax.plot(x, vals, color=PALETTE["blue"], linewidth=2.4, marker="o", markersize=7)
    ax.fill_between(x, vals, [max(vals) + 0.7] * len(vals), color=PALETTE["sky"], alpha=0.16)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.16, f"{v:.2f}", ha="center", va="top", fontsize=8.5, color="#1F2937")
    ax.annotate(labels(lang, "降低 4.24 秒", "-4.24 s"),
                xy=(len(vals) - 1, vals[-1]),
                xytext=(len(vals) - 2.35, vals[-1] + 2.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["mauve"], lw=1.6),
                fontsize=9.5, color=PALETTE["mauve"], ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels(lang, zh, en))
    ax.set_ylabel(labels(lang, "MAE / 秒", "MAE / s"), fontsize=10)
    ax.set_title(labels(lang, "主线增量提升", "Incremental Improvement"), fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.set_ylim(min(vals) - 1.0, max(vals) + 3.0)
    clean_axes(ax)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.95])
    savefig(f"02_incremental_improvement_{lang}.svg")


def fig_modality_ablation(df, lang):
    names = [
        "SMILES-only", "Morgan-only", "SMILES+Morgan residual cross-attention",
        "+ RDKit/GBDT blend", "PureGIN seed777", "Final CCRL ensemble + GNN PTDF",
    ]
    zh = ["SMILES\n序列", "Morgan\n指纹", "SMILES+Morgan\n主干", "RDKit\n描述符", "图分支", "完整模型"]
    en = ["SMILES", "Morgan", "SMILES+Morgan\nbackbone", "RDKit\ndescriptors", "Graph branch", "Full model"]
    vals = [value(df, n) for n in names]
    fig, ax = plt.subplots(figsize=(9.4, 5.35))
    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=[PALETTE["blue"], PALETTE["sky"], PALETTE["lavender"], PALETTE["pink"], PALETTE["mauve"], PALETTE["blue"]], height=0.62)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(labels(lang, "MAE / 秒", "MAE / s"), fontsize=10)
    ax.set_title(labels(lang, "输入模态消融", "Input Modality Ablation"), fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    label_x = -max(vals) * 0.07
    for yi, lab in zip(y, labels(lang, zh, en)):
        ax.text(label_x, yi, lab, ha="right", va="center", fontsize=9.5, color="#1F2937", linespacing=1.2)
    for b in bars:
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=8.8, color="#1F2937")
    ax.set_xlim(-max(vals) * 0.30, max(vals) * 1.16)
    clean_axes(ax)
    ax.grid(axis="x", color="#E8EDF5", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0.04, 0.04, 0.99, 0.95])
    savefig(f"03_input_modality_ablation_{lang}.svg")


def fig_prediction_scatter(lang):
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    y = pred["true_rt"].to_numpy(float)
    p = pred["pred_final"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.6, 6.15))
    ax.scatter(y, p, s=8, color=PALETTE["blue"], alpha=0.28, linewidths=0)
    lim_min = min(y.min(), p.min()) - 30
    lim_max = max(y.max(), p.max()) + 30
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=PALETTE["mauve"], lw=1.8, label=labels(lang, "理想线", "Ideal line"))
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(labels(lang, "真实保留时间 / 秒", "True RT / s"), fontsize=10)
    ax.set_ylabel(labels(lang, "预测保留时间 / 秒", "Predicted RT / s"), fontsize=10)
    ax.set_title(labels(lang, "最终模型预测散点", "Final Prediction Scatter"), fontsize=13.5, weight="bold", color=PALETTE["blue"], pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=8.8, borderaxespad=0.0)
    clean_axes(ax)
    fig.tight_layout(rect=[0.06, 0.06, 0.98, 0.95])
    savefig(f"04_prediction_scatter_{lang}.svg")


def fig_error_distribution(lang):
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    ae = np.abs(pred["pred_final"].to_numpy(float) - pred["true_rt"].to_numpy(float))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bins = np.linspace(0, np.quantile(ae, 0.995), 42)
    ax.hist(ae, bins=bins, color=PALETTE["sky"], edgecolor="white", linewidth=0.6)
    median = float(np.median(ae))
    p95 = float(np.quantile(ae, 0.95))
    ax.axvline(median, color=PALETTE["blue"], lw=2.0, label=labels(lang, f"中位误差 = {median:.2f}秒", f"MedAE = {median:.2f}s"))
    ax.axvline(p95, color=PALETTE["mauve"], lw=2.0, linestyle="--", label=f"P95 = {p95:.2f}s")
    ax.set_xlabel(labels(lang, "绝对误差 / 秒", "Absolute Error / s"), fontsize=10)
    ax.set_ylabel(labels(lang, "样本数", "Count"), fontsize=10)
    ax.set_title(labels(lang, "绝对误差分布", "Absolute Error Distribution"), fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=8.8)
    clean_axes(ax)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.95])
    savefig(f"05_absolute_error_distribution_{lang}.svg")


def fig_rt_bin_error(lang):
    pred = pd.read_csv(ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv")
    pred["abs_error"] = (pred["pred_final"] - pred["true_rt"]).abs()
    pred["rt_bin"] = pd.qcut(pred["true_rt"], 10, duplicates="drop")
    g = pred.groupby("rt_bin", observed=False).agg(mae=("abs_error", "mean"), n=("abs_error", "size")).reset_index()
    xlabels = [f"{int(i.left)}-{int(i.right)}" for i in g["rt_bin"]]
    vals = g["mae"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    bars = ax.bar(np.arange(len(vals)), vals, color=[COLORS[i % len(COLORS)] for i in range(len(vals))], width=0.68)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_ylabel(labels(lang, "MAE / 秒", "MAE / s"), fontsize=10)
    ax.set_xlabel(labels(lang, "真实 RT 区间 / 秒", "True RT bins / s"), fontsize=10)
    ax.set_title(labels(lang, "不同 RT 区间误差", "Error by RT Range"), fontsize=14, weight="bold", color=PALETTE["blue"], pad=14)
    annotate_bars(ax, bars, dy=max(vals) * 0.015)
    ax.set_ylim(0, max(vals) * 1.22)
    clean_axes(ax)
    fig.tight_layout(rect=[0.04, 0.08, 0.99, 0.95])
    savefig(f"06_error_by_rt_bin_{lang}.svg")


def fig_mechanism(lang):
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(labels(lang, "模型整体机制", "Overall Model Mechanism"), fontsize=15, weight="bold", color=PALETTE["blue"], pad=10)

    def t(zh, en):
        return zh if lang == "zh" else en

    def box(x, y, w, h, text, fc, ec=None, color="#111827"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.4)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=color, linespacing=1.25)

    def arrow(x1, y1, x2, y2, color=PALETTE["blue"]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=color))

    box(0.35, 2.55, 1.25, 0.72, t("输入SMILES", "Input SMILES"), "#F8FAFC", PALETTE["blue"])
    box(2.15, 4.55, 1.65, 0.78, t("SMILES序列\nChemBERTa", "SMILES sequence\nChemBERTa"), PALETTE["sky"])
    box(2.15, 3.35, 1.65, 0.78, t("Morgan指纹", "Morgan fingerprint"), PALETTE["lavender"])
    box(2.15, 2.15, 1.65, 0.78, t("RDKit描述符", "RDKit descriptors"), PALETTE["pink"])
    box(2.15, 0.95, 1.65, 0.78, t("分子图", "Molecular graph"), PALETTE["mauve"])
    box(4.55, 3.75, 1.75, 0.88, t("SupCR多模态主干", "SupCR multimodal\nbackbone"), "#EEF4FB", PALETTE["blue"])
    box(4.55, 2.25, 1.75, 0.88, t("GBDT残差校正", "GBDT residual\ncorrection"), "#FBEFF6", PALETTE["mauve"])
    box(4.55, 0.72, 1.75, 0.88, t("PureGIN三种seed", "Three PureGIN seeds"), "#F2EFF9", PALETTE["mauve"])
    box(7.15, 2.35, 1.65, 0.92, t("堆叠集成", "Stacked ensemble"), "#F8FAFC", PALETTE["blue"])
    box(8.95, 2.35, 0.85, 0.92, t("RT预测\n25.89秒", "RT prediction\n25.89s"), PALETTE["blue"], PALETTE["blue"], color="white")
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
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.95])
    savefig(f"07_model_mechanism_{lang}.svg")


def main():
    df = load_ablation()
    for lang in ["zh", "en"]:
        fig_model_performance(df, lang)
        fig_pipeline(df, lang)
        fig_modality_ablation(df, lang)
        fig_prediction_scatter(lang)
        fig_error_distribution(lang)
        fig_rt_bin_error(lang)
        fig_mechanism(lang)

    with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "language"])
        for p in sorted(OUT.glob("*.svg")):
            writer.writerow([p.name, "Chinese" if p.stem.endswith("_zh") else "English"])


if __name__ == "__main__":
    main()


