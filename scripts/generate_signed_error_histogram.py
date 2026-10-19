from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "final_fusion_ccrl_gnn" / "predictions_test_final.csv"
OUTPUT = ROOT / "deliverables" / "paper_experiment_package_2026-07-25" / "03_figures"

WIDTH, HEIGHT = 1200, 780
LEFT, RIGHT, TOP, BOTTOM = 118, 45, 86, 106
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM
X_MIN, X_MAX, BIN_WIDTH = -200.0, 200.0, 10.0


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(errors: np.ndarray, language: str) -> str:
    bins = np.arange(X_MIN, X_MAX + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(errors, bins=bins)
    max_count = int(counts.max())
    y_max = int(np.ceil(max_count / 100.0) * 100)
    mean = float(errors.mean())
    median = float(np.median(errors))
    in_range = int(((errors >= X_MIN) & (errors <= X_MAX)).sum())
    coverage = in_range / len(errors) * 100.0
    left_out = int((errors < X_MIN).sum())
    right_out = int((errors > X_MAX).sum())

    if language == "zh":
        title = "最终模型有符号预测误差分布"
        subtitle = "误差定义：预测保留时间 − 真实保留时间"
        x_label = "有符号误差 / 秒"
        y_label = "样本数"
        mean_label = f"平均误差 = {mean:.2f} 秒"
        median_label = f"中位误差 = {median:.2f} 秒"
        zero_label = "零误差"
        coverage_label = f"图示范围覆盖 {coverage:.2f}% 样本"
        overflow_label = f"< −200 秒：{left_out}；> 200 秒：{right_out}"
        desc = "SMRT 测试集最终模型预测值减真实值的有符号误差直方图，分布集中于零附近。"
    else:
        title = "Signed Prediction Error Distribution of the Final Model"
        subtitle = "Error = predicted retention time − true retention time"
        x_label = "Signed error / s"
        y_label = "Count"
        mean_label = f"Mean error = {mean:.2f} s"
        median_label = f"Median error = {median:.2f} s"
        zero_label = "Zero error"
        coverage_label = f"Displayed range covers {coverage:.2f}% of samples"
        overflow_label = f"< −200 s: {left_out}; > 200 s: {right_out}"
        desc = "Histogram of signed retention-time prediction errors on the SMRT test set, centered near zero."

    def x_pos(value: float) -> float:
        return LEFT + (value - X_MIN) / (X_MAX - X_MIN) * PLOT_W

    def y_pos(value: float) -> float:
        return TOP + PLOT_H - value / y_max * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title)}</title>",
        f"<desc id=\"desc\">{esc(desc)}</desc>",
        '<rect width="1200" height="780" fill="#ffffff"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","Arial",sans-serif;fill:#172033} .axis{stroke:#64748b;stroke-width:1.4} .grid{stroke:#e7ecf3;stroke-width:1} .tick{font-size:18px;fill:#475569} .label{font-size:22px;fill:#26364f} .legend{font-size:17px;fill:#334155}</style>',
        f'<text x="{WIDTH/2:.1f}" y="38" text-anchor="middle" font-size="28" font-weight="500" fill="#173f72">{esc(title)}</text>',
        f'<text x="{WIDTH/2:.1f}" y="68" text-anchor="middle" font-size="17" fill="#64748b">{esc(subtitle)}</text>',
    ]

    for tick in range(0, y_max + 1, 100):
        y = y_pos(tick)
        parts.append(f'<line class="grid" x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + PLOT_W}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{LEFT - 16}" y="{y + 6:.2f}" text-anchor="end">{tick}</text>')

    for tick in range(int(X_MIN), int(X_MAX) + 1, 50):
        x = x_pos(tick)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" y2="{TOP + PLOT_H}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{TOP + PLOT_H + 32}" text-anchor="middle">{tick}</text>')

    bar_width = PLOT_W / len(counts)
    for index, count in enumerate(counts):
        x = LEFT + index * bar_width + 1.2
        y = y_pos(float(count))
        height = TOP + PLOT_H - y
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2.4:.2f}" height="{height:.2f}" '
            'fill="#7fb7df" stroke="#ffffff" stroke-width="0.8"/>'
        )

    zero_x = x_pos(0.0)
    mean_x = x_pos(mean)
    median_x = x_pos(median)
    parts.extend(
        [
            f'<line x1="{zero_x:.2f}" y1="{TOP}" x2="{zero_x:.2f}" y2="{TOP + PLOT_H}" stroke="#475569" stroke-width="2"/>',
            f'<line x1="{mean_x:.2f}" y1="{TOP}" x2="{mean_x:.2f}" y2="{TOP + PLOT_H}" stroke="#155a9c" stroke-width="3"/>',
            f'<line x1="{median_x:.2f}" y1="{TOP}" x2="{median_x:.2f}" y2="{TOP + PLOT_H}" stroke="#8e669f" stroke-width="3" stroke-dasharray="9 7"/>',
            f'<line class="axis" x1="{LEFT}" y1="{TOP + PLOT_H}" x2="{LEFT + PLOT_W}" y2="{TOP + PLOT_H}"/>',
            f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + PLOT_H}"/>',
            f'<text class="label" x="{LEFT + PLOT_W/2:.2f}" y="{HEIGHT - 30}" text-anchor="middle">{esc(x_label)}</text>',
            f'<text class="label" transform="translate(34 {TOP + PLOT_H/2:.2f}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>',
        ]
    )

    legend_x, legend_y = LEFT + PLOT_W - 330, TOP + 28
    parts.extend(
        [
            f'<rect x="{legend_x - 18}" y="{legend_y - 24}" width="332" height="128" rx="8" fill="#ffffff" fill-opacity="0.93" stroke="#d9e1ea"/>',
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 42}" y2="{legend_y}" stroke="#475569" stroke-width="2"/>',
            f'<text class="legend" x="{legend_x + 54}" y="{legend_y + 6}">{esc(zero_label)}</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 34}" x2="{legend_x + 42}" y2="{legend_y + 34}" stroke="#155a9c" stroke-width="3"/>',
            f'<text class="legend" x="{legend_x + 54}" y="{legend_y + 40}">{esc(mean_label)}</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 68}" x2="{legend_x + 42}" y2="{legend_y + 68}" stroke="#8e669f" stroke-width="3" stroke-dasharray="9 7"/>',
            f'<text class="legend" x="{legend_x + 54}" y="{legend_y + 74}">{esc(median_label)}</text>',
        ]
    )
    parts.extend(
        [
            f'<text x="{LEFT + 12}" y="{TOP + 26}" font-size="17" fill="#475569">N = {len(errors):,}</text>',
            f'<text x="{LEFT + 12}" y="{TOP + 50}" font-size="16" fill="#64748b">{esc(coverage_label)}</text>',
            f'<text x="{LEFT + 12}" y="{TOP + 74}" font-size="16" fill="#64748b">{esc(overflow_label)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    frame = pd.read_csv(PREDICTIONS)
    errors = frame["pred_final"].to_numpy(dtype=float) - frame["true_rt"].to_numpy(dtype=float)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for language in ("zh", "en"):
        (OUTPUT / f"08_signed_error_histogram_{language}.svg").write_text(
            render_svg(errors, language),
            encoding="utf-8",
        )
    pd.DataFrame({"signed_error_s": errors}).to_csv(
        OUTPUT / "08_signed_error_histogram_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"N={len(errors)} mean={errors.mean():.6f} median={np.median(errors):.6f} "
        f"range=[{errors.min():.6f}, {errors.max():.6f}]"
    )


if __name__ == "__main__":
    main()
