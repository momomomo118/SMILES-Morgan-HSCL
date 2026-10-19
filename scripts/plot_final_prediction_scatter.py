"""Plot every saved independent-test prediction of the final PTDF model."""
from pathlib import Path
import sys
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work' / 'figure2_plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

FONT = Path('C:/Windows/Fonts/simhei.ttf')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
    'font.size': 11,
    'axes.unicode_minus': False,
    'svg.fonttype': 'path',
    'pdf.fonttype': 42,
})

SOURCE = ROOT / 'outputs/final_fusion_ccrl_gnn/predictions_test_final_ccrl_gnn_ptdf.csv'
OUT = ROOT / 'deliverables/aic_prediction_scatter'
OUT.mkdir(parents=True, exist_ok=True)
PALETTE = {
    'blue': '#33539E',
    'light_blue': '#7FACD6',
    'lavender': '#BFB8DA',
    'pink': '#E8B7D4',
    'purple': '#A5678E',
}
with SOURCE.open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 7798
assert len({row['sample_id'] for row in rows}) == len(rows)
truth = np.array([float(row['true_rt']) for row in rows])
pred = np.array([float(row['pred_final_ccrl_gnn_ptdf']) for row in rows])
assert np.all(np.isfinite(truth)) and np.all(np.isfinite(pred))
branches = np.array([[float(row[c]) for c in ['ccrl_ensemble', 'g2026', 'g3407', 'g777']] for row in rows])
assert np.allclose(branches @ np.array([.30, .24, .18, .28]), pred, atol=1e-9, rtol=0)
error = pred - truth
metrics = {
    'n': len(rows),
    'mae_s': float(np.abs(error).mean()),
    'rmse_s': float(np.sqrt(np.mean(error ** 2))),
    'r2': float(1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2)),
    'true_min_s': float(truth.min()), 'true_max_s': float(truth.max()),
    'pred_min_s': float(pred.min()), 'pred_max_s': float(pred.max()),
    'source': str(SOURCE),
    'prediction_column': 'pred_final_ccrl_gnn_ptdf',
    'selection': 'all 7798 rows, no filtering or subsampling',
}

upper = math.ceil(max(truth.max(), pred.max()) / 200) * 200
lower = min(0, math.floor(min(truth.min(), pred.min()) / 200) * 200)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
fig.subplots_adjust(left=.14, right=.97, bottom=.12, top=.88)
fig.suptitle('最终融合模型预测值与真实值散点图', fontsize=17, y=.975,
             color=PALETTE['blue'])
ax.set_title('SMRT独立测试集', fontsize=11, color=PALETTE['purple'], pad=13)
ax.set_axisbelow(True)
ax.grid(True, color=PALETTE['light_blue'], alpha=.27, linewidth=.65)
ax.scatter(truth, pred, s=10, alpha=.32, color=PALETTE['blue'],
           edgecolors='none', label='测试样本', zorder=2)
ax.plot([lower, upper], [lower, upper], '--', color=PALETTE['purple'],
        linewidth=1.5, dashes=(5, 3), label='理想预测（y = x）', zorder=3)
ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('真实保留时间（秒）', fontsize=13, labelpad=10)
ax.set_ylabel('预测保留时间（秒）', fontsize=13, labelpad=10)
tick_step = 200 if upper - lower <= 2000 else 400
ax.xaxis.set_major_locator(MultipleLocator(tick_step))
ax.yaxis.set_major_locator(MultipleLocator(tick_step))
ax.tick_params(direction='out', length=4, color=PALETTE['blue'])
for spine in ax.spines.values():
    spine.set_linewidth(.85)
    spine.set_color(PALETTE['blue'])
summary = (f"测试样本：{len(rows):,}\n"
           f"MAE = {metrics['mae_s']:.2f} 秒\n"
           f"RMSE = {metrics['rmse_s']:.2f} 秒\n"
           f"$R^2$ = {metrics['r2']:.4f}")
ax.text(.035, .965, summary, transform=ax.transAxes,
        ha='left', va='top', fontsize=11, linespacing=1.65,
        color=PALETTE['blue'],
        bbox={'boxstyle':'square,pad=0.65', 'facecolor':'#FCF7FA',
              'edgecolor':PALETTE['pink'], 'linewidth':.85, 'alpha':.97}, zorder=5)
ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=.95,
          edgecolor=PALETTE['lavender'], markerscale=2)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(OUT / f'final_prediction_vs_true_palette.{ext}', dpi=450, facecolor='white')
plt.close(fig)
metrics['palette'] = PALETTE
(OUT / 'figure_metrics_palette.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False))
print(str(OUT / 'final_prediction_vs_true_palette.png'))
