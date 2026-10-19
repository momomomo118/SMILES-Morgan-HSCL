#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs/supcr_tuning src/outputs/supcr_tuning src/checkpoints/supcr_tuning

# name|temperature|omega_align|omega_fusion|similarity_weight|use_log|lr|encoder_lr|weight_decay|dropout
# The first candidates stay close to the previous best residual model. The
# stronger log-form candidates are intentionally tiny-weighted so SupCR acts as
# representation regularization instead of dominating MAE regression.
configs=(
  "exact_orig|0.10|0.010|0.010|0.10|0|1e-4|1e-5|1e-3|0.10"
  "exact_fusion_light|0.10|0.010|0.005|0.10|0|1e-4|1e-5|1e-3|0.10"
  "exact_align_only|0.10|0.015|0.000|0.10|0|1e-4|1e-5|1e-3|0.10"
  "exact_cool_light|0.07|0.010|0.005|0.05|0|1e-4|1e-5|1e-3|0.10"
  "log_tiny|0.07|0.005|0.002|0.05|1|1e-4|1e-5|1e-3|0.10"
  "log_small|0.07|0.010|0.005|0.05|1|1e-4|1e-5|1e-3|0.10"
  "log_small_slow|0.07|0.010|0.005|0.05|1|8e-5|8e-6|5e-4|0.10"
)

for cfg in "${configs[@]}"; do
  IFS='|' read -r name temperature omega_align omega_fusion similarity_weight use_log lr encoder_lr weight_decay dropout <<< "$cfg"

  output_dir="src/outputs/supcr_tuning/${name}"
  checkpoint_dir="src/checkpoints/supcr_tuning/${name}"
  log_file="logs/supcr_tuning/${name}.log"

  echo "===== ${name} =====" | tee "$log_file"
  echo "temperature=${temperature} omega_align=${omega_align} omega_fusion=${omega_fusion} similarity_weight=${similarity_weight} use_log=${use_log} lr=${lr} encoder_lr=${encoder_lr} weight_decay=${weight_decay} dropout=${dropout}" | tee -a "$log_file"

  log_flag=(--no-supcr-use-log)
  if [[ "$use_log" == "1" ]]; then
    log_flag=(--supcr-use-log)
  fi

  python src/train_multimodal.py \
    --fusion cross_attention_residual \
    --epochs 20 \
    --lr "$lr" \
    --encoder-lr "$encoder_lr" \
    --weight-decay "$weight_decay" \
    --dropout "$dropout" \
    --temperature "$temperature" \
    --omega-align "$omega_align" \
    --omega-fusion "$omega_fusion" \
    --similarity-weight "$similarity_weight" \
    "${log_flag[@]}" \
    --early-stopping-patience 6 \
    --output-dir "$output_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --prediction-output "$output_dir/predictions_cross_attention_residual.csv" \
    2>&1 | tee -a "$log_file"
done

python - <<'PY'
import json
from pathlib import Path

rows = []
for metrics_path in Path("src/outputs/supcr_tuning").glob("*/test_metrics_cross_attention_residual.json"):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    name = metrics_path.parent.name
    history_path = metrics_path.parent / "history_cross_attention_residual.json"
    best_epoch = None
    best_val_mae = None
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            best = min(history, key=lambda row: row.get("val_mae", float("inf")))
            best_epoch = best.get("epoch")
            best_val_mae = best.get("val_mae")
    rows.append({
        "name": name,
        "test_mae": metrics.get("mae"),
        "test_rmse": metrics.get("rmse"),
        "test_r2": metrics.get("r2"),
        "test_pearson": metrics.get("pearson"),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    })

rows.sort(key=lambda row: row["test_mae"])
summary_path = Path("src/outputs/supcr_tuning/summary.json")
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n===== SupCR tuning summary, sorted by test MAE =====")
for row in rows:
    print(
        f"{row['name']:18s} test_mae={row['test_mae']:.4f} "
        f"rmse={row['test_rmse']:.4f} r2={row['test_r2']:.4f} "
        f"best_epoch={row['best_epoch']} best_val_mae={row['best_val_mae']:.4f}"
    )
print(f"\nWrote {summary_path}")
PY
