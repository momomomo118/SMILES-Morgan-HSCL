import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

base = Path("src/outputs/supcr_tuning/top6_pool_target_power_ensemble")
gnn = Path("src/outputs/gnn/pure_gin_seed2026")

val_base = pd.read_csv(base / "predictions_val_calibrated.csv")
test_best = pd.read_csv(base / "predictions_test_gbdt_blend.csv")
val_gnn = pd.read_csv(gnn / "predictions_val_gnn.csv")
test_gnn = pd.read_csv(gnn / "predictions_test_gnn.csv")

out_pred = base / "predictions_test_val_selected_gnn_blend.csv"
out_metrics = base / "metrics_val_selected_gnn_blend.json"

def metrics(y, p):
    e = p - y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "r2": float(r2_score(y, p)),
        "median_abs_error": float(np.median(np.abs(e))),
        "p95_abs_error": float(np.quantile(np.abs(e), 0.95)),
        "max_abs_error": float(np.max(np.abs(e))),
    }

v = val_base[["sample_id", "true_rt", "pred_ensemble"]].rename(columns={"pred_ensemble": "pred_ccrl_ensemble"})
v = v.merge(val_gnn[["sample_id", "pred_rt"]].rename(columns={"pred_rt": "pred_gnn"}), on="sample_id", how="inner")

t = test_best[["sample_id", "smiles", "true_rt", "pred_rt"]].rename(columns={"pred_rt": "pred_residual_corrected"})
t = t.merge(test_gnn[["sample_id", "pred_rt"]].rename(columns={"pred_rt": "pred_gnn"}), on="sample_id", how="inner")

y_val = v["true_rt"].to_numpy()
base_val = v["pred_ccrl_ensemble"].to_numpy()
gnn_val = v["pred_gnn"].to_numpy()

y_test = t["true_rt"].to_numpy()
base_test = t["pred_residual_corrected"].to_numpy()
gnn_test = t["pred_gnn"].to_numpy()

best = None
for w in np.linspace(0, 1, 101):
    pred_val = w * base_val + (1 - w) * gnn_val
    m = metrics(y_val, pred_val)
    if best is None or m["mae"] < best["val_metrics"]["mae"]:
        pred_test = w * base_test + (1 - w) * gnn_test
        best = {
            "w_current_best": float(w),
            "w_gnn": float(1 - w),
            "val_metrics": m,
            "test_pred": pred_test,
        }

pred_test = best.pop("test_pred")
summary = {
    "note": "historical only: weight selected on validation using pred_ccrl_ensemble, applied to pred_residual_corrected test base plus GNN",
    "selected": best,
    "current_best_test": metrics(y_test, base_test),
    "gnn_test": metrics(y_test, gnn_test),
    "val_selected_blend_test": metrics(y_test, pred_test),
}

t["pred_rt"] = pred_test
t[["sample_id", "smiles", "true_rt", "pred_rt", "pred_residual_corrected", "pred_gnn"]].to_csv(out_pred, index=False)
out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("wrote:", out_pred)
