from __future__ import annotations

import json
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_fusion_ccrl_gnn"
LEGACY_OUT = ROOT / "outputs" / "top6_pool_target_power_ensemble"


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    err = [p - y for y, p in zip(y_true, y_pred)]
    abs_err = [abs(e) for e in err]
    y_mean = sum(y_true) / len(y_true)
    ss_res = float(sum(e * e for e in err))
    ss_tot = float(sum((y - y_mean) ** 2 for y in y_true))
    return {
        "mae": float(sum(abs_err) / len(abs_err)),
        "rmse": float(math.sqrt(sum(e * e for e in err) / len(err))),
        "r2": float(1.0 - ss_res / ss_tot),
        "median_abs_error": float(quantile(abs_err, 0.5)),
        "p95_abs_error": float(quantile(abs_err, 0.95)),
        "max_abs_error": float(max(abs_err)),
    }


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    print(f"OK {path.relative_to(ROOT)}")


def main() -> None:
    required = [
        ROOT / "data" / "unified" / "train.csv",
        ROOT / "data" / "unified" / "val.csv",
        ROOT / "data" / "unified" / "test.csv",
        OUT / "metrics_final_ccrl_gnn_ptdf.json",
        OUT / "predictions_test_final_ccrl_gnn_ptdf.csv",
        OUT / "summary_final_ccrl_gnn_ptdf.csv",
        OUT / "ablation_summary_final_ccrl_gnn_ptdf.csv",
        LEGACY_OUT / "predictions_val_calibrated.csv",
        LEGACY_OUT / "predictions_test_calibrated.csv",
        ROOT / "checkpoints" / "gnn" / "pure_gin_seed2026" / "best_pure_gin.pt",
        ROOT / "checkpoints" / "gnn" / "pure_gin_seed3407" / "best_pure_gin.pt",
        ROOT / "checkpoints" / "gnn" / "pure_gin_seed777" / "best_pure_gin.pt",
        ROOT / "code" / "train_pure_gnn_for_ensemble.py",
        ROOT / "scripts" / "run_final_fusion_ccrl_gnn.py",
        ROOT / "pretrained" / "ChemBERTa-zinc-base-v1" / "config.json",
    ]
    for path in required:
        require(path)

    saved = json.loads((OUT / "metrics_final_ccrl_gnn_ptdf.json").read_text(encoding="utf-8"))
    with (OUT / "predictions_test_final_ccrl_gnn_ptdf.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    true_col = "true_rt" if "true_rt" in rows[0] else "label"
    pred_col = "pred_final_ccrl_gnn_ptdf"
    y_true = [float(r[true_col]) for r in rows]
    y_pred = [float(r[pred_col]) for r in rows]
    recomputed = metrics(y_true, y_pred)

    best = saved["final_test_metrics"]
    print("\nSaved final metrics:")
    print(json.dumps(best, indent=2))
    print("\nRecomputed from final prediction CSV:")
    print(json.dumps(recomputed, indent=2))
    print(f"\nFINAL Scheme A MAE = {best['mae']}")

    if abs(best["mae"] - recomputed["mae"]) > 1e-6:
        raise SystemExit("Metric mismatch: saved MAE and recomputed MAE differ.")
    print("\nPackage verification passed.")


if __name__ == "__main__":
    main()
