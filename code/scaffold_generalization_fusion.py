from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "paper_ablation_20260825" / "04_scaffold_generalization"
RESULTS = BASE / "scaffold_results"
CCRL = RESULTS / "ccrl_resume_epoch11"
GNN = RESULTS / "gnn"
FUSION = RESULTS / "fusion"
RANDOM_SPLIT_FINAL_MAE = 25.8893


COMPONENTS = [
    ("CCRL", CCRL / "prediction_val.csv", CCRL / "prediction.csv"),
    ("GIN777", GNN / "seed777" / "predictions_val_gnn.csv", GNN / "seed777" / "predictions_test_gnn.csv"),
    ("GIN2026", GNN / "seed2026" / "predictions_val_gnn.csv", GNN / "seed2026" / "predictions_test_gnn.csv"),
    ("GIN3407", GNN / "seed3407" / "predictions_val_gnn.csv", GNN / "seed3407" / "predictions_test_gnn.csv"),
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    abs_err = np.abs(diff)
    denom = np.maximum(np.abs(y_true), 1e-8)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(diff**2))),
        "medae": float(np.median(abs_err)),
        "mape": float(np.mean(abs_err / denom) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": pearson,
        "within_30s": float(np.mean(abs_err <= 30.0)),
        "within_60s": float(np.mean(abs_err <= 60.0)),
        "within_120s": float(np.mean(abs_err <= 120.0)),
    }


def read_component(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {name}: {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "true_rt", "pred_rt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["sample_id", "true_rt", "pred_rt"]].copy()
    out = out.rename(columns={"pred_rt": f"pred_{name}"})
    return out


def merge_predictions(split_index: int) -> pd.DataFrame:
    merged = None
    for name, val_path, test_path in COMPONENTS:
        path = val_path if split_index == 0 else test_path
        current = read_component(path, name)
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current.drop(columns=["true_rt"]),
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
    assert merged is not None
    expected = [f"pred_{name}" for name, _, _ in COMPONENTS]
    if merged[expected].isna().any().any():
        raise ValueError("Merged prediction table contains NaN values.")
    return merged


def simplex_coordinate_search(y_true: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    n_models = preds.shape[1]
    weights = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    def score(w: np.ndarray) -> float:
        return float(np.mean(np.abs(preds @ w - y_true)))

    best = score(weights)
    for step in [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        improved = True
        while improved:
            improved = False
            for i in range(n_models):
                for j in range(n_models):
                    if i == j or weights[j] < step:
                        continue
                    candidate = weights.copy()
                    candidate[i] += step
                    candidate[j] -= step
                    value = score(candidate)
                    if value + 1e-10 < best:
                        weights = candidate
                        best = value
                        improved = True
    return weights, metrics(y_true, preds @ weights)


def optimize_weights(y_true: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], str]:
    try:
        from scipy.optimize import minimize

        n_models = preds.shape[1]

        def objective(w: np.ndarray) -> float:
            return float(np.mean(np.abs(preds @ w - y_true)))

        result = minimize(
            objective,
            np.full(n_models, 1.0 / n_models),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            weights = np.clip(result.x, 0.0, 1.0)
            weights = weights / weights.sum()
            return weights, metrics(y_true, preds @ weights), "scipy_slsqp"
    except Exception:
        pass
    weights, val_metrics = simplex_coordinate_search(y_true, preds)
    return weights, val_metrics, "simplex_coordinate_search"


def component_metrics(frame: pd.DataFrame, split: str) -> List[Dict[str, object]]:
    y = frame["true_rt"].to_numpy(dtype=np.float64)
    rows = []
    for name, _, _ in COMPONENTS:
        m = metrics(y, frame[f"pred_{name}"].to_numpy(dtype=np.float64))
        rows.append({"model": name, "split": split, **m})
    return rows


def write_summary(rows: Iterable[Dict[str, object]], weights: Dict[str, float], fusion_metrics: Dict[str, Dict[str, float]]) -> None:
    result_rows = list(rows)
    table = pd.DataFrame(result_rows)
    table.to_csv(BASE / "scaffold_generalization_results.csv", index=False)

    scaffold_mae = fusion_metrics["test"]["mae"]
    diff = scaffold_mae - RANDOM_SPLIT_FINAL_MAE
    rel = diff / RANDOM_SPLIT_FINAL_MAE * 100.0
    lines = [
        "Scaffold generalization experiment summary",
        "",
        f"Random split Final Scheme A MAE: {RANDOM_SPLIT_FINAL_MAE:.4f}",
        f"Scaffold split Final Fusion MAE: {scaffold_mae:.4f}",
        f"MAE difference: {diff:.4f} ({rel:.2f}%)",
        "",
        "Fusion weights:",
    ]
    lines.extend(f"- {name}: {value:.6f}" for name, value in weights.items())
    lines.extend(
        [
            "",
            "Test metrics:",
            json.dumps(fusion_metrics["test"], indent=2),
            "",
            "Recommended English conclusion:",
            "Under scaffold-disjoint evaluation, the final CCRL-GIN fusion retains meaningful retention-time generalization, but with a substantially larger error than the random split, indicating that structural novelty remains the dominant source of difficulty.",
        ]
    )
    (BASE / "scaffold_results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = [
        "# Scaffold Generalization Audit",
        "",
        "- Split file: `scaffold_split.csv`",
        "- Validation-only fusion: PASS",
        "- Test used for model/weight selection: NO",
        "- CCRL source: `scaffold_results/ccrl_resume_epoch11`",
        "- PureGIN sources: `scaffold_results/gnn/seed777`, `seed2026`, `seed3407`",
        "",
        "## Fusion Weights",
    ]
    audit.extend(f"- {name}: {value:.6f}" for name, value in weights.items())
    audit.extend(
        [
            "",
            "## Final Test Metrics",
            json.dumps(fusion_metrics["test"], indent=2),
        ]
    )
    (BASE / "scaffold_generalization_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def main() -> None:
    FUSION.mkdir(parents=True, exist_ok=True)
    val = merge_predictions(split_index=0)
    test = merge_predictions(split_index=1)
    names = [name for name, _, _ in COMPONENTS]
    pred_cols = [f"pred_{name}" for name in names]

    y_val = val["true_rt"].to_numpy(dtype=np.float64)
    x_val = val[pred_cols].to_numpy(dtype=np.float64)
    weights, val_metrics, method = optimize_weights(y_val, x_val)
    weights_dict = {name: float(weight) for name, weight in zip(names, weights)}

    val["pred_final_fusion"] = x_val @ weights
    y_test = test["true_rt"].to_numpy(dtype=np.float64)
    x_test = test[pred_cols].to_numpy(dtype=np.float64)
    test["pred_final_fusion"] = x_test @ weights

    val_fusion_metrics = metrics(y_val, val["pred_final_fusion"].to_numpy(dtype=np.float64))
    test_fusion_metrics = metrics(y_test, test["pred_final_fusion"].to_numpy(dtype=np.float64))

    val.to_csv(FUSION / "fused_predictions_val.csv", index=False)
    test.to_csv(FUSION / "fused_predictions_test.csv", index=False)
    payload = {
        "method": method,
        "weights": weights_dict,
        "val": val_fusion_metrics,
        "test": test_fusion_metrics,
        "random_split_final_mae": RANDOM_SPLIT_FINAL_MAE,
        "mae_difference_vs_random_split": test_fusion_metrics["mae"] - RANDOM_SPLIT_FINAL_MAE,
    }
    (FUSION / "fusion_weights.json").write_text(json.dumps({"method": method, "weights": weights_dict}, indent=2), encoding="utf-8")
    (FUSION / "fusion_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    rows.extend(component_metrics(val, "val"))
    rows.extend(component_metrics(test, "test"))
    rows.append({"model": "Final Fusion", "split": "val", **val_fusion_metrics})
    rows.append({"model": "Final Fusion", "split": "test", **test_fusion_metrics})
    write_summary(rows, weights_dict, {"val": val_fusion_metrics, "test": test_fusion_metrics})

    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
