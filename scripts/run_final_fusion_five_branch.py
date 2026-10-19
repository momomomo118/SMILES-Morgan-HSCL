import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "final_fusion_five_branch"

TOP6 = ROOT / "outputs" / "top6_pool_target_power_ensemble"
GNN_DIRS = {
    "g2026": ROOT / "outputs" / "pure_gin_seed2026",
    "g3407": ROOT / "outputs" / "pure_gin_seed3407",
    "g777": ROOT / "outputs" / "pure_gin_seed777",
}

BRANCHES = [
    {
        "branch": "CCRL Top6 ensemble prediction",
        "short": "ccrl_ensemble",
        "val_file": TOP6 / "predictions_val_calibrated.csv",
        "test_file": TOP6 / "predictions_test_calibrated.csv",
        "column": "pred_ensemble",
        "provenance": "CCRL Top6 ensemble; pred_ensemble from calibrated ensemble prediction files.",
    },
    {
        "branch": "Residual corrected CCRL prediction",
        "short": "residual_ccrl",
        "val_file": TOP6 / "predictions_val_calibrated.csv",
        "test_file": TOP6 / "predictions_test_calibrated.csv",
        "column": "pred_calibrated",
        "provenance": "Residual/calibrated CCRL correction; pred_calibrated from calibrated prediction files.",
    },
    {
        "branch": "GNN branch 1",
        "short": "g2026",
        "val_file": GNN_DIRS["g2026"] / "predictions_val_gnn.csv",
        "test_file": GNN_DIRS["g2026"] / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "Pure GIN GNN prediction, seed 2026.",
    },
    {
        "branch": "GNN branch 2",
        "short": "g3407",
        "val_file": GNN_DIRS["g3407"] / "predictions_val_gnn.csv",
        "test_file": GNN_DIRS["g3407"] / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "Pure GIN GNN prediction, seed 3407.",
    },
    {
        "branch": "GNN branch 3",
        "short": "g777",
        "val_file": GNN_DIRS["g777"] / "predictions_val_gnn.csv",
        "test_file": GNN_DIRS["g777"] / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "Pure GIN GNN prediction, seed 777.",
    },
]


def metric_dict(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
        "median_abs_error": float(np.median(np.abs(err))),
    }


def load_split(split):
    merged = None
    source_rows = []
    for spec in BRANCHES:
        path = spec[f"{split}_file"]
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"sample_id", "true_rt", spec["column"]}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        cols = ["sample_id", "true_rt", spec["column"]]
        if "smiles" in df.columns:
            cols.insert(1, "smiles")
        part = df[cols].copy()
        part = part.rename(
            columns={
                "true_rt": f"true_rt_{spec['short']}",
                spec["column"]: spec["short"],
            }
        )
        if merged is None:
            merged = part
        else:
            drop_cols = [c for c in ["smiles"] if c in part.columns and c in merged.columns]
            part = part.drop(columns=drop_cols)
            merged = merged.merge(part, on="sample_id", how="inner")
        source_rows.append(
            {
                "branch name": spec["branch"],
                "prediction source file": str(path.relative_to(ROOT)),
                "data provenance": spec["provenance"],
                "prediction column": spec["column"],
            }
        )

    true_cols = [c for c in merged.columns if c.startswith("true_rt_")]
    y0 = merged[true_cols[0]].to_numpy(dtype=float)
    for col in true_cols[1:]:
        if not np.allclose(y0, merged[col].to_numpy(dtype=float), atol=1e-3, rtol=0):
            raise ValueError(f"true_rt mismatch in {split}: {col}")
    merged["true_rt"] = y0
    merged = merged.drop(columns=true_cols)
    return merged, source_rows


def _mae(y, pred):
    return float(np.mean(np.abs(pred - y)))


def _project_simplex(v):
    v = np.asarray(v, dtype=float)
    if np.all(v >= 0.0) and abs(float(v.sum()) - 1.0) < 1e-12:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    theta = cssv[cond][-1] / ind[cond][-1]
    return np.maximum(v - theta, 0.0)


def _weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _optimize_simplex_mae_pairwise(y, x):
    n, k = x.shape
    branch_mae = np.array([_mae(y, x[:, i]) for i in range(k)])
    starts = [
        np.ones(k) / k,
        np.eye(k)[int(np.argmin(branch_mae))],
    ]
    # Include the two project baselines as warm starts when branch order matches BRANCHES.
    starts.extend(
        [
            np.array([0.30, 0.00, 0.24, 0.18, 0.28], dtype=float),
            np.array([0.00, 0.46, 0.20, 0.12, 0.22], dtype=float),
        ]
    )
    rng = np.random.default_rng(20260814)
    starts.extend(rng.dirichlet(np.ones(k), size=16))

    best_w = None
    best_mae = np.inf

    for start in starts:
        w = _project_simplex(start)
        pred = x @ w
        current = _mae(y, pred)

        for _ in range(200):
            improved = False
            for i in range(k):
                for j in range(i + 1, k):
                    mass = w[i] + w[j]
                    if mass <= 1e-15:
                        continue

                    # Move alpha mass to branch i and mass-alpha to branch j.
                    base_pred = pred - w[i] * x[:, i] - w[j] * x[:, j] + mass * x[:, j]
                    direction = x[:, i] - x[:, j]
                    residual_at_zero = base_pred - y
                    nz = np.abs(direction) > 1e-12

                    candidates = [0.0, mass]
                    if np.any(nz):
                        roots = -residual_at_zero[nz] / direction[nz]
                        root_weights = np.abs(direction[nz])
                        in_bounds = (roots >= 0.0) & (roots <= mass)
                        if np.any(in_bounds):
                            candidates.append(_weighted_median(roots[in_bounds], root_weights[in_bounds]))

                    pair_best_alpha = w[i]
                    pair_best_mae = current
                    for alpha in candidates:
                        pair_pred = base_pred + alpha * direction
                        candidate_mae = _mae(y, pair_pred)
                        if candidate_mae + 1e-12 < pair_best_mae:
                            pair_best_alpha = float(alpha)
                            pair_best_mae = candidate_mae

                    if pair_best_mae + 1e-12 < current:
                        w[i] = pair_best_alpha
                        w[j] = mass - pair_best_alpha
                        pred = base_pred + pair_best_alpha * direction
                        current = pair_best_mae
                        improved = True

            if not improved:
                break

        if current < best_mae:
            best_mae = current
            best_w = w.copy()

    best_w[np.abs(best_w) < 1e-10] = 0.0
    best_w = best_w / best_w.sum()
    return best_w


def optimize_simplex_mae(y, x):
    try:
        from scipy.optimize import linprog
    except ImportError:
        return _optimize_simplex_mae_pairwise(y, x), "Pairwise coordinate descent with exact one-dimensional MAE updates"

    n, k = x.shape
    c = np.r_[np.zeros(k), np.ones(n) / n]

    # residual r_i = x_i w - y_i, with u_i >= r_i and u_i >= -r_i.
    a_ub = []
    b_ub = []
    for i in range(n):
        row = np.zeros(k + n)
        row[:k] = x[i]
        row[k + i] = -1.0
        a_ub.append(row)
        b_ub.append(y[i])

        row = np.zeros(k + n)
        row[:k] = -x[i]
        row[k + i] = -1.0
        a_ub.append(row)
        b_ub.append(-y[i])

    a_eq = np.zeros((1, k + n))
    a_eq[0, :k] = 1.0
    bounds = [(0.0, 1.0)] * k + [(0.0, None)] * n

    try:
        result = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            A_eq=a_eq,
            b_eq=np.asarray([1.0]),
            bounds=bounds,
            method="highs",
        )
    except TypeError:
        result = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            A_eq=a_eq,
            b_eq=np.asarray([1.0]),
            bounds=bounds,
        )
    if not result.success:
        raise RuntimeError(result.message)

    w = np.asarray(result.x[:k], dtype=float)
    w[np.abs(w) < 1e-12] = 0.0
    w = w / w.sum()
    return w, "Exact linear-programming constrained MAE minimization"


def weighted_prediction(df, weights):
    cols = [spec["short"] for spec in BRANCHES]
    return df[cols].to_numpy(dtype=float) @ np.asarray(weights, dtype=float)


def build_summary_rows(val, test, weights, val_fusion_metrics, test_fusion_metrics):
    rows = []
    for spec, weight in zip(BRANCHES, weights):
        val_m = metric_dict(val["true_rt"], val[spec["short"]])
        test_m = metric_dict(test["true_rt"], test[spec["short"]])
        rows.append(
            {
                "experiment": "five-branch PTDF fusion",
                "row_type": "branch",
                "branch name": spec["branch"],
                "prediction source file": f"validation={spec['val_file'].relative_to(ROOT)}; test={spec['test_file'].relative_to(ROOT)}",
                "data provenance": spec["provenance"],
                "optimized weight": float(weight),
                "validation performance": json.dumps(val_m, sort_keys=True),
                "test performance": json.dumps(test_m, sort_keys=True),
            }
        )

    rows.append(
        {
            "experiment": "five-branch PTDF fusion",
            "row_type": "fusion",
            "branch name": "Optimized five-branch PTDF fusion",
            "prediction source file": "Derived from the five branch prediction columns listed above.",
            "data provenance": "Weights optimized on validation MAE only; test evaluated once with fixed weights.",
            "optimized weight": json.dumps({spec["short"]: float(w) for spec, w in zip(BRANCHES, weights)}, sort_keys=True),
            "validation performance": json.dumps(val_fusion_metrics, sort_keys=True),
            "test performance": json.dumps(test_fusion_metrics, sort_keys=True),
        }
    )
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val, _ = load_split("val")
    test, _ = load_split("test")
    pred_cols = [spec["short"] for spec in BRANCHES]

    weights, optimization_method = optimize_simplex_mae(
        val["true_rt"].to_numpy(dtype=float),
        val[pred_cols].to_numpy(dtype=float),
    )

    val["pred_five_branch_ptdf_fusion"] = weighted_prediction(val, weights)
    test["pred_five_branch_ptdf_fusion"] = weighted_prediction(test, weights)

    val_fusion_metrics = metric_dict(val["true_rt"], val["pred_five_branch_ptdf_fusion"])
    test_fusion_metrics = metric_dict(test["true_rt"], test["pred_five_branch_ptdf_fusion"])

    baseline_a_weights = {
        "ccrl_ensemble": 0.30,
        "residual_ccrl": 0.00,
        "g2026": 0.24,
        "g3407": 0.18,
        "g777": 0.28,
    }
    baseline_b_weights = {
        "ccrl_ensemble": 0.00,
        "residual_ccrl": 0.46,
        "g2026": 0.20,
        "g3407": 0.12,
        "g777": 0.22,
    }
    baselines = {}
    for name, w_map in [
        ("Baseline A: CCRL ensemble + GNN fusion", baseline_a_weights),
        ("Baseline B: Residual corrected CCRL + GNN fusion", baseline_b_weights),
    ]:
        w = np.asarray([w_map[c] for c in pred_cols], dtype=float)
        val_pred = weighted_prediction(val, w)
        test_pred = weighted_prediction(test, w)
        baselines[name] = {
            "weights": w_map,
            "validation": metric_dict(val["true_rt"], val_pred),
            "test": metric_dict(test["true_rt"], test_pred),
        }

    summary_rows = build_summary_rows(val, test, weights, val_fusion_metrics, test_fusion_metrics)
    for name, item in baselines.items():
        summary_rows.append(
            {
                "experiment": "five-branch PTDF fusion",
                "row_type": "baseline",
                "branch name": name,
                "prediction source file": "Derived from existing branch predictions.",
                "data provenance": "Fixed baseline weights supplied by current project context; no test-set tuning.",
                "optimized weight": json.dumps(item["weights"], sort_keys=True),
                "validation performance": json.dumps(item["validation"], sort_keys=True),
                "test performance": json.dumps(item["test"], sort_keys=True),
            }
        )

    val.to_csv(OUT_DIR / "predictions_val_five_branch_ptdf_fusion.csv", index=False)
    test.to_csv(OUT_DIR / "predictions_test_five_branch_ptdf_fusion.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary_five_branch_ptdf_fusion.csv", index=False)

    result = {
        "experiment": "five-branch PTDF fusion",
        "optimization": f"{optimization_method} on validation set",
        "weights": {spec["short"]: float(w) for spec, w in zip(BRANCHES, weights)},
        "validation": val_fusion_metrics,
        "test": test_fusion_metrics,
        "baselines": baselines,
        "best_mae_reference_s": 25.8893,
        "exceeds_best_mae_reference": bool(test_fusion_metrics["mae"] < 25.8893),
    }
    (OUT_DIR / "metrics_five_branch_ptdf_fusion.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"wrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
