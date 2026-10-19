from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRUE_CANDIDATES = ["true_rt", "rt", "label", "y_true"]
PRED_CANDIDATES = [
    "pred_final",
    "pred_final_ccrl_gnn_ptdf",
    "pred_ccrl_ensemble",
    "pred_residual_corrected",
    "pred_rt",
    "prediction",
    "pred",
    "y_pred",
]


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def first_existing(columns: list[str], candidates: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise KeyError(f"Column {explicit!r} not found. Available columns: {columns}")
        return explicit
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"None of {candidates} found. Available columns: {columns}")


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    if not y_true:
        raise ValueError("No predictions to evaluate")

    errors = [p - y for y, p in zip(y_true, y_pred)]
    abs_errors = [abs(e) for e in errors]
    nonzero = [(y, e) for y, e in zip(y_true, abs_errors) if y != 0.0]
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)

    return {
        "n": float(len(y_true)),
        "mae_s": sum(abs_errors) / len(abs_errors),
        "mape_pct": 100.0 * sum(e / abs(y) for y, e in nonzero) / len(nonzero),
        "medae_s": quantile(abs_errors, 0.5),
        "rmse_s": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "r2": 1.0 - ss_res / ss_tot,
        "p95_abs_error_s": quantile(abs_errors, 0.95),
        "max_abs_error_s": max(abs_errors),
        "within_30s_pct": 100.0 * sum(e <= 30.0 for e in abs_errors) / len(abs_errors),
        "within_60s_pct": 100.0 * sum(e <= 60.0 for e in abs_errors) / len(abs_errors),
        "within_120s_pct": 100.0 * sum(e <= 120.0 for e in abs_errors) / len(abs_errors),
    }


def load_prediction_csv(
    path: Path,
    true_col: str | None,
    pred_col: str | None,
) -> tuple[list[float], list[float], str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = list(rows[0])
    true_name = first_existing(columns, TRUE_CANDIDATES, true_col)
    pred_name = first_existing(columns, PRED_CANDIDATES, pred_col)
    y_true = [float(row[true_name]) for row in rows]
    y_pred = [float(row[pred_name]) for row in rows]
    return y_true, y_pred, true_name, pred_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retention-time predictions with one metric protocol.")
    parser.add_argument("--predictions", required=True, type=Path, help="CSV containing true and predicted RT values.")
    parser.add_argument("--method", required=True, help="Method name written into the output.")
    parser.add_argument("--true-col", default=None, help="True RT column. Auto-detected when omitted.")
    parser.add_argument("--pred-col", default=None, help="Predicted RT column. Auto-detected when omitted.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional metrics JSON output path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional one-row metrics CSV output path.")
    args = parser.parse_args()

    y_true, y_pred, true_name, pred_name = load_prediction_csv(args.predictions, args.true_col, args.pred_col)
    metrics = compute_metrics(y_true, y_pred)
    record: dict[str, object] = {
        "method": args.method,
        "prediction_file": str(args.predictions),
        "true_col": true_name,
        "pred_col": pred_name,
        **metrics,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
