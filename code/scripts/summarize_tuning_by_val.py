"""Summarize SupCR tuning runs by validation MAE, not test MAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-dir", type=Path, default=Path("src/outputs/supcr_tuning"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for history_path in args.tuning_dir.glob("*/history_cross_attention_residual.json"):
        name = history_path.parent.name
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row.get("val_mae", float("inf")))
        metrics_path = history_path.parent / "test_metrics_cross_attention_residual.json"
        test_metrics = {}
        if metrics_path.exists():
            test_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "config": name,
                "best_epoch": best.get("epoch"),
                "best_val_mae": best.get("val_mae"),
                "best_val_rmse": best.get("val_rmse"),
                "best_val_r2": best.get("val_r2"),
                "test_mae_reference_only": test_metrics.get("mae"),
                "test_rmse_reference_only": test_metrics.get("rmse"),
                "test_r2_reference_only": test_metrics.get("r2"),
            }
        )

    df = pd.DataFrame(rows).sort_values("best_val_mae", ascending=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
