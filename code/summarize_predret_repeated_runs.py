"""Summarize repeated PredRet runs as mean +/- std and compare with the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAPER_MAE = {
    "Eawag_XBridgeC18": 28.594,
    "FEM_Lipids": 62.072,
    "FEM_long": 121.703,
    "FEM_short": 49.629,
    "IPB_Halle": 17.372,
    "LIFE_new": 10.913,
    "LIFE_old": 9.682,
    "UniToyama_Atlantis": 43.161,
    "Cao_HILIC": 38.121,
    "MTBLS87": 50.963,
}


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "full_ensemble_test" in data:
        data = data["full_ensemble_test"]
    elif "blend_test" in data:
        data = data["blend_test"]
    elif "test" in data:
        data = data["test"]
    return {
        "mae": float(data["mae"]),
        "rmse": float(data["rmse"]),
        "r2": float(data["r2"]),
        "pearson": float(data.get("pearson", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="Root like outputs/predret_repeat_smiles_morgan")
    ap.add_argument(
        "--metrics-name",
        default="test_metrics_cross_attention_residual.json",
        help="Metrics filename inside <root>/<system>/seed_<seed>/",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for system_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        system = system_dir.name
        run_dirs = [
            p
            for p in system_dir.iterdir()
            if p.is_dir() and (p.name.startswith("seed_") or p.name.startswith("fold_"))
        ]
        for run_dir in sorted(run_dirs):
            run_id = run_dir.name.replace("seed_", "").replace("fold_", "")
            metrics = load_metrics(run_dir / args.metrics_name)
            if metrics is None:
                continue
            rows.append({"system": system, "run": int(run_id), "run_name": run_dir.name, **metrics})

    if not rows:
        raise SystemExit(f"No metrics found under {root} with filename {args.metrics_name}")

    detail = pd.DataFrame(rows).sort_values(["system", "run"])
    summary = (
        detail.groupby("system")
        .agg(
            n=("run", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
        )
        .reset_index()
    )
    summary["paper_mae"] = summary["system"].map(PAPER_MAE)
    summary["delta_mae_mean_minus_paper"] = summary["mae_mean"] - summary["paper_mae"]
    summary["improved_vs_paper"] = summary["mae_mean"] < summary["paper_mae"]
    summary["mae_mean_std"] = summary.apply(
        lambda r: f"{r['mae_mean']:.3f} +/- {0.0 if pd.isna(r['mae_std']) else r['mae_std']:.3f}",
        axis=1,
    )

    output = Path(args.output) if args.output else root / "summary_mean_std.csv"
    detail_output = output.with_name(output.stem + "_detail.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    detail.to_csv(detail_output, index=False)

    cols = ["system", "n", "paper_mae", "mae_mean_std", "delta_mae_mean_minus_paper", "improved_vs_paper"]
    print(summary[cols].sort_values("system").to_string(index=False))
    print(f"\nWrote: {output}")
    print(f"Wrote: {detail_output}")


if __name__ == "__main__":
    main()
