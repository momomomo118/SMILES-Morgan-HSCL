from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "fair_smrt_eval"
OUT_CSV = EVAL_DIR / "fair_smrt_comparison.csv"
OUT_MD = EVAL_DIR / "fair_smrt_comparison.md"


REPRODUCED_JSONS = [
    EVAL_DIR / "reproduced_transformer_lstm.json",
    EVAL_DIR / "deepgcn_rt.json",
    EVAL_DIR / "ours_final_ccrl_gnn_ptdf.json",
]


LITERATURE_ONLY = [
    ("GCN (2021)", 29.40, "4*", 15.02, 0.89, "literature reported; official/author code candidate"),
    ("DNNpwa (2021)", 39.62, "5*", 25.08, 0.85, "literature reported; no clear public implementation found"),
    ("GNN-RT (2021)", 39.87, "5*", 25.24, 0.85, "literature reported; code candidate needs integration"),
    ("Blender (2022)", 34.00, "N.A", 17.20, "N.A", "literature reported; CMM-RT repo candidate, descriptor dependency"),
    ("1D CNN (2022)", 34.70, 4.30, 18.47, "N.A", "literature reported; SMRT_transfer repo candidate, R workflow"),
    ("MPNN (2022)", 31.50, 4.00, 16.00, 0.88, "literature reported; likely needs local reimplementation"),
    ("CPORT (2022)", 44.00, 5.50, 26.00, "N.A", "literature reported; official repo candidate, heavy 3D tensors"),
    ("DeepGCN-RT (2023) literature value", 26.55, 3.30, 12.38, 0.89, "literature reported; official repo candidate"),
    ("RT-transformer (2024)", 27.30, 3.42, 12.46, 0.88, "literature reported; official repo candidate"),
]


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_reproduced() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in REPRODUCED_JSONS:
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": record["method"],
                "status": "same-split reproduced",
                "n": int(record["n"]),
                "mae_s": record["mae_s"],
                "mape_pct": record["mape_pct"],
                "medae_s": record["medae_s"],
                "r2": record["r2"],
                "rmse_s": record["rmse_s"],
                "p95_abs_error_s": record["p95_abs_error_s"],
                "note": record.get("prediction_file", record.get("note", "")),
            }
        )
    return rows


def literature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, mae, mape, medae, r2, note in LITERATURE_ONLY:
        rows.append(
            {
                "method": method,
                "status": "literature reported",
                "n": "N.A",
                "mae_s": mae,
                "mape_pct": mape,
                "medae_s": medae,
                "r2": r2,
                "rmse_s": "N.A",
                "p95_abs_error_s": "N.A",
                "note": note,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "status", "n", "mae_s", "mape_pct", "medae_s", "r2", "rmse_s", "p95_abs_error_s", "note"]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fair SMRT Comparison",
        "",
        "Rows marked `same-split reproduced` were evaluated from per-sample predictions on the fixed SMRT test split with one local metric script.",
        "Rows marked `literature reported` are Table 2 values and are not yet same-split local reruns.",
        "",
        "| Method | Status | N | MAE (s) | MAPE (%) | MedAE (s) | R2 | RMSE (s) | P95 AE (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {n} | {mae} | {mape} | {medae} | {r2} | {rmse} | {p95} |".format(
                method=row["method"],
                status=row["status"],
                n=row["n"],
                mae=fmt(row["mae_s"]),
                mape=fmt(row["mape_pct"]),
                medae=fmt(row["medae_s"]),
                r2=fmt(row["r2"], 4),
                rmse=fmt(row["rmse_s"]),
                p95=fmt(row["p95_abs_error_s"]),
            )
        )
    lines.append("")
    lines.append("Reproduction routing details are in `docs/smrt_literature_reproduction_plan.md`.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_reproduced() + literature_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
