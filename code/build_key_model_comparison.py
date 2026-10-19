from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "fair_smrt_eval"
OUT_CSV = EVAL_DIR / "key_model_comparison.csv"
OUT_MD = EVAL_DIR / "key_model_comparison.md"


INPUTS = [
    EVAL_DIR / "reproduced_transformer_lstm.json",
    EVAL_DIR / "deepgcn_rt.json",
    EVAL_DIR / "graphormer_rt.json",
    EVAL_DIR / "gnn_rt.json",
    EVAL_DIR / "gcn_kensert_earlystop.json",
    EVAL_DIR / "one_d_cnn_original_style.json",
    EVAL_DIR / "ours_final_ccrl_gnn_ptdf.json",
    EVAL_DIR / "rt_transformer_failed.json",
    EVAL_DIR / "mpnn_failed.json",
]


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in INPUTS]
    lstm = next(r for r in records if r["method"] == "Reproduced Transformer-LSTM")
    ours = next(r for r in records if r["method"] == "Final CCRL ensemble + GNN PTDF")

    rows = []
    for record in records:
        rows.append(
            {
                "method": record["method"],
                "n": int(record["n"]),
                "mae_s": record["mae_s"],
                "mape_pct": record["mape_pct"],
                "medae_s": record["medae_s"],
                "rmse_s": record["rmse_s"],
                "r2": record["r2"],
                "p95_abs_error_s": record["p95_abs_error_s"],
                "max_abs_error_s": record["max_abs_error_s"],
                "delta_mae_vs_lstm": record["mae_s"] - lstm["mae_s"],
                "delta_mae_vs_ours": record["mae_s"] - ours["mae_s"],
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Key SMRT Same-Split Comparison",
        "",
        "| Method | N | MAE (s) | MAPE (%) | MedAE (s) | RMSE (s) | R2 | P95 AE (s) | Max AE (s) | Delta MAE vs LSTM | Delta MAE vs Ours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n} | {mae} | {mape} | {medae} | {rmse} | {r2} | {p95} | {maxae} | {dlstm} | {dours} |".format(
                method=row["method"],
                n=row["n"],
                mae=fmt(row["mae_s"]),
                mape=fmt(row["mape_pct"]),
                medae=fmt(row["medae_s"]),
                rmse=fmt(row["rmse_s"]),
                r2=fmt(row["r2"]),
                p95=fmt(row["p95_abs_error_s"]),
                maxae=fmt(row["max_abs_error_s"]),
                dlstm=fmt(row["delta_mae_vs_lstm"]),
                dours=fmt(row["delta_mae_vs_ours"]),
            )
        )
    lines.extend(
        [
            "",
            "Lower is better for MAE, MAPE, MedAE, RMSE, and P95 AE; higher is better for R2.",
            "DeepGCN-RT metrics were copied from the server same-split evaluation output on 2026-07-01.",
            "Graphormer-RT metrics were copied from the server fixed-test evaluation output on 2026-07-22. The published checkpoint may overlap with SMRT data, so this row is not claimed to be leakage-free.",
            "GNN-RT metrics were copied from the server same-split evaluation output on 2026-07-02.",
            "GCN metrics were copied from the server same-split early-stop evaluation output on 2026-07-04.",
            "1D CNN original-style metrics were copied from the server same-split early-stop evaluation output on 2026-07-04.",
            "RT-transformer rerun is retained only as a failed reproduction record: predictions collapsed near the mean and are not comparable to the paper-reported result.",
            "MPNN rerun is retained only as a failed reproduction record: the simplified PyG implementation collapsed near the mean and is not comparable to the paper-reported result.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
