"""Generate worst-case and binned error analysis for prediction CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover
    raise SystemExit("RDKit is required for this script.") from exc


def choose_prediction(project_root: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    candidates = list((project_root / "src" / "outputs").rglob("predictions*.csv"))
    candidates += list((project_root / "server_results").rglob("predictions*.csv"))
    preferred = [
        p
        for p in candidates
        if "exact_cool_light" in str(p) or "multimodal_results_20epoch" in str(p)
    ]
    if preferred:
        return sorted(preferred, key=lambda p: ("exact_cool_light" not in str(p), str(p)))[0]
    if not candidates:
        raise FileNotFoundError("No predictions*.csv found under src/outputs or server_results.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def mol_features(smiles: pd.Series) -> pd.DataFrame:
    rows = []
    for smi in smiles.astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({})
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
        bit_density = float(fp.GetNumOnBits()) / 2048.0
        rows.append(
            {
                "smiles_length": len(smi),
                "morgan_on_bits": fp.GetNumOnBits(),
                "morgan_bit_density": bit_density,
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Crippen.MolLogP(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "RotBonds": Lipinski.NumRotatableBonds(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
            }
        )
    return pd.DataFrame(rows)


def bin_summary(df: pd.DataFrame, column: str, bins: int = 10) -> pd.DataFrame:
    tmp = df.copy()
    tmp[f"{column}_bin"] = pd.qcut(tmp[column], q=bins, duplicates="drop")
    return (
        tmp.groupby(f"{column}_bin", observed=True)
        .agg(
            n=("abs_error", "size"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_median=("abs_error", "median"),
            abs_error_p90=("abs_error", lambda x: x.quantile(0.9)),
            true_rt_mean=("true_rt", "mean"),
        )
        .reset_index()
    )


def save_bar(summary: pd.DataFrame, bin_col: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(range(len(summary)), summary[y])
    plt.xticks(range(len(summary)), summary[bin_col].astype(str), rotation=45, ha="right", fontsize=8)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prediction", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.stage_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = choose_prediction(args.project_root, args.prediction)
    df = pd.read_csv(pred_path)
    if "abs_error" not in df:
        df["error"] = df["pred_rt"] - df["true_rt"]
        df["abs_error"] = df["error"].abs()
    features = mol_features(df["smiles"])
    df = pd.concat([df, features], axis=1)

    df.sort_values("abs_error", ascending=False).head(100).to_csv(out_dir / "worst_100_errors.csv", index=False)
    df.sort_values("abs_error", ascending=True).head(100).to_csv(out_dir / "best_100_errors.csv", index=False)
    median_idx = (df["abs_error"] - df["abs_error"].median()).abs().sort_values().head(100).index
    df.loc[median_idx].sort_values("abs_error").to_csv(out_dir / "median_100_errors.csv", index=False)

    summaries = {
        "error_by_rt_bin.csv": bin_summary(df, "true_rt"),
        "error_by_smiles_length_bin.csv": bin_summary(df, "smiles_length"),
        "error_by_morgan_bit_density_bin.csv": bin_summary(df, "morgan_bit_density"),
    }
    desc_rows = []
    for col in ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "RingCount"]:
        s = bin_summary(df, col, bins=6)
        s.insert(0, "descriptor", col)
        desc_rows.append(s.rename(columns={f"{col}_bin": "bin"}))
    summaries["error_by_rdkit_descriptor_bins.csv"] = pd.concat(desc_rows, ignore_index=True)
    for name, summary in summaries.items():
        summary.to_csv(out_dir / name, index=False)

    plt.figure(figsize=(7, 4.5))
    plt.hist(df["abs_error"], bins=80)
    plt.xlabel("Absolute error (s)")
    plt.ylabel("Count")
    plt.title("Absolute error distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "abs_error_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(df["true_rt"], df["pred_rt"], s=8, alpha=0.35)
    lo, hi = df[["true_rt", "pred_rt"]].min().min(), df[["true_rt", "pred_rt"]].max().max()
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("True RT")
    plt.ylabel("Predicted RT")
    plt.title("True vs predicted RT")
    plt.tight_layout()
    plt.savefig(out_dir / "true_vs_predicted_rt_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["true_rt"], df["error"], s=8, alpha=0.35)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("True RT")
    plt.ylabel("Residual (pred - true)")
    plt.title("Residual vs true RT")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_vs_true_rt.png", dpi=180)
    plt.close()

    save_bar(summaries["error_by_rt_bin.csv"], "true_rt_bin", "abs_error_mean", out_dir / "abs_error_by_rt_bin.png", "Abs error by RT bin")
    save_bar(summaries["error_by_smiles_length_bin.csv"], "smiles_length_bin", "abs_error_mean", out_dir / "abs_error_by_smiles_length_bin.png", "Abs error by SMILES length bin")
    save_bar(summaries["error_by_morgan_bit_density_bin.csv"], "morgan_bit_density_bin", "abs_error_mean", out_dir / "abs_error_by_morgan_bit_density_bin.png", "Abs error by Morgan bit density bin")

    meta = {
        "prediction_path": str(pred_path),
        "rows": int(len(df)),
        "mae": float(df["abs_error"].mean()),
        "median_abs_error": float(df["abs_error"].median()),
        "p95_abs_error": float(df["abs_error"].quantile(0.95)),
        "max_abs_error": float(df["abs_error"].max()),
    }
    (out_dir / "error_analysis_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
