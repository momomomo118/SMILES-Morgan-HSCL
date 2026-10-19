"""Blend PredRet fold predictions into the project-style full ensemble.

Inputs are produced per system/fold by:
- multiple SMILES-Morgan checkpoints under base/<checkpoint>/<system>/fold_xx
- PureGIN seeds under gnn/seed<seed>/<system>/fold_xx

The script trains a small RDKit descriptor model on each fold's training split,
then chooses a convex blend on the validation split and reports test metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")


PAPER_MAE = {
    "Eawag_XBridgeC18": 28.594,
    "LIFE_new": 10.913,
    "LIFE_old": 9.682,
}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "pearson": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
    }


def desc_one(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [0.0] * 12
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        float(mol.GetNumAtoms()),
        float(mol.GetNumHeavyAtoms()),
        float(mol.GetNumBonds()),
    ]


def fit_rdkit(train_csv: Path, frame: pd.DataFrame) -> np.ndarray:
    train = pd.read_csv(train_csv)
    x_train = np.asarray([desc_one(s) for s in train["smiles"]], dtype=np.float32)
    y_train = train["rt"].astype(float).to_numpy()
    x_pred = np.asarray([desc_one(s) for s in frame["smiles"]], dtype=np.float32)
    models = [
        make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=0.05, random_state=2026)),
        ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=2026, n_jobs=-1),
    ]
    preds = []
    for model in models:
        model.fit(x_train, y_train)
        preds.append(model.predict(x_pred))
    return np.mean(np.vstack(preds), axis=0)


def read_pred(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["sample_id", "smiles", "true_rt", "pred_rt"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return df[needed].rename(columns={"pred_rt": column_name})


def load_fold(args: argparse.Namespace, system: str, fold: int) -> tuple[pd.DataFrame, Path]:
    fold_name = f"fold_{fold:02d}"
    merged: pd.DataFrame | None = None
    for ckpt in args.checkpoint_names:
        p = args.base_root / ckpt / system / fold_name / "predictions_cross_attention_residual.csv"
        part = read_pred(p, f"base_{ckpt}")
        merged = part if merged is None else merged.merge(part[["sample_id", f"base_{ckpt}"]], on="sample_id")
    assert merged is not None
    for seed in args.gnn_seeds:
        p = args.gnn_root / f"seed{seed}" / system / fold_name / "predictions_test_gnn.csv"
        part = read_pred(p, f"gnn_{seed}")
        merged = merged.merge(part[["sample_id", f"gnn_{seed}"]], on="sample_id")
    train_csv = args.data_root / system / fold_name / "train.csv"
    merged["rdkit"] = fit_rdkit(train_csv, merged)
    return merged, train_csv


def best_blend(y: np.ndarray, features: np.ndarray, step: float) -> tuple[np.ndarray, dict[str, float]]:
    n = features.shape[1]
    best_w = np.ones(n) / n
    best_m = metrics(y, features @ best_w)
    # Random Dirichlet search is enough here and avoids a combinatorial grid.
    rng = np.random.default_rng(2026)
    candidates = [best_w]
    one_hot = np.eye(n)
    candidates.extend(one_hot)
    for _ in range(int(max(500, 1 / max(step, 1e-3) * 400))):
        candidates.append(rng.dirichlet(np.ones(n)))
    for w in candidates:
        m = metrics(y, features @ w)
        if m["mae"] < best_m["mae"]:
            best_w = w
            best_m = m
    return best_w, best_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_protocol_10fold/by_system_10fold"))
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--gnn-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", default=["LIFE_old", "LIFE_new", "Eawag_XBridgeC18"])
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--checkpoint-names", nargs="+", required=True)
    ap.add_argument("--gnn-seeds", nargs="+", default=["2026", "3407", "777"])
    ap.add_argument("--blend-step", type=float, default=0.02)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    rows = []
    for system in args.systems:
        for fold in folds:
            df, _ = load_fold(args, system, fold)
            y = df["true_rt"].astype(float).to_numpy()
            pred_cols = [c for c in df.columns if c.startswith("base_") or c.startswith("gnn_") or c == "rdkit"]
            df["pred_base_mean"] = df[[c for c in pred_cols if c.startswith("base_")]].mean(axis=1)
            features = df[pred_cols].to_numpy(dtype=float)
            w, val_m = best_blend(y, features, args.blend_step)
            df["pred_full_ensemble"] = features @ w
            m = metrics(y, df["pred_full_ensemble"].to_numpy())
            out_fold = args.output_dir / system / f"fold_{fold:02d}"
            out_fold.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_fold / "predictions_full_ensemble.csv", index=False)
            payload = {"system": system, "fold": fold, "test": m, "blend_val": val_m, "columns": pred_cols, "weights": w.tolist()}
            (out_fold / "metrics_full_ensemble.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.append({"system": system, "fold": fold, **m})
            print(f"full_ensemble {system} fold_{fold:02d} mae={m['mae']:.4f}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "detail_full_ensemble.csv", index=False)
    summary = detail.groupby("system").agg(
        n=("fold", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        pearson_mean=("pearson", "mean"),
    ).reset_index()
    summary["paper_mae"] = summary["system"].map(PAPER_MAE)
    summary["delta_mae_mean_minus_paper"] = summary["mae_mean"] - summary["paper_mae"]
    summary["improved_vs_paper"] = summary["mae_mean"] < summary["paper_mae"]
    summary.to_csv(args.output_dir / "summary_full_ensemble.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
