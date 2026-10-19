"""Strict PredRet full ensemble.

Blend weights are selected only on inner-validation predictions, then frozen and
applied once to the outer-test predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")


SYSTEMS = [
    "Eawag_XBridgeC18",
    "FEM_Lipids",
    "FEM_long",
    "FEM_short",
    "IPB_Halle",
    "LIFE_new",
    "LIFE_old",
    "UniToyama_Atlantis",
    "Cao_HILIC",
    "MTBLS87",
]


def parse_folds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    abs_err = np.abs(err)
    denom = np.clip(np.abs(y), 1e-12, None)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "medae": float(np.median(abs_err)),
        "mape": float(np.mean(abs_err / denom) * 100.0),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "pearson": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(p) > 0 and np.std(y) > 0 else float("nan"),
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
        make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=0.05, random_state=20260909),
        ),
        ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=20260909, n_jobs=-1),
    ]
    preds = []
    for model in models:
        model.fit(x_train, y_train)
        preds.append(model.predict(x_pred))
    return np.mean(np.vstack(preds), axis=0)


def read_pred(path: Path, column_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = ["sample_id", "smiles", "true_rt", "pred_rt"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return df[needed].rename(columns={"pred_rt": column_name})


def merge_prediction_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part[["sample_id", part.columns[-1]]], on="sample_id", how="inner")
    if len(merged) != len(parts[0]):
        raise ValueError("Prediction merge lost rows; sample_id sets are inconsistent.")
    return merged


def load_split_predictions(args: argparse.Namespace, system: str, fold: int, split: str) -> pd.DataFrame:
    fold_name = f"fold_{fold:02d}"
    parts = []
    for member in args.checkpoint_names:
        pred_name = "predictions_val_cross_attention_residual.csv" if split == "val" else "predictions_cross_attention_residual.csv"
        parts.append(read_pred(args.base_root / member / system / fold_name / pred_name, f"base_{member}"))
    for seed in args.gnn_seeds:
        pred_name = "predictions_val_gnn.csv" if split == "val" else "predictions_test_gnn.csv"
        parts.append(read_pred(args.gnn_root / f"seed{seed}" / system / fold_name / pred_name, f"gnn_{seed}"))
    return merge_prediction_parts(parts)


def best_blend(y: np.ndarray, features: np.ndarray, candidates: int) -> tuple[np.ndarray, dict[str, float]]:
    n = features.shape[1]
    rng = np.random.default_rng(20260909)
    best_w = np.ones(n) / n
    best_m = metrics(y, features @ best_w)
    all_candidates = [best_w]
    all_candidates.extend(np.eye(n))
    for _ in range(candidates):
        all_candidates.append(rng.dirichlet(np.ones(n)))
    for w in all_candidates:
        m = metrics(y, features @ w)
        if m["mae"] < best_m["mae"]:
            best_w = w
            best_m = m
    return best_w, best_m


def summarize(detail: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        detail.groupby("system")
        .agg(
            n=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            medae_mean=("medae", "mean"),
            medae_std=("medae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_path, index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold/by_system_10fold"))
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--base-checkpoint-root", type=Path, default=None)
    ap.add_argument("--gnn-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--checkpoint-names", nargs="+", required=True)
    ap.add_argument("--gnn-seeds", nargs="+", default=["2026", "3407", "777"])
    ap.add_argument("--blend-candidates", type=int, default=8000)
    args = ap.parse_args()
    if args.base_checkpoint_root is None:
        args.base_checkpoint_root = args.base_root

    if args.output_dir.resolve() == Path("server_results/predret_compare").resolve():
        raise ValueError("Refusing to write strict ensemble into historical result directory.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    folds = parse_folds(args.folds)
    for system in args.systems:
        for fold in folds:
            fold_name = f"fold_{fold:02d}"
            fold_dir = args.data_root / system / fold_name
            val_df = load_split_predictions(args, system, fold, "val")
            test_df = load_split_predictions(args, system, fold, "test")
            train_csv = fold_dir / "train.csv"
            val_df["rdkit"] = fit_rdkit(train_csv, val_df)
            test_df["rdkit"] = fit_rdkit(train_csv, test_df)
            pred_cols = [c for c in val_df.columns if c.startswith("base_") or c.startswith("gnn_") or c == "rdkit"]

            y_val = val_df["true_rt"].astype(float).to_numpy()
            val_features = val_df[pred_cols].to_numpy(dtype=float)
            weights, val_metrics = best_blend(y_val, val_features, args.blend_candidates)

            y_test = test_df["true_rt"].astype(float).to_numpy()
            test_features = test_df[pred_cols].to_numpy(dtype=float)
            val_df["pred_full_ensemble"] = val_features @ weights
            test_df["pred_full_ensemble"] = test_features @ weights
            test_metrics = metrics(y_test, test_df["pred_full_ensemble"].to_numpy())

            out_fold = args.output_dir / system / fold_name
            out_fold.mkdir(parents=True, exist_ok=True)
            val_df.to_csv(out_fold / "predictions_full_ensemble_val.csv", index=False)
            test_df.to_csv(out_fold / "predictions_full_ensemble.csv", index=False)
            payload = {
                "system": system,
                "fold": fold,
                "blend_selection_split": "inner_validation",
                "outer_test_used_for_weight_selection": False,
                "columns": pred_cols,
                "weights": weights.tolist(),
                "validation": val_metrics,
                "test": test_metrics,
            }
            (out_fold / "metrics_full_ensemble.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            checkpoint_paths = [
                str(args.base_checkpoint_root / member / system / fold_name / "best_cross_attention_residual.pt")
                for member in args.checkpoint_names
            ] + [
                str(args.gnn_root / f"seed{seed}" / system / fold_name / "best_pure_gin.pt")
                for seed in args.gnn_seeds
            ]
            prediction_paths = [
                str(args.base_root / member / system / fold_name / "predictions_val_cross_attention_residual.csv")
                for member in args.checkpoint_names
            ] + [
                str(args.base_root / member / system / fold_name / "predictions_cross_attention_residual.csv")
                for member in args.checkpoint_names
            ] + [
                str(args.gnn_root / f"seed{seed}" / system / fold_name / "predictions_val_gnn.csv")
                for seed in args.gnn_seeds
            ] + [
                str(args.gnn_root / f"seed{seed}" / system / fold_name / "predictions_test_gnn.csv")
                for seed in args.gnn_seeds
            ]
            rows.append(
                {
                    "system": system,
                    "fold": fold,
                    **test_metrics,
                    "best_epoch": "",
                    "validation_mae": val_metrics["mae"],
                    "test_mae": test_metrics["mae"],
                    "blend_selection_split": "inner_validation",
                    "outer_test_used_for_weight_selection": False,
                    "blend_columns_json": json.dumps(pred_cols),
                    "blend_weights_json": json.dumps(weights.tolist()),
                    "checkpoint_paths_json": json.dumps(checkpoint_paths),
                    "prediction_paths_json": json.dumps(prediction_paths),
                    "metrics_path": str(out_fold / "metrics_full_ensemble.json"),
                    "validation_prediction_path": str(out_fold / "predictions_full_ensemble_val.csv"),
                    "test_prediction_path": str(out_fold / "predictions_full_ensemble.csv"),
                }
            )
            print(f"strict_full_ensemble {system} {fold_name} val_mae={val_metrics['mae']:.4f} test_mae={test_metrics['mae']:.4f}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "ours_full_ensemble_detail.csv", index=False)
    summary = summarize(detail, args.output_dir / "ours_full_ensemble_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
