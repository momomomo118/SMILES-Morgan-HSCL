"""Run simple Stage 2 baselines on the unified RT split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
except ImportError as exc:  # pragma: no cover
    raise SystemExit("RDKit is required for this script.") from exc


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.allclose(y_pred, y_pred[0]) or np.std(y_true) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson,
    }


def morgan_matrix(smiles: pd.Series, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits, useChirality=True)
        arr[i, :] = np.fromiter(fp.ToBitString(), dtype="U1").astype(np.uint8)
    return arr


def descriptor_matrix(smiles: pd.Series) -> np.ndarray:
    rows = []
    for smi in smiles.astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0.0] * 8)
            continue
        rows.append(
            [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def eval_model(name: str, model, x_train, y_train, splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict], dict[str, np.ndarray]]:
    model.fit(x_train, y_train)
    rows = []
    preds = {}
    for split, (x, y) in splits.items():
        pred = model.predict(x)
        preds[split] = pred
        rows.append({"baseline": name, "split": split, **metrics(y, pred)})
    return rows, preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = args.stage_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.project_root / "data" / "unified"
    train = pd.read_csv(data_root / "train.csv")
    val = pd.read_csv(data_root / "val.csv")
    test = pd.read_csv(data_root / "test.csv")

    y_train = train["rt"].to_numpy(dtype=np.float32)
    y_val = val["rt"].to_numpy(dtype=np.float32)
    y_test = test["rt"].to_numpy(dtype=np.float32)

    rows = []
    pred_rows = pd.DataFrame({"sample_id": test["sample_id"], "smiles": test["smiles"], "true_rt": y_test})

    for name, constant in [("mean_rt", float(np.mean(y_train))), ("median_rt", float(np.median(y_train)))]:
        for split, y in [("val", y_val), ("test", y_test)]:
            pred = np.full_like(y, constant, dtype=np.float32)
            rows.append({"baseline": name, "split": split, **metrics(y, pred)})
            if split == "test":
                pred_rows[f"pred_{name}"] = pred

    cache = out_dir / "morgan_cache.npz"
    if cache.exists():
        cached = np.load(cache)
        x_train_m, x_val_m, x_test_m = cached["x_train"], cached["x_val"], cached["x_test"]
    else:
        x_train_m = morgan_matrix(train["smiles"])
        x_val_m = morgan_matrix(val["smiles"])
        x_test_m = morgan_matrix(test["smiles"])
        np.savez_compressed(cache, x_train=x_train_m, x_val=x_val_m, x_test=x_test_m)

    split_m = {"val": (x_val_m, y_val), "test": (x_test_m, y_test)}
    for name, model in [
        ("morgan_ridge", make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0))),
        ("morgan_hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=80, learning_rate=0.08, max_leaf_nodes=31, random_state=args.seed)),
    ]:
        model_rows, preds = eval_model(name, model, x_train_m, y_train, split_m)
        rows.extend(model_rows)
        pred_rows[f"pred_{name}"] = preds["test"]

    x_train_d = descriptor_matrix(train["smiles"])
    x_val_d = descriptor_matrix(val["smiles"])
    x_test_d = descriptor_matrix(test["smiles"])
    split_d = {"val": (x_val_d, y_val), "test": (x_test_d, y_test)}
    for name, model in [
        ("rdkit_desc_ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("rdkit_desc_hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, random_state=args.seed)),
    ]:
        model_rows, preds = eval_model(name, model, x_train_d, y_train, split_d)
        rows.extend(model_rows)
        pred_rows[f"pred_{name}"] = preds["test"]

    metrics_df = pd.DataFrame(rows).sort_values(["split", "mae"])
    metrics_df.to_csv(out_dir / "baseline_metrics.csv", index=False)
    pred_rows.to_csv(out_dir / "baseline_predictions_test.csv", index=False)
    (out_dir / "baseline_run_config.json").write_text(json.dumps({"seed": args.seed}, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
