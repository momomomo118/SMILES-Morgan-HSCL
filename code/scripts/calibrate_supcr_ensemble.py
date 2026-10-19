"""Build and calibrate a SupCR ensemble on the unified validation/test split.

The script loads trained multimodal checkpoints, exports validation/test
predictions, searches ensemble weights on validation MAE, fits a conservative
residual calibrator on validation predictions, and reports calibrated test
metrics. The test set is never used to fit weights or calibration.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_multimodal import (  # noqa: E402
    as_plain_list,
    build_model,
    build_smiles_views,
    move_batch,
    predict_with_tta,
    regression_metrics,
)
from utils.multimodal_dataset import MultimodalRTDataset  # noqa: E402


RDLogger.DisableLog("rdApp.*")


@dataclass
class PredictionFrame:
    name: str
    split: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validation-weighted SupCR ensemble with residual correction.")
    parser.add_argument("--val-data", default=str(PROJECT_ROOT / "data" / "unified" / "val.csv"))
    parser.add_argument("--test-data", default=str(PROJECT_ROOT / "data" / "unified" / "test.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "outputs" / "supcr_tuning" / "ensemble_aug035_huber_calibrated"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-smiles", type=int, default=1)
    parser.add_argument("--tta-seed", type=int, default=2026)
    parser.add_argument("--no-tta-canonical", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="name=path. Repeat for each seed checkpoint.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Grid step for validation ensemble weight search.",
    )
    parser.add_argument("--max-calibration-abs", type=float, default=80.0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing prediction CSVs.")
    return parser.parse_args()


def default_checkpoints() -> Dict[str, Path]:
    base = PROJECT_ROOT / "src" / "checkpoints" / "supcr_tuning"
    return {
        "seed1234_v3": base / "aug035_huber_v3" / "best_cross_attention_residual.pt",
        "seed2025": base / "aug035_huber_seed2025" / "best_cross_attention_residual.pt",
        "seed3407": base / "aug035_huber_seed3407" / "best_cross_attention_residual.pt",
        "seed42": base / "aug035_huber_seed42" / "best_cross_attention_residual.pt",
        "seed777": base / "aug035_huber_seed777" / "best_cross_attention_residual.pt",
    }


def parse_checkpoints(values: List[str]) -> Dict[str, Path]:
    if not values:
        return default_checkpoints()
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint must be name=path, got: {value}")
        name, path = value.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def namespace_from_checkpoint(checkpoint: dict, cli_args: argparse.Namespace, split_path: str) -> SimpleNamespace:
    ckpt_args = dict(checkpoint.get("args", {}))
    ckpt_args["test_data"] = split_path
    ckpt_args["eval_batch_size"] = cli_args.eval_batch_size or ckpt_args.get("eval_batch_size", 32)
    ckpt_args["num_workers"] = cli_args.num_workers
    ckpt_args["device"] = cli_args.device
    ckpt_args["tta_smiles"] = cli_args.tta_smiles
    ckpt_args["tta_seed"] = cli_args.tta_seed
    ckpt_args["no_tta_canonical"] = cli_args.no_tta_canonical
    return SimpleNamespace(**ckpt_args)


def build_eval_loader(args: SimpleNamespace, tokenizer) -> DataLoader:
    dataset = MultimodalRTDataset(
        args.test_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        morgan_mode=args.morgan_mode,
        morgan_shuffle_seed=args.seed + 2000,
    )
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


@torch.no_grad()
def predict_checkpoint(
    name: str,
    checkpoint_path: Path,
    split: str,
    split_path: str,
    cli_args: argparse.Namespace,
    output_dir: Path,
) -> PredictionFrame:
    output_path = output_dir / "base_predictions" / f"{name}_{split}.csv"
    if output_path.exists() and not cli_args.force:
        return PredictionFrame(name=name, split=split, frame=pd.read_csv(output_path))

    checkpoint = torch.load(checkpoint_path, map_location=cli_args.device, weights_only=False)
    args = namespace_from_checkpoint(checkpoint, cli_args, split_path)
    tokenizer = AutoTokenizer.from_pretrained(args.smiles_model)
    loader = build_eval_loader(args, tokenizer)
    model = build_model(args).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    use_tta = args.tta_smiles > 1
    for batch in tqdm(loader, desc=f"{name} {split}", leave=False):
        batch = move_batch(batch, args.device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        prediction = predict_with_tta(model, batch, tokenizer, args) if use_tta else output.prediction
        sample_ids = as_plain_list(batch.get("sample_id", ""))
        smiles_list = as_plain_list(batch.get("smiles", ""))
        true_values = batch["labels"].detach().cpu().numpy()
        pred_values = prediction.detach().cpu().numpy()
        for sample_id, smiles, true_rt, pred_rt in zip(sample_ids, smiles_list, true_values, pred_values):
            rows.append(
                {
                    "sample_id": sample_id,
                    "smiles": smiles,
                    "true_rt": float(true_rt),
                    f"pred_{name}": float(pred_rt),
                }
            )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return PredictionFrame(name=name, split=split, frame=frame)


def merge_predictions(predictions: List[PredictionFrame]) -> pd.DataFrame:
    merged = predictions[0].frame.copy()
    for pred in predictions[1:]:
        merged = merged.merge(
            pred.frame[["sample_id", f"pred_{pred.name}"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
    return merged


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_t = torch.tensor(y_true, dtype=torch.float32)
    pred_t = torch.tensor(y_pred, dtype=torch.float32)
    return regression_metrics(true_t, pred_t)


def weight_grid(n_models: int, step: float) -> Iterable[np.ndarray]:
    units = int(round(1.0 / step))
    for combo in itertools.product(range(units + 1), repeat=n_models):
        if sum(combo) == units:
            yield np.array(combo, dtype=np.float64) / units


def find_best_weights(val_frame: pd.DataFrame, names: List[str], step: float) -> tuple[np.ndarray, Dict[str, float]]:
    y_true = val_frame["true_rt"].to_numpy(dtype=np.float64)
    preds = val_frame[[f"pred_{name}" for name in names]].to_numpy(dtype=np.float64)
    best_weights = None
    best_metrics = None
    best_mae = math.inf
    for weights in weight_grid(len(names), step):
        y_pred = preds @ weights
        metrics = metric_dict(y_true, y_pred)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_metrics = metrics
            best_weights = weights
    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics


def rdkit_features(smiles_values: Iterable[str]) -> pd.DataFrame:
    rows = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            rows.append({})
            continue
        rows.append(
            {
                "mol_wt": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "rings": rdMolDescriptors.CalcNumRings(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                "smiles_len": len(str(smiles)),
            }
        )
    return pd.DataFrame(rows).fillna(0.0)


def calibration_features(frame: pd.DataFrame, names: List[str], weights: np.ndarray) -> pd.DataFrame:
    preds = frame[[f"pred_{name}" for name in names]].to_numpy(dtype=np.float64)
    ensemble = preds @ weights
    pred_std = preds.std(axis=1)
    pred_min = preds.min(axis=1)
    pred_max = preds.max(axis=1)
    features = pd.DataFrame(
        {
            "ensemble_pred": ensemble,
            "ensemble_pred_sq": ensemble**2,
            "ensemble_pred_sqrt": np.sqrt(np.clip(ensemble, 0.0, None)),
            "pred_std": pred_std,
            "pred_range": pred_max - pred_min,
        }
    )
    return pd.concat([features, rdkit_features(frame["smiles"])], axis=1)


def candidate_calibrators(random_state: int):
    return {
        "ridge_poly2": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=10.0),
        ),
        "huber": make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def fit_calibrator(
    val_frame: pd.DataFrame,
    names: List[str],
    weights: np.ndarray,
    max_abs: float,
) -> tuple[str, object, float, Dict[str, float]]:
    y_true = val_frame["true_rt"].to_numpy(dtype=np.float64)
    base_pred = val_frame[[f"pred_{name}" for name in names]].to_numpy(dtype=np.float64) @ weights
    residual = y_true - base_pred
    x_val = calibration_features(val_frame, names, weights)

    best = ("none", None, 0.0, metric_dict(y_true, base_pred))
    best_mae = best[3]["mae"]
    for model_name, model in candidate_calibrators(random_state=1234).items():
        model.fit(x_val, residual)
        raw_delta = np.clip(model.predict(x_val), -max_abs, max_abs)
        for shrinkage in np.linspace(0.0, 1.0, 21):
            pred = base_pred + shrinkage * raw_delta
            metrics = metric_dict(y_true, pred)
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best = (model_name, model, float(shrinkage), metrics)
    return best


def apply_calibrator(
    frame: pd.DataFrame,
    names: List[str],
    weights: np.ndarray,
    model,
    shrinkage: float,
    max_abs: float,
) -> pd.DataFrame:
    output = frame.copy()
    preds = output[[f"pred_{name}" for name in names]].to_numpy(dtype=np.float64)
    base_pred = preds @ weights
    if model is None or shrinkage <= 0.0:
        delta = np.zeros_like(base_pred)
    else:
        delta = np.clip(model.predict(calibration_features(output, names, weights)), -max_abs, max_abs) * shrinkage
    output["pred_ensemble"] = base_pred
    output["calibration_delta"] = delta
    output["pred_calibrated"] = base_pred + delta
    output["error_calibrated"] = output["pred_calibrated"] - output["true_rt"]
    output["abs_error_calibrated"] = output["error_calibrated"].abs()
    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = parse_checkpoints(args.checkpoint)
    names = list(checkpoints.keys())

    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")

    val_predictions = [
        predict_checkpoint(name, path, "val", args.val_data, args, output_dir)
        for name, path in checkpoints.items()
    ]
    test_predictions = [
        predict_checkpoint(name, path, "test", args.test_data, args, output_dir)
        for name, path in checkpoints.items()
    ]
    val_frame = merge_predictions(val_predictions)
    test_frame = merge_predictions(test_predictions)

    weights, val_ensemble_metrics = find_best_weights(val_frame, names, args.weight_step)
    cal_name, cal_model, shrinkage, val_cal_metrics = fit_calibrator(
        val_frame,
        names,
        weights,
        args.max_calibration_abs,
    )
    val_out = apply_calibrator(val_frame, names, weights, cal_model, shrinkage, args.max_calibration_abs)
    test_out = apply_calibrator(test_frame, names, weights, cal_model, shrinkage, args.max_calibration_abs)

    test_base_metrics = metric_dict(test_out["true_rt"].to_numpy(), test_out["pred_ensemble"].to_numpy())
    test_cal_metrics = metric_dict(test_out["true_rt"].to_numpy(), test_out["pred_calibrated"].to_numpy())

    weights_dict = {name: float(weight) for name, weight in zip(names, weights)}
    summary = {
        "weights": weights_dict,
        "calibrator": cal_name,
        "calibration_shrinkage": shrinkage,
        "max_calibration_abs": args.max_calibration_abs,
        "val_ensemble": val_ensemble_metrics,
        "val_calibrated": val_cal_metrics,
        "test_ensemble": test_base_metrics,
        "test_calibrated": test_cal_metrics,
    }
    (output_dir / "calibrated_ensemble_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    val_out.to_csv(output_dir / "predictions_val_calibrated.csv", index=False)
    test_out.to_csv(output_dir / "predictions_test_calibrated.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
