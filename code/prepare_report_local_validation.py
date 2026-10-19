import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
REPORT_IN = ROOT / "data" / "RepoRT_prepared" / "report_train_ready.csv"
ORIGINAL_ALL = ROOT / "data" / "unified" / "all.csv"
OUT = ROOT / "data" / "RepoRT_validation"
OUT.mkdir(parents=True, exist_ok=True)


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def infer_smiles_col(df):
    for col in ["smiles", "SMILES", "canonical_smiles"]:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column in {list(df.columns)}")


def infer_rt_col(df):
    for col in ["rt", "RT", "true_rt", "label", "target"]:
        if col in df.columns:
            return col
    raise ValueError(f"No RT column in {list(df.columns)}")


def metrics(y, pred):
    err = np.asarray(pred) - np.asarray(y)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "median_abs_error": float(np.median(np.abs(err))),
        "p95_abs_error": float(np.quantile(np.abs(err), 0.95)),
    }


def descriptor_features(smiles):
    rows = []
    for s in smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append([np.nan] * 12)
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
        ])
    return np.asarray(rows, dtype=np.float32)


def group_split(df):
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=2026)
    train_idx, temp_idx = next(gss1.split(df, groups=df["dataset_id"]))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=3407)
    val_rel, test_rel = next(gss2.split(temp, groups=temp["dataset_id"]))
    val = temp.iloc[val_rel].copy()
    test = temp.iloc[test_rel].copy()

    train["report_split"] = "train"
    val["report_split"] = "val"
    test["report_split"] = "test"
    return train, val, test


def main():
    print("[1/6] reading RepoRT")
    report = pd.read_csv(REPORT_IN, low_memory=False)
    smi_col = infer_smiles_col(report)
    rt_col = infer_rt_col(report)
    report = report.rename(columns={smi_col: "smiles", rt_col: "rt_raw"}).copy()
    report["rt_raw"] = pd.to_numeric(report["rt_raw"], errors="coerce")
    report = report.dropna(subset=["smiles", "rt_raw", "dataset_id"]).reset_index(drop=True)

    # RepoRT stores many datasets in minutes. Preserve raw values and add a seconds-scale target
    # for comparability with the current SMRT-style model outputs.
    report["rt_sec"] = report["rt_raw"] * 60.0

    print("[2/6] canonicalizing RepoRT")
    unique_smi = pd.Series(report["smiles"].astype(str).unique())
    can_map = {s: canonical_smiles(s) for s in unique_smi}
    report["canonical_smiles_nostereo"] = report["smiles"].astype(str).map(can_map)
    report = report[report["canonical_smiles_nostereo"] != ""].reset_index(drop=True)

    print("[3/6] checking overlap with original unified data")
    original = pd.read_csv(ORIGINAL_ALL, low_memory=False)
    orig_smi_col = infer_smiles_col(original)
    orig_unique = pd.Series(original[orig_smi_col].dropna().astype(str).unique())
    orig_can = {s: canonical_smiles(s) for s in orig_unique}
    original_can_set = {v for v in orig_can.values() if v}
    report["overlap_with_original"] = report["canonical_smiles_nostereo"].isin(original_can_set)
    no_overlap = report[~report["overlap_with_original"]].copy().reset_index(drop=True)

    print("[4/6] writing dataset-held-out split")
    train, val, test = group_split(no_overlap)
    split_all = pd.concat([train, val, test], ignore_index=True)

    keep = [
        "sample_id", "source", "dataset_id", "compound_id", "name",
        "smiles", "canonical_smiles_nostereo", "inchikey",
        "rt_raw", "rt_sec", "report_split", "overlap_with_original",
    ]
    keep = [c for c in keep if c in split_all.columns]

    no_overlap.to_csv(OUT / "report_external_no_overlap_all.csv", index=False, encoding="utf-8-sig")
    split_all[keep].to_csv(OUT / "report_dataset_holdout_all.csv", index=False, encoding="utf-8-sig")
    train[keep].to_csv(OUT / "report_dataset_holdout_train.csv", index=False, encoding="utf-8-sig")
    val[keep].to_csv(OUT / "report_dataset_holdout_val.csv", index=False, encoding="utf-8-sig")
    test[keep].to_csv(OUT / "report_dataset_holdout_test.csv", index=False, encoding="utf-8-sig")

    print("[5/6] running light local RDKit baseline")
    rng = np.random.default_rng(2026)
    max_train = min(len(train), 60000)
    train_sample = train.iloc[rng.choice(len(train), size=max_train, replace=False)].copy()

    X_train = descriptor_features(train_sample["smiles"])
    y_train = train_sample["rt_sec"].to_numpy(dtype=np.float32)
    X_val = descriptor_features(val["smiles"])
    X_test = descriptor_features(test["smiles"])

    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=2026,
    )
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    baseline = {
        "target_unit": "seconds, computed as RepoRT raw RT * 60",
        "train_rows_used_for_baseline": int(len(train_sample)),
        "val": metrics(val["rt_sec"], pred_val),
        "test": metrics(test["rt_sec"], pred_test),
    }
    pd.DataFrame({
        "sample_id": test["sample_id"].values,
        "dataset_id": test["dataset_id"].values,
        "smiles": test["smiles"].values,
        "true_rt_sec": test["rt_sec"].values,
        "pred_rdkit_hgb_sec": pred_test,
    }).to_csv(OUT / "predictions_test_rdkit_hgb.csv", index=False, encoding="utf-8-sig")

    print("[6/6] writing summary")
    summary = {
        "input_file": str(REPORT_IN),
        "original_reference_file": str(ORIGINAL_ALL),
        "report_rows_after_basic_clean": int(len(report)),
        "report_unique_dataset_id": int(report["dataset_id"].nunique()),
        "report_unique_canonical_smiles": int(report["canonical_smiles_nostereo"].nunique()),
        "overlap_rows_with_original": int(report["overlap_with_original"].sum()),
        "overlap_unique_canonical_smiles_with_original": int(
            report.loc[report["overlap_with_original"], "canonical_smiles_nostereo"].nunique()
        ),
        "external_no_overlap_rows": int(len(no_overlap)),
        "split": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_dataset_id": int(train["dataset_id"].nunique()),
            "val_dataset_id": int(val["dataset_id"].nunique()),
            "test_dataset_id": int(test["dataset_id"].nunique()),
        },
        "rdkit_hgb_baseline": baseline,
        "recommended_server_next": [
            "zero-shot current final model on report_dataset_holdout_test.csv",
            "calibration-only on report_dataset_holdout_train/val/test.csv",
            "RepoRT fine-tuning or retraining on report_dataset_holdout_train.csv",
            "compare against this RDKit HGB local baseline and SMILES-only baseline",
        ],
    }
    (OUT / "report_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = f"""# RepoRT Local Validation Preparation

## What Was Prepared

- RepoRT rows after cleaning: {summary['report_rows_after_basic_clean']}
- RepoRT datasets: {summary['report_unique_dataset_id']}
- Unique canonical SMILES: {summary['report_unique_canonical_smiles']}
- Rows overlapping with original SMRT-style data: {summary['overlap_rows_with_original']}
- Non-overlap rows for external validation: {summary['external_no_overlap_rows']}

## Dataset-Held-Out Split

| Split | Rows | Dataset IDs |
|---|---:|---:|
| Train | {summary['split']['train_rows']} | {summary['split']['train_dataset_id']} |
| Val | {summary['split']['val_rows']} | {summary['split']['val_dataset_id']} |
| Test | {summary['split']['test_rows']} | {summary['split']['test_dataset_id']} |

## Local RDKit HGB Baseline

Target unit is seconds, computed as `RepoRT raw RT * 60`.

| Split | MAE | RMSE | R2 | MedAE | P95 |
|---|---:|---:|---:|---:|---:|
| Val | {baseline['val']['mae']:.4f} | {baseline['val']['rmse']:.4f} | {baseline['val']['r2']:.4f} | {baseline['val']['median_abs_error']:.4f} | {baseline['val']['p95_abs_error']:.4f} |
| Test | {baseline['test']['mae']:.4f} | {baseline['test']['rmse']:.4f} | {baseline['test']['r2']:.4f} | {baseline['test']['median_abs_error']:.4f} | {baseline['test']['p95_abs_error']:.4f} |

## Recommended Server Experiments

1. Zero-shot: apply the current final 26.05s model to `report_dataset_holdout_test.csv`.
2. Calibration-only: fit only a small calibrator on train/val, keep the original model frozen.
3. Fine-tuning/retraining: train the multimodal model on RepoRT train and evaluate on held-out datasets.
4. Report comparison: RDKit HGB, SMILES-only, Morgan-only, SMILES+Morgan SupCR, final ensemble.
"""
    (OUT / "report_validation_plan.md").write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
