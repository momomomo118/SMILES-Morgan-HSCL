import json
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, HuberRegressor

RDLogger.DisableLog("rdApp.*")

base = Path("src/outputs/supcr_tuning/top6_pool_target_power_ensemble")
val_base = pd.read_csv(base / "predictions_val_calibrated.csv")
test_blend = pd.read_csv(base / "predictions_test_val_selected_gnn_blend.csv")
val_gnn = pd.read_csv("src/outputs/gnn/pure_gin_seed2026/predictions_val_gnn.csv")
metrics_blend = json.loads((base / "metrics_val_selected_gnn_blend.json").read_text())

out_pred = base / "predictions_test_gnn_blend_residual.csv"
out_metrics = base / "metrics_gnn_blend_residual.json"

def rdkit_features(smiles):
    rows = []
    for s in smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append({})
            continue
        rows.append({
            "smiles_len": len(s),
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rot_bonds": Lipinski.NumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        })
    return pd.DataFrame(rows).fillna(0.0)

def metrics(y, p):
    e = p - y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "r2": float(r2_score(y, p)),
        "median_abs_error": float(np.median(np.abs(e))),
        "p95_abs_error": float(np.quantile(np.abs(e), 0.95)),
        "max_abs_error": float(np.max(np.abs(e))),
    }

def features(df, pred_col="pred_rt"):
    x = pd.DataFrame({
        "pred": df[pred_col],
        "pred_sq": df[pred_col] ** 2,
        "pred_sqrt": np.sqrt(np.clip(df[pred_col], 0, None)),
    })
    if "pred_ccrl_ensemble" in df:
        x["pred_ccrl_ensemble"] = df["pred_ccrl_ensemble"]
    if "pred_residual_corrected" in df:
        x["pred_residual_corrected"] = df["pred_residual_corrected"]
    if "pred_gnn" in df:
        x["pred_gnn"] = df["pred_gnn"]
        reference = df.get("pred_residual_corrected", df.get("pred_ccrl_ensemble", df[pred_col]))
        x["gnn_minus_reference"] = df["pred_gnn"] - reference
    return pd.concat([x, rdkit_features(df["smiles"])], axis=1)

# Reconstruct validation blend using selected weights.
w_current = metrics_blend["selected"]["w_current_best"]
w_gnn = metrics_blend["selected"]["w_gnn"]

v = val_base[["sample_id", "smiles", "true_rt", "pred_ensemble"]].rename(columns={"pred_ensemble": "pred_ccrl_ensemble"})
v = v.merge(val_gnn[["sample_id", "pred_rt"]].rename(columns={"pred_rt": "pred_gnn"}), on="sample_id", how="inner")
v["pred_rt"] = w_current * v["pred_ccrl_ensemble"] + w_gnn * v["pred_gnn"]

t = test_blend.copy()
if "pred_base" in t.columns and "pred_residual_corrected" not in t.columns:
    t = t.rename(columns={"pred_base": "pred_residual_corrected"})

x_val = features(v)
x_test = features(t)

y_val = v["true_rt"].to_numpy()
base_val = v["pred_rt"].to_numpy()
res_val = y_val - base_val

y_test = t["true_rt"].to_numpy()
base_test = t["pred_rt"].to_numpy()

models = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=150.0)),
    "ridge_poly2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=300.0)),
    "huber": make_pipeline(StandardScaler(), HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000)),
}

clips = [10, 15, 20, 25, 30]
shrinks = np.linspace(0.0, 0.35, 15)

best = None

for name, model in models.items():
    model.fit(x_val, res_val)
    raw_val_all = model.predict(x_val)
    raw_test_all = model.predict(x_test)

    for clip in clips:
        raw_val = np.clip(raw_val_all, -clip, clip)
        raw_test = np.clip(raw_test_all, -clip, clip)

        for shrink in shrinks:
            pred_val = base_val + shrink * raw_val
            m_val = metrics(y_val, pred_val)

            if best is None or m_val["mae"] < best["val_metrics"]["mae"]:
                pred_test = base_test + shrink * raw_test
                best = {
                    "model": name,
                    "clip": float(clip),
                    "shrinkage": float(shrink),
                    "val_metrics": m_val,
                    "test_pred": pred_test,
                }

pred_test = best.pop("test_pred")

summary = {
    "selected": best,
    "base_val": metrics(y_val, base_val),
    "base_test_gnn_blend": metrics(y_test, base_test),
    "residual_test": metrics(y_test, pred_test),
}

out = t.copy()
out["pred_before_residual"] = base_test
out["pred_rt"] = pred_test
out["correction_delta"] = pred_test - base_test
out.to_csv(out_pred, index=False)

out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("wrote:", out_pred)
