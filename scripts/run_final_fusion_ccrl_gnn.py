import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "final_fusion_ccrl_gnn"
TOP6 = ROOT / "outputs" / "top6_pool_target_power_ensemble"

BRANCHES = [
    {
        "name": "CCRL Top6 ensemble",
        "short": "ccrl_ensemble",
        "val_file": TOP6 / "predictions_val_calibrated.csv",
        "test_file": TOP6 / "predictions_test_calibrated.csv",
        "column": "pred_ensemble",
        "provenance": "Top6 CCRL ensemble using unified pred_ensemble on validation and test.",
    },
    {
        "name": "PureGIN seed 2026",
        "short": "g2026",
        "val_file": ROOT / "outputs" / "pure_gin_seed2026" / "predictions_val_gnn.csv",
        "test_file": ROOT / "outputs" / "pure_gin_seed2026" / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "PureGIN topological GNN predictor, seed 2026.",
    },
    {
        "name": "PureGIN seed 3407",
        "short": "g3407",
        "val_file": ROOT / "outputs" / "pure_gin_seed3407" / "predictions_val_gnn.csv",
        "test_file": ROOT / "outputs" / "pure_gin_seed3407" / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "PureGIN topological GNN predictor, seed 3407.",
    },
    {
        "name": "PureGIN seed 777",
        "short": "g777",
        "val_file": ROOT / "outputs" / "pure_gin_seed777" / "predictions_val_gnn.csv",
        "test_file": ROOT / "outputs" / "pure_gin_seed777" / "predictions_test_gnn.csv",
        "column": "pred_rt",
        "provenance": "PureGIN topological GNN predictor, seed 777.",
    },
]

FINAL_WEIGHTS_REFERENCE = {
    "ccrl_ensemble": 0.30,
    "g2026": 0.24,
    "g3407": 0.18,
    "g777": 0.28,
}


def metric_dict(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(abs_err)),
        "mape_pct": float(np.mean(abs_err / np.maximum(np.abs(y), 1e-12)) * 100.0),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
        "medae": float(np.median(abs_err)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "max_abs_error": float(np.max(abs_err)),
        "mean_signed_error": float(np.mean(err)),
        "error_std": float(np.std(err)),
        "within_30s_pct": float(np.mean(abs_err <= 30.0) * 100.0),
        "within_60s_pct": float(np.mean(abs_err <= 60.0) * 100.0),
        "within_120s_pct": float(np.mean(abs_err <= 120.0) * 100.0),
    }


def load_split(split):
    merged = None
    for spec in BRANCHES:
        path = spec[f"{split}_file"]
        df = pd.read_csv(path)
        need = {"sample_id", "true_rt", spec["column"]}
        missing = need.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        cols = ["sample_id", "true_rt", spec["column"]]
        if "smiles" in df.columns:
            cols.insert(1, "smiles")
        part = df[cols].copy().rename(
            columns={"true_rt": f"true_rt_{spec['short']}", spec["column"]: spec["short"]}
        )
        if merged is None:
            merged = part
        else:
            if "smiles" in part.columns and "smiles" in merged.columns:
                part = part.drop(columns=["smiles"])
            merged = merged.merge(part, on="sample_id", how="inner")

    true_cols = [c for c in merged.columns if c.startswith("true_rt_")]
    y = merged[true_cols[0]].to_numpy(float)
    for col in true_cols[1:]:
        if not np.allclose(y, merged[col].to_numpy(float), atol=1e-3, rtol=0.0):
            raise ValueError(f"true_rt mismatch in {split}: {col}")
    merged["true_rt"] = y
    return merged.drop(columns=true_cols)


def mae(y, pred):
    return float(np.mean(np.abs(pred - y)))


def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def optimize_pairwise_mae(y, x):
    n, k = x.shape
    starts = [np.ones(k) / k, np.eye(k)[int(np.argmin([mae(y, x[:, i]) for i in range(k)]))]]
    starts.append(np.array([FINAL_WEIGHTS_REFERENCE[c] for c in [b["short"] for b in BRANCHES]], dtype=float))
    rng = np.random.default_rng(20260814)
    starts.extend(rng.dirichlet(np.ones(k), size=32))

    best_w = None
    best_mae = np.inf

    for start in starts:
        w = np.asarray(start, dtype=float)
        w = np.maximum(w, 0.0)
        w = w / w.sum()
        pred = x @ w
        current = mae(y, pred)

        for _ in range(300):
            improved = False
            for i in range(k):
                for j in range(i + 1, k):
                    mass = w[i] + w[j]
                    if mass <= 1e-15:
                        continue
                    base_pred = pred - w[i] * x[:, i] - w[j] * x[:, j] + mass * x[:, j]
                    direction = x[:, i] - x[:, j]
                    residual0 = base_pred - y
                    nz = np.abs(direction) > 1e-12

                    candidates = [0.0, mass]
                    if np.any(nz):
                        roots = -residual0[nz] / direction[nz]
                        root_weights = np.abs(direction[nz])
                        ok = (roots >= 0.0) & (roots <= mass)
                        if np.any(ok):
                            candidates.append(weighted_median(roots[ok], root_weights[ok]))

                    pair_alpha = w[i]
                    pair_mae = current
                    for alpha in candidates:
                        cand_pred = base_pred + alpha * direction
                        cand_mae = mae(y, cand_pred)
                        if cand_mae + 1e-12 < pair_mae:
                            pair_alpha = float(alpha)
                            pair_mae = cand_mae

                    if pair_mae + 1e-12 < current:
                        w[i] = pair_alpha
                        w[j] = mass - pair_alpha
                        pred = base_pred + pair_alpha * direction
                        current = pair_mae
                        improved = True
            if not improved:
                break

        if current < best_mae:
            best_mae = current
            best_w = w.copy()

    best_w[np.abs(best_w) < 1e-10] = 0.0
    return best_w / best_w.sum()


def add_pred(df, weights, out_col):
    cols = [b["short"] for b in BRANCHES]
    df[out_col] = df[cols].to_numpy(float) @ np.asarray(weights, dtype=float)
    return df


def canonical_prediction_frame(df):
    keep = ["sample_id"]
    if "smiles" in df.columns:
        keep.append("smiles")
    out = df[keep].copy()
    out["true_rt"] = df["true_rt"]
    out["pred_ccrl_ensemble"] = df["ccrl_ensemble"]
    out["pred_g2026"] = df["g2026"]
    out["pred_g3407"] = df["g3407"]
    out["pred_g777"] = df["g777"]
    out["pred_final"] = df["pred_final_ccrl_gnn_ptdf"]
    return out


def load_prediction(path, pred_col=None):
    df = pd.read_csv(path)
    if pred_col is None:
        for candidate in ["pred_rt", "pred", "pred_ensemble", "pred_calibrated", "prediction"]:
            if candidate in df.columns:
                pred_col = candidate
                break
    if pred_col is None:
        raise ValueError(f"Cannot find prediction column in {path}")
    return df, pred_col


def source_metrics(path, pred_col=None):
    df, col = load_prediction(path, pred_col)
    return metric_dict(df["true_rt"], df[col]), col


def get_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_ablation(final_val_metrics, final_test_metrics):
    calibrated = get_json(TOP6 / "calibrated_ensemble_metrics.json")
    gbdt = get_json(TOP6 / "metrics_gbdt_blend.json")
    gnn_residual = get_json(TOP6 / "metrics_gnn_blend_residual.json")

    val = load_split("val")
    test = load_split("test")
    cols = [b["short"] for b in BRANCHES]

    rows = []

    def append(name, source, pred_definition, validation_protocol, weights, val_metrics, test_metrics, role):
        rows.append(
            {
                "stage": name,
                "prediction_source": source,
                "prediction_definition": pred_definition,
                "validation_protocol": validation_protocol,
                "fusion_weights": json.dumps(weights, sort_keys=True) if isinstance(weights, dict) else weights,
                "val_mae": val_metrics.get("mae") if val_metrics else "",
                "val_rmse": val_metrics.get("rmse") if val_metrics else "",
                "val_r2": val_metrics.get("r2") if val_metrics else "",
                "val_medae": val_metrics.get("medae", val_metrics.get("median_abs_error")) if val_metrics else "",
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_medae": test_metrics.get("medae", test_metrics.get("median_abs_error")),
                "mae_drop_vs_ccrl_backbone": "",
                "mae_improvement_pct_vs_ccrl_backbone": "",
                "role": role,
            }
        )

    base_model_cols = [
        c
        for c in pd.read_csv(TOP6 / "predictions_test_calibrated.csv", nrows=1).columns
        if c.startswith("pred_target_power_")
    ]
    single_metrics = []
    val_cal = pd.read_csv(TOP6 / "predictions_val_calibrated.csv")
    test_cal = pd.read_csv(TOP6 / "predictions_test_calibrated.csv")
    for col in base_model_cols:
        single_metrics.append((col, metric_dict(test_cal["true_rt"], test_cal[col])))
    backbone_col, backbone_test_metrics = min(single_metrics, key=lambda item: item[1]["mae"])
    backbone_val_metrics = metric_dict(val_cal["true_rt"], val_cal[backbone_col])

    append(
        "A. CCRL backbone",
        str((TOP6 / "predictions_test_calibrated.csv").relative_to(ROOT)),
        f"Single best CCRL checkpoint prediction column: {backbone_col}",
        "Existing validation/test split; no fusion.",
        {"selected_column": backbone_col},
        backbone_val_metrics,
        backbone_test_metrics,
        "backbone baseline",
    )
    append(
        "B. Top6 ensemble",
        str((TOP6 / "predictions_test_calibrated.csv").relative_to(ROOT)),
        "Top6 CCRL ensemble pred_ensemble.",
        "Top6 weights from calibrated_ensemble_metrics.json; same pred_ensemble definition on val/test.",
        calibrated["weights"],
        metric_dict(val_cal["true_rt"], val_cal["pred_ensemble"]),
        metric_dict(test_cal["true_rt"], test_cal["pred_ensemble"]),
        "CCRL ensemble baseline",
    )
    append(
        "C. Top6 + residual correction",
        str((TOP6 / "predictions_test_calibrated.csv").relative_to(ROOT)),
        "Top6 CCRL pred_calibrated.",
        "Calibration selected on validation; same pred_calibrated definition on val/test.",
        {"residual_ccrl": 1.0},
        metric_dict(val_cal["true_rt"], val_cal["pred_calibrated"]),
        metric_dict(test_cal["true_rt"], test_cal["pred_calibrated"]),
        "physicochemical residual refinement candidate",
    )
    append(
        "D. Top6 + RDKit/GBDT residual fusion",
        str((TOP6 / "predictions_test_gbdt_blend.csv").relative_to(ROOT)),
        "OOF residual base blended with descriptor GBDT prediction.",
        "Historical residual/GBDT validation in metrics_gbdt_blend.json.",
        {"w_base": 0.92, "w_gbdt": 0.08},
        gbdt["selected"]["val_metrics"],
        gbdt["blend_test"],
        "physicochemical residual refinement ablation",
    )
    g2026_only_w = {"ccrl_ensemble": 0.41, "g2026": 0.59}
    g2026_val_pred = val["ccrl_ensemble"] * 0.41 + val["g2026"] * 0.59
    g2026_test_pred = test["ccrl_ensemble"] * 0.41 + test["g2026"] * 0.59
    append(
        "E. Top6 + PureGIN",
        "outputs/top6_pool_target_power_ensemble/predictions_test_val_selected_gnn_blend.csv",
        "Validation-selected two-branch blend using pred_ensemble and PureGIN seed 2026.",
        "Recomputed here with consistent pred_ensemble + g2026 definitions for val/test.",
        g2026_only_w,
        metric_dict(val["true_rt"], g2026_val_pred),
        metric_dict(test["true_rt"], g2026_test_pred),
        "single topological branch ablation",
    )
    multi_w = FINAL_WEIGHTS_REFERENCE
    multi_vec = np.asarray([multi_w[c] for c in cols], dtype=float)
    append(
        "F. Top6 + PureGIN + multi-GNN fusion",
        "outputs/top6_pool_target_power_ensemble/predictions_test_multi_gnn_blend.csv",
        "Historical four-branch weight set reapplied to unified pred_ensemble + g2026/g3407/g777 definitions.",
        "Weights selected on validation; historical 26.0520 test file is not used as default because its test base was residual-corrected.",
        multi_w,
        metric_dict(val["true_rt"], val[cols].to_numpy(float) @ multi_vec),
        metric_dict(test["true_rt"], test[cols].to_numpy(float) @ multi_vec),
        "historical weight set under corrected definition",
    )
    append(
        "G. Final CCRL ensemble + GNN PTDF",
        str((OUT_DIR / "predictions_test_final_ccrl_gnn_ptdf.csv").relative_to(ROOT)),
        "Final validation-optimized PTDF decision fusion over pred_ensemble + g2026/g3407/g777.",
        "Weights optimized only on validation MAE; fixed weights evaluated once on test.",
        multi_w,
        final_val_metrics,
        final_test_metrics,
        "final model",
    )
    append(
        "Residual module. CCRL + residual correction + GNN fusion",
        str((TOP6 / "predictions_test_gnn_blend_residual.csv").relative_to(ROOT)),
        "Residual-corrected CCRL/GNN fusion retained as auxiliary calibration module.",
        "Historical validation/test metrics from metrics_gnn_blend_residual.json.",
        {"residual_module": "auxiliary"},
        gnn_residual["selected"]["val_metrics"],
        gnn_residual["residual_test"],
        "auxiliary residual calibration analysis",
    )

    baseline = rows[0]["test_mae"]
    for row in rows:
        drop = baseline - row["test_mae"]
        row["mae_drop_vs_ccrl_backbone"] = drop
        row["mae_improvement_pct_vs_ccrl_backbone"] = 100.0 * drop / baseline
    return rows


def scan_result_references():
    patterns = ["26.0520", "26.052", "26.6091", "26.609", "25.8893", "25.889"]
    exts = {".csv", ".txt", ".json", ".md", ".tex", ".py"}
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="gbk")
            except UnicodeDecodeError:
                continue
        hits = [p for p in patterns if p in text]
        if hits:
            relative = str(path.relative_to(ROOT))
            normalized = relative.replace("\\", "/")
            hit_text = ";".join(sorted(set(hits)))
            if (
                "/predictions_" in normalized
                or normalized.startswith("data/")
                or normalized.startswith("external_literature_code/")
                or normalized.startswith("checkpoints/")
            ):
                action = "raw numeric hit in data/prediction file; no citation update needed"
            elif "Historical" in text or "historical" in text or "not default" in text or "traceability" in text:
                action = "historical/non-default reference retained for traceability"
            elif "25.889" in hit_text and "26.052" not in hit_text and "26.609" not in hit_text:
                action = "confirm this cites final Scheme A result"
            else:
                action = "update final-model citation to outputs/final_fusion_ccrl_gnn"
            rows.append(
                {
                    "file": relative,
                    "matched_values": hit_text,
                    "recommended_action": action,
                }
            )
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = load_split("val")
    test = load_split("test")
    cols = [b["short"] for b in BRANCHES]

    weights = optimize_pairwise_mae(val["true_rt"].to_numpy(float), val[cols].to_numpy(float))
    weight_map = {c: float(w) for c, w in zip(cols, weights)}

    add_pred(val, weights, "pred_final_ccrl_gnn_ptdf")
    add_pred(test, weights, "pred_final_ccrl_gnn_ptdf")
    final_val_metrics = metric_dict(val["true_rt"], val["pred_final_ccrl_gnn_ptdf"])
    final_test_metrics = metric_dict(test["true_rt"], test["pred_final_ccrl_gnn_ptdf"])

    val.to_csv(OUT_DIR / "predictions_val_final_ccrl_gnn_ptdf.csv", index=False)
    test.to_csv(OUT_DIR / "predictions_test_final_ccrl_gnn_ptdf.csv", index=False)
    canonical_prediction_frame(val).to_csv(OUT_DIR / "predictions_val_final.csv", index=False)
    canonical_prediction_frame(test).to_csv(OUT_DIR / "predictions_test_final.csv", index=False)

    branch_rows = []
    for spec, w in zip(BRANCHES, weights):
        branch_rows.append(
            {
                "branch_name": spec["name"],
                "prediction_source_file": f"validation={spec['val_file'].relative_to(ROOT)}; test={spec['test_file'].relative_to(ROOT)}",
                "prediction_column": spec["column"],
                "data_provenance": spec["provenance"],
                "optimized_weight": float(w),
                "validation_performance": json.dumps(metric_dict(val["true_rt"], val[spec["short"]]), sort_keys=True),
                "test_performance": json.dumps(metric_dict(test["true_rt"], test[spec["short"]]), sort_keys=True),
            }
        )
    pd.DataFrame(branch_rows).to_csv(OUT_DIR / "summary_final_ccrl_gnn_ptdf.csv", index=False)

    ablation_rows = build_ablation(final_val_metrics, final_test_metrics)
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "ablation_summary_final_ccrl_gnn_ptdf.csv", index=False)

    result_map = [
        {
            "result_name": "Final Scheme A: CCRL ensemble + GNN PTDF",
            "prediction_source": "outputs/final_fusion_ccrl_gnn/predictions_test_final_ccrl_gnn_ptdf.csv",
            "validation_protocol": "weights optimized on validation MAE using pred_ensemble + g2026/g3407/g777; test evaluated once",
            "fusion_weights": json.dumps(weight_map, sort_keys=True),
            "test_metrics": json.dumps(final_test_metrics, sort_keys=True),
            "default_final": "yes",
        },
        {
            "result_name": "Historical multi-GNN blend MAE=26.0520",
            "prediction_source": "outputs/top6_pool_target_power_ensemble/predictions_test_multi_gnn_blend.csv",
            "validation_protocol": "not default; historical file blended residual-corrected test base while validation used Top6 base",
            "fusion_weights": json.dumps(FINAL_WEIGHTS_REFERENCE, sort_keys=True),
            "test_metrics": json.dumps(
                source_metrics(TOP6 / "predictions_test_multi_gnn_blend.csv", "pred_rt")[0], sort_keys=True
            ),
            "default_final": "no",
        },
        {
            "result_name": "Five-branch residual candidate MAE=26.6091",
            "prediction_source": "outputs/final_fusion_five_branch/predictions_test_five_branch_ptdf_fusion.csv",
            "validation_protocol": "residual corrected CCRL treated as extra candidate branch",
            "fusion_weights": json.dumps({"ccrl_ensemble": 0.0, "residual_ccrl": 0.46, "g2026": 0.20, "g3407": 0.12, "g777": 0.22}, sort_keys=True),
            "test_metrics": json.dumps(
                source_metrics(ROOT / "outputs" / "final_fusion_five_branch" / "predictions_test_five_branch_ptdf_fusion.csv", "pred_five_branch_ptdf_fusion")[0],
                sort_keys=True,
            )
            if (ROOT / "outputs" / "final_fusion_five_branch" / "predictions_test_five_branch_ptdf_fusion.csv").exists()
            else "",
            "default_final": "no",
        },
    ]
    pd.DataFrame(result_map).to_csv(OUT_DIR / "result_mapping_final_ccrl_gnn_ptdf.csv", index=False)

    references = scan_result_references()
    pd.DataFrame(references).to_csv(OUT_DIR / "paper_result_reference_scan.csv", index=False)

    paper_updates = [
        {
            "paper_section": "Abstract",
            "required_change": "Report final Scheme A MAE/RMSE/R2/MedAE from outputs/final_fusion_ccrl_gnn; remove 26.0520 as final default.",
        },
        {
            "paper_section": "Methods / PTDF",
            "required_change": "Define PTDF as validation-optimized decision fusion over CCRL ensemble, physicochemical refinement candidates, and topological GNN predictors.",
        },
        {
            "paper_section": "Results",
            "required_change": "Use Final CCRL ensemble + GNN PTDF metrics and weights; describe 26.0520 as historical non-default result only.",
        },
        {
            "paper_section": "Ablation",
            "required_change": "Use ablation_summary_final_ccrl_gnn_ptdf.csv; residual correction is an auxiliary calibration/refinement ablation.",
        },
        {
            "paper_section": "Tables/Figures",
            "required_change": "Regenerate tables from final_ablation and result_mapping files; final workflow must not show residual correction as mandatory.",
        },
    ]
    pd.DataFrame(paper_updates).to_csv(OUT_DIR / "paper_sections_to_update.csv", index=False)

    report = {
        "final_model_definition": "Scheme A: CCRL Top6 ensemble pred_ensemble plus PureGIN seeds 2026/3407/777, fused by validation-optimized PTDF decision fusion.",
        "final_fusion_weights": weight_map,
        "final_validation_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "residual_repositioning": "Residual correction is retained as a physicochemical residual refinement module and ablation candidate, not the final base prediction.",
        "outputs": {
            "metrics": "outputs/final_fusion_ccrl_gnn/metrics_final_ccrl_gnn_ptdf.json",
            "summary": "outputs/final_fusion_ccrl_gnn/summary_final_ccrl_gnn_ptdf.csv",
            "ablation": "outputs/final_fusion_ccrl_gnn/ablation_summary_final_ccrl_gnn_ptdf.csv",
            "mapping": "outputs/final_fusion_ccrl_gnn/result_mapping_final_ccrl_gnn_ptdf.csv",
            "paper_scan": "outputs/final_fusion_ccrl_gnn/paper_result_reference_scan.csv",
        },
    }
    (OUT_DIR / "metrics_final_ccrl_gnn_ptdf.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    weights_payload = {
        "final_scheme": "Scheme A: CCRL Top6 ensemble + GNN decision fusion",
        "validation_protocol": "weights optimized on validation MAE only; test labels not used for weight selection",
        "objective": "minimum validation MAE under wi >= 0 and sum(wi)=1",
        "branches": [
            {
                "branch_name": spec["name"],
                "canonical_column": {
                    "ccrl_ensemble": "pred_ccrl_ensemble",
                    "g2026": "pred_g2026",
                    "g3407": "pred_g3407",
                    "g777": "pred_g777",
                }[spec["short"]],
                "source_column": spec["column"],
                "validation_source": str(spec["val_file"].relative_to(ROOT)),
                "test_source": str(spec["test_file"].relative_to(ROOT)),
                "weight": float(weight_map[spec["short"]]),
            }
            for spec in BRANCHES
        ],
        "weights": weight_map,
        "validation_mae": final_val_metrics["mae"],
    }
    (OUT_DIR / "weights.json").write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")
    metrics_payload = {
        "MAE": final_test_metrics["mae"],
        "RMSE": final_test_metrics["rmse"],
        "R2": final_test_metrics["r2"],
        "MedAE": final_test_metrics["medae"],
        "MAPE": final_test_metrics["mape_pct"],
        "validation_MAE": final_val_metrics["mae"],
        "prediction_file": "outputs/final_fusion_ccrl_gnn/predictions_test_final.csv",
        "final_prediction_column": "pred_final",
    }
    (OUT_DIR / "metrics_final.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    ablation_dir = ROOT / "outputs" / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation_rows).to_csv(ablation_dir / "ablation_summary.csv", index=False)
    (ablation_dir / "README.md").write_text(
        "Final ablation summary for Scheme A. Residual correction rows are auxiliary/ablation only and are not part of the deployed final model.\n",
        encoding="utf-8",
    )

    md = [
        "# Final Scheme A Experiment Report",
        "",
        "## Final model definition",
        "CCRL multi-view representation -> Top6 CCRL ensemble -> PureGIN/GNN topology enhancement -> validation optimized PTDF decision fusion -> RT prediction.",
        "",
        "Residual correction is retained as a physicochemical residual refinement candidate and ablation module, not as the final mandatory base.",
        "",
        "## Final fusion weights",
    ]
    for key, value in weight_map.items():
        md.append(f"- {key}: {value:.6f}")
    md.extend(
        [
            "",
            "## Final test metrics",
            f"- MAE: {final_test_metrics['mae']:.6f} s",
            f"- RMSE: {final_test_metrics['rmse']:.6f} s",
            f"- R2: {final_test_metrics['r2']:.6f}",
            f"- MedAE: {final_test_metrics['medae']:.6f} s",
            "",
            "## Updated ablation table",
            "See `ablation_summary_final_ccrl_gnn_ptdf.csv`.",
            "",
            "## Residual correction contribution analysis",
            "Residual correction improves validation performance of the Top6 CCRL branch but does not become the final base under Scheme A. It remains useful as an auxiliary physicochemical calibration module and as an ablation showing RDKit/GBDT residual refinement effects.",
            "",
            "## Paper sections requiring updates",
            "See `paper_sections_to_update.csv` and `paper_result_reference_scan.csv`.",
            "",
            "## Old-to-new result mapping",
            "See `result_mapping_final_ccrl_gnn_ptdf.csv`.",
        ]
    )
    (OUT_DIR / "final_experiment_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
