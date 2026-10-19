# SMILES-Morgan-HSCL RT Prediction Reproduction Package

This folder contains the clean reproduction package for the current final model.

Current final result on the unified SMRT test split:

- MAE: 25.8893 s
- RMSE: 52.2262 s
- R2: 0.9124
- MedAE: 13.6490 s

## What Is Included

- `data/unified/`: fixed train/validation/test split used in the experiments.
- `outputs/final_fusion_ccrl_gnn/`: final Scheme A PTDF predictions, metrics, ablation tables, result mapping, and paper support files.
- `outputs/top6_pool_target_power_ensemble/`: CCRL Top6 ensemble, residual refinement candidates, and historical fusion outputs.
- `outputs/pure_gin_seed2026/`, `outputs/pure_gin_seed3407/`, `outputs/pure_gin_seed777/`: PureGIN validation/test predictions used by the final decision fusion.
- `checkpoints/gnn/`: PureGIN checkpoints for seed 2026, 3407, and 777.
- `checkpoints/supcr_tuning/`: recovered CCRL/SupCR top-pool checkpoints needed for fuller reconstruction.
- `code/` and `scripts/`: training, calibration, GNN, and fusion scripts.
- `pretrained/ChemBERTa-zinc-base-v1/`: local ChemBERTa files to avoid downloading the pretrained model again.
- `docs/model_mechanism.md`: final mechanism diagram and PTDF explanation.

## Fast Verification

From this directory:

```powershell
python code\verify_reproduction.py
```

Expected key line:

```text
FINAL Scheme A MAE = 25.889280359643887
```

This verification recomputes metrics from the saved final prediction CSV and checks that the key checkpoints, scripts, and final output files exist.

## Final Model: Scheme A

The final model is a validation-optimized decision fusion:

```text
CCRL multi-view representation
-> Top6 CCRL ensemble
-> PureGIN/GNN topology enhancement
-> validation-optimized PTDF decision fusion
-> RT prediction
```

Final formula:

```text
y_RT =
0.30 * pred_ensemble
+ 0.24 * g2026
+ 0.18 * g3407
+ 0.28 * g777
```

The CCRL branch uses `pred_ensemble` consistently on validation and test. Residual correction is retained as a physicochemical residual refinement candidate and ablation module, but it is not the final base prediction.

Graph Transformer is not included in this clean reproduction package. The selected graph branch is PureGIN.

## Main Final Files

- Final metrics: `outputs/final_fusion_ccrl_gnn/metrics_final_ccrl_gnn_ptdf.json`
- Final predictions: `outputs/final_fusion_ccrl_gnn/predictions_test_final_ccrl_gnn_ptdf.csv`
- Final summary: `outputs/final_fusion_ccrl_gnn/summary_final_ccrl_gnn_ptdf.csv`
- Final ablation: `outputs/final_fusion_ccrl_gnn/ablation_summary_final_ccrl_gnn_ptdf.csv`
- Result mapping: `outputs/final_fusion_ccrl_gnn/result_mapping_final_ccrl_gnn_ptdf.csv`
- Final fusion script: `scripts/run_final_fusion_ccrl_gnn.py`

## Historical Note

The historical `outputs/top6_pool_target_power_ensemble/predictions_test_multi_gnn_blend.csv` result with MAE 26.0520 s is preserved for traceability, but it is no longer the default final result because its validation/test base prediction definitions are not used as the final Scheme A definition.

## Notes

This folder is intended to be self-contained for data, code, saved predictions, checkpoints, and local pretrained model files. A compatible Python environment with common ML packages is still required to execute the full model code.
