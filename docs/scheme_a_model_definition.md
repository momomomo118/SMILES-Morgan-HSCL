# Scheme A Final Model Definition

Freeze date: 2026-08-16 01:15:52 +0800

Model name: Scheme A - CCRL Top6 ensemble + GNN decision fusion

Final formula:

```text
y_final =
0.30 * pred_ccrl_ensemble
+0.24 * pred_g2026
+0.18 * pred_g3407
+0.28 * pred_g777
```

Input branches:

| branch | canonical final column | source definition | weight |
|---|---|---|---:|
| CCRL Top6 ensemble | `pred_ccrl_ensemble` | Top6 CCRL ensemble `pred_ensemble` | 0.30 |
| PureGIN seed 2026 | `pred_g2026` | raw GNN `pred_rt`, canonicalized in final file | 0.24 |
| PureGIN seed 3407 | `pred_g3407` | raw GNN `pred_rt`, canonicalized in final file | 0.18 |
| PureGIN seed 777 | `pred_g777` | raw GNN `pred_rt`, canonicalized in final file | 0.28 |

Final base:

```text
pred_ccrl_ensemble
```

Forbidden as final input:

```text
pred_residual_corrected
pred_calibrated
pred_base
pred_rt
```

Residual positioning:

Residual correction is retained only as auxiliary calibration / ablation / historical analysis. It is not part of the deployed final Scheme A prediction pipeline.
