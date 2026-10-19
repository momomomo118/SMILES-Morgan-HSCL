MCRL Supporting Information and companion data
Prepared: 2026-10-19 (Asia/Taipei)

Start with MCRL_Supporting_Information.pdf. The data/ directory contains UTF-8 CSV tables; configs/ contains model arguments and evaluation protocols; metadata/ contains field definitions, source-file hashes and completed consistency checks.

Contents:
S1 - Fixed SMRT random/scaffold partitions, cleaned PredRet records, system statistics, molecular overlap counts and exact nested-fold assignments.
S2 - Six CCI candidate configurations, three PureGIN configurations, adopted fusion coefficients and recorded runtime environments.
S3 - Retained identity-aligned baseline records and component/RTCL ablation results with saved predictions.
S4 - Scaffold predictions and all 100 PredRet fold results, eight predictor weights per fold and 2,197 outer-test prediction pairs.

All RT values and absolute-error metrics are in seconds. MAPE is in percent; R2 is dimensionless. SD is sample SD (ddof=1). Unrounded values are retained in CSV; PDF values are rounded for display. A system-level fold-mean metric is distinct from its separately named pooled metric.

Keys and joins:
- SMRT: sample_id; the explicit split column gives partition membership.
- PredRet cleaned data: system + sample_id.
- PredRet nested assignments/results: system + outer_fold, with sample_id for record-level tables. Fold identifiers are zero-based.
- S1_predret_fold_assignments.csv joins to S1_predret_cleaned.csv by sample_id.
- Paths in configuration snapshots are normalized to project-relative form; source_inventory.json records the original source-file byte hashes.

Verification:
Run python -B verify_package.py from this directory. This checks package hashes, table row counts, key counts and the included PDF page count using only standard-library modules except the optional PDF page-count check.

This package was assembled from preserved experimental records. Preparation recomputed canonical identifiers and metrics, read model metadata, and checked saved prediction values. It did not train models or execute neural-network inference.

This companion package is distributed with the source-code repository. See ../docs/REPRODUCTION_DATA.md for data preparation commands and the mapping to training inputs.

Suggested manuscript Supporting Information description:
Dataset processing and partition statistics; molecular overlap checks; CCI and PureGIN configurations and ensemble coefficients; baseline and ablation results; scaffold evaluation and nested PredRet protocols and fold-level results (PDF). Machine-readable partition assignments, saved predictions, metrics, configurations and provenance records (CSV and JSON).
