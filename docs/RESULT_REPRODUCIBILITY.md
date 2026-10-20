# Data, results, and workflow resources

## Resource map

| Task | Resources | Entry point |
| --- | --- | --- |
| Read the Supporting Information | SI PDF, 25 CSV tables, configurations, data dictionary | supporting_information/README.txt |
| Recalculate archived metrics | Fixed assignments and saved SMRT, scaffold, baseline, ablation, and PredRet predictions | tools/verify_saved_predictions.py |
| Prepare fixed training inputs | SMRT partitions and nested PredRet assignments | tools/prepare_reproduction_data.py |
| Obtain SMRT model weights | Six CCI candidates and three PureGIN checkpoints | tools/download_model_artifacts.py |
| Perform fixed-weight SMRT inference | Five active CCI members, three PureGIN members, pinned ChemBERTa, fixed training RTs | tools/infer_mcrl_from_checkpoints.py |
| Train or evaluate models | Recorded scripts, seeds, member arguments, and environments | docs/TRAINING_AND_INFERENCE.md |

## Archived results and experiment scope

The archived final SMRT prediction has MAE 25.8893 s, RMSE 52.2262 s, and R² 0.9124. Its components are combined at prediction level using the coefficients documented in README.md and configs/scheme_a/model_artifacts.json. The CCI ensemble and individual PureGIN predictions are separately named in the SI tables.

The SI includes fixed-coefficient scaffold predictions, component and three-seed RTCL ablation records, baseline prediction tables, and PredRet results for ten systems and 100 outer folds. PredRet uses labelled target-system training samples and inner-validation selection; fold-mean and pooled metrics are separately reported.

## Metric conventions

- Targets, MAE, RMSE, and MedAE are in seconds; MAPE is percent.
- R² is 1−SSE/SST.
- Recalculation uses fixed assignment-table RT labels and checks agreement with embedded prediction labels to 0.001 s.
- Sample standard deviation uses ddof=1.
- Conventional median averages the middle two values for an even sample count.
- Structural separation checks use the archived canonical-SMILES and Murcko keys.

The archived-prediction verifier reads saved records and calculates metrics. Neural inference and training are separate commands in the execution guide.

## Model resources

The checkpoint release distributes the original SMRT CCI and PureGIN files together with a machine-readable member inventory and SHA256SUMS.txt. The pinned ChemBERTa backbone has a separate official download location. PredRet model fitting uses the target-system and fold assignments documented in the SI.
