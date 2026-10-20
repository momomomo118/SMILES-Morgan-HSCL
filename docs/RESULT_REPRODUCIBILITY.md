# What can be reproduced, and at which level?

This document separates archived-prediction recalculation from neural execution and retraining. It applies to the fixed public source commit b5caa73bace8f280d12e79ca062301c613922855 and the subsequently published source supplement and SMRT checkpoint release.

| Level | Supported material | Limit |
| --- | --- | --- |
| Read manuscript/SI | The 19-page SI, 25 CSV tables, configurations, data dictionary, and provenance | The main manuscript is not part of this source snapshot |
| Recompute saved metrics | Fixed SMRT/scaffold labels; final fusion/CCI/GIN predictions; ten component runs; three identity-aligned baselines; 2,197 paired PredRet predictions and 100 fold results | Does not establish original model identity for unavailable historical baseline artifacts |
| Run checkpoint inference | Matching model source and pinned ChemBERTa resource; explicit inventory of six CCI candidates and three GINs | Trained SMRT weights are available as separately hashed release assets; new standalone inference is statically checked, not dynamically validated here |
| Retrain | Fixed train/validation/test data, scaffold and nested target assignments, scripts, arguments, seeds, and operation-specific environments | Historical hardware/software and some training assets are incomplete; no bitwise or identical-score guarantee |

## Archived results

The fixed final SMRT predictor achieves approximately MAE 25.8893 s, RMSE 52.2262 s, and R² 0.9124. It is distinct from the CCI ensemble (~29.8620 s) and individual GIN models. The fixed-coefficient scaffold archived prediction gives MAE 35.1168 s. Three of five active scaffold CCI member training archives are incomplete.

Controlled residual-cross-attention RTCL ablations use seeds 2026, 777, and 3407. The difference is MAE_without − MAE_with, with mean −0.0305 s and sample SD 0.6622 s. Toggling the auxiliary objective simultaneously changes cross-view contrastive, paired-similarity, and fusion-level losses. These results do not demonstrate a consistently positive effect or isolate the full final ensemble's RTCL contribution.

PredRet uses labelled target-system training samples, grouped outer folds, and an inner validation subset. Adapted MCRL has lower fold-mean MAE in six systems and higher MAE in four. Ten folds are not ten independent training repetitions; sample SD across folds is descriptive. Fold-mean and pooled metrics are separately named.

## Baseline provenance

| Baseline record | What is available | What remains unverified |
| --- | --- | --- |
| GCN, GNN-RT, 1D-CNN in SI Table S13 | Sample identifiers, SMILES, targets, predictions, and recalculated metrics | Full retraining equivalence is a separate question |
| Historical Transformer-LSTM 26.235159 s | Local archive has two columns, rt and prediction; original numeric metrics can be recalculated | Molecular mapping, original training project/checkpoint, strict fixed-split equivalence |
| Local ChemBERTa–BiLSTM 32.760509 s | B checkpoint configuration, identity-aligned SMRT predictions, and 100 historical target-fold load reports | Dynamic loading and new inference were not performed here; this is not the 26.24-s run |
| DeepGCN 26.622841 s | Local numeric archive and summary | Stored IDs do not bind to the fixed test RTs; the molecular mapping must be recovered before strict ranking |
| Graphormer 47.103665 s | Historical summary JSON | Per-sample predictions and full result identity |

The three identity-aligned public SI baseline records must not be described as independent verification of every model in the manuscript comparison table.

## Exact conventions

- Targets and MAE/RMSE/MedAE are in seconds; MAPE is percent; R² is 1−SSE/SST.
- Reported metrics use fixed assignment-table targets, after checking agreement with embedded saved labels to 0.001 s.
- Conventional median averages the middle two values for an even count. Legacy torch.median uses the lower middle value; not every MedAE discrepancy is simple decimal rounding.
- SD in the SI is sample SD (ddof=1).
- Structural overlap checks in the lightweight verifier use archived canonical-SMILES/Murcko keys, not a new RDKit canonicalization.
- SI composite predictions support final metric recalculation. They do not alone allow reconstruction of every memberwise PredRet validation prediction or repeat all original model selection.
- No statistical-significance claim is made from these archived comparisons.

## Public versus local availability

The original source snapshot contains no neural weights in Git history. The nine original SMRT CCI/PureGIN files are now available in the [checkpoint release](https://github.com/momomomo118/SMILES-Morgan-HSCL/releases/tag/mcrl-smrt-checkpoints-20261020), with server SHA-256 verification. The Hugging Face ChemBERTa resource has a separate verified public location and fixed revision. The release does not include every adapted PredRet or scaffold weight, and publishing checkpoint bytes does not establish successful new inference or full historical retraining equivalence.
