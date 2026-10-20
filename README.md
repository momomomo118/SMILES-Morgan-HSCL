# MCRL retention-time prediction

Prepared on 2026-10-20 from public commit [b5caa73bace8f280d12e79ca062301c613922855](https://github.com/momomomo118/SMILES-Morgan-HSCL/tree/b5caa73bace8f280d12e79ca062301c613922855). This update adds reproducibility documentation, verification tools, and links to the separately published trained checkpoint release. Archived scientific data, prediction values, existing model source files, and the SI PDF are preserved.

MCRL denotes the complete SMRT framework: a SMILES/Morgan CCI ensemble and three independent PureGIN predictors combined at prediction level. The final archived predictor is:

~~~text
RT = 0.30 * CCI_ensemble
   + 0.24 * PureGIN_seed2026
   + 0.18 * PureGIN_seed3407
   + 0.28 * PureGIN_seed777.
~~~

CCI performs feature-level interaction between ChemBERTa–BiLSTM SMILES features and Morgan fingerprint features. RTCL provides continuous-RT auxiliary supervision during CCI training. PureGIN consumes explicit-hydrogen graphs and ten descriptor/string features. Physicochemical residual correction is an auxiliary analysis and does not replace the CCI ensemble in the final formula.

## Model names and experiment boundaries

| Label | Meaning in these records | Result scope |
| --- | --- | --- |
| MCRL / Scheme A | Complete CCI-ensemble plus three-PureGIN decision fusion | Archived SMRT MAE 25.8893 s, RMSE 52.2262 s, R² 0.9124 |
| CCRL / HSCL / SupCR in historical paths | Historical names used for SMILES/Morgan predictors, auxiliary losses, or their ensemble; inspect the configuration | A matching filename alone does not identify the complete MCRL model |
| CCI | SMILES/Morgan feature-interaction predictor | Six ensemble candidates; five have nonzero weights |
| RTCL | Cross-view ranking ratio, paired-projection similarity, and fusion-level ranking ratio objectives | Three-seed MAE changes are small and seed-dependent |
| PTDF | Final scalar prediction fusion | Separate from feature-level fusion inside CCI or PureGIN |
| RMCL / CMA / RSCL | Independently archived single-predictor supplement experiments | No Top6/PureGIN/PTDF training; the archived seed1234 predictor MAE 31.0575 s is not final MCRL performance |

PredRet uses a target-adapted variant: four fine-tuned CCI predictors, three target-trained PureGIN predictors, and a twelve-descriptor predictor. Eight-candidate coefficients are selected on each inner validation set. It is supervised adaptation, not zero-shot cross-system prediction.

## Start with reproducible archived results

From a clean repository copy, these commands require only Python, except optional PDF page counting with pypdf:

~~~bash
python -B tools/verify_source_package.py
python -B supporting_information/verify_package.py
python -B tools/verify_saved_predictions.py
~~~

The restored SI uses a single package root containing its PDF, README, verifier, manifest, data/, configs/, and metadata/. The accompanying paper-folder copy had separated the PDF from this root and omitted its README.

For fixed training inputs without re-splitting:

~~~bash
python -B tools/prepare_reproduction_data.py --output-dir ../generated_inputs
~~~

Run source-manifest checks before creating other files inside the snapshot. Generated outputs should use a separate destination.

## Model resources and execution

- [Pinned ChemBERTa model and file identities](docs/CHEMBERTA_PRETRAINED_MODEL.md)
- [CCI/PureGIN checkpoint inventory and access limits](docs/MODEL_CHECKPOINTS.md)
- [Training, evaluation, and inference commands](docs/TRAINING_AND_INFERENCE.md)
- [Four levels of reproducibility and result limitations](docs/RESULT_REPRODUCIBILITY.md)
- [Original published data instructions](docs/REPRODUCTION_DATA.md)
- [B01–B14 preparation provenance](docs/B_PREPARATION_PROVENANCE.json)

The original trained SMRT CCI/PureGIN files are available in the [checkpoint release](https://github.com/momomomo118/SMILES-Morgan-HSCL/releases/tag/mcrl-smrt-checkpoints-20261020). All nine assets have verified server SHA-256 digests. Five active CCI and three PureGIN members are required for fixed MCRL inference; the zero-weight CCI candidate is optional. Download them using `python -B tools/download_model_artifacts.py --model-root ../model_assets`, then follow the resource and inference instructions. These SMRT artifacts do not replace target-adapted PredRet fold checkpoints.

The new standalone inference entry point applies the train-fitted inverse target transformation before CCI ensembling. It was syntax-checked and its help interface was tested, but full neural loading and inference were not executed during this preparation. Do not treat its presence as an inference-reproduction result. The existing calibration exporter still has the known inverse-transform gap documented in the execution guide.

## Data sources and environments

SMRT: [Figshare DOI 10.6084/m9.figshare.8038913](https://doi.org/10.6084/m9.figshare.8038913). PredRet: [official resource](https://predret.org/). Fixed processed data are archived here; raw-to-processed SMRT provenance remains incomplete.

Environments differ by operation. SI metadata retain release verification, checkpoint reevaluation, and scaffold-run records separately. requirements.txt is a recorded supplement dependency set, not a universal lock for all historical runs.

Original attribution and third-party terms remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This preparation does not assign new rights to original data or model artifacts.
