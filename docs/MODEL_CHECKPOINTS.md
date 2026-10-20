# Trained checkpoint inventory

The nine original archived SMRT checkpoints are publicly distributed as [GitHub Release assets](https://github.com/momomomo118/SMILES-Morgan-HSCL/releases/tag/mcrl-smrt-checkpoints-20261020). Their sizes and SHA-256 values were verified against both local source files and GitHub server digests. They are separate downloads because of their size; checkpoint bytes were not changed or reserialized.

Complete MCRL is not a single checkpoint. Fixed SMRT inference uses five active CCI predictors and three PureGIN predictors. The sixth CCI candidate exact777 has zero weight and is retained for provenance; it is not needed for the final scalar prediction.

| Family / member | Seed | Inner weight | Outer weight | Required for fixed final inference | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| CCI / exact1234 | 1234 | 0.3 | 0.3 | yes | 582426311 | 22b03ab0db278757c46fde87bcff32e40c72cb9b66bc419c9ab43a506c41e485 |
| CCI / exact2025 | 2025 | 0.15 | 0.3 | yes | 582426311 | 4d8329d9edf462f79bf0730da4e949c962b1e45f60bb77de5267b941ab50f68b |
| CCI / exact777 | 777 | 0.0 | 0.3 | no (zero-weight candidate) | 582426311 | 60acb4d89c6a5488a687464cb0296354cd00bd3c6a80e4000adf77473c1c4da2 |
| CCI / dropout015_seed777 | 777 | 0.25 | 0.3 | yes | 582426311 | 15a219712af05f54610a4e96d64fff6ed4256b02e96503a4432885b54c79f579 |
| CCI / lowfusion_seed777 | 777 | 0.1 | 0.3 | yes | 582426311 | 29b2c3f5763475e2064bff45326db8cf04bd44cd0bb55ee4434410b8d4dca632 |
| CCI / aug050_seed777 | 777 | 0.2 | 0.3 | yes | 582426311 | 92efe5d8f6d28ae8aa6d5f38cafa2943ce0e1ade9cd318f084fa11b611ce935a |
| PureGIN / gin2026 | 2026 | 1 | 0.24 | yes | 3356165 | 65e03cac36cb3d0b4c3c1ff1955e2f0400b0c1bd88c621578d43752afc7f7d9b |
| PureGIN / gin3407 | 3407 | 1 | 0.18 | yes | 3356165 | 295e16f87d1a57ba1a71031191e52808f9780707d96d02daaf38319eb4c58f0b |
| PureGIN / gin777 | 777 | 1 | 0.28 | yes | 3356165 | da9855b4be566606f6188de0220217ef11d9b2a2788ee8b27261aa430a6ecdf5 |

## Expected relative paths

- exact1234: checkpoints/supcr_tuning_aliases/exact1234_best.pt
- exact2025: checkpoints/supcr_tuning_aliases/exact2025_best.pt
- exact777: checkpoints/supcr_tuning_aliases/exact777_best.pt
- dropout015_seed777: checkpoints/supcr_tuning/target_power_div_dropout015_seed777/best_cross_attention_residual.pt
- lowfusion_seed777: checkpoints/supcr_tuning/target_power_div_lowfusion_seed777/best_cross_attention_residual.pt
- aug050_seed777: checkpoints/supcr_tuning/target_power_div_aug050_seed777/best_cross_attention_residual.pt
- gin2026: checkpoints/gnn/pure_gin_seed2026/best_pure_gin.pt
- gin3407: checkpoints/gnn/pure_gin_seed3407/best_pure_gin.pt
- gin777: checkpoints/gnn/pure_gin_seed777/best_pure_gin.pt

## Other inputs and models

- Pinned ChemBERTa config/tokenizer/weights are separate backbone artifacts; see CHEMBERTA_PRETRAINED_MODEL.md.
- Fixed SMRT training RTs reconstruct the train-fitted Yeo–Johnson transformation. The six historical CCI checkpoints do not embed that transformer.
- The dropout member also has a historical alias under checkpoints/supcr_tuning_aliases/. The canonical nonalias path in this inventory is used for final inference.
- PredRet uses fitted target-system/fold-specific models. The six original SMRT CCI and three original SMRT GIN artifacts are not the fitted models for all 100 target folds.

## Access status

Download the active files without importing ML libraries:

~~~bash
python -B tools/download_model_artifacts.py --model-root ../model_assets
~~~

Add `--include-zero-weight` to download the optional exact777 provenance candidate. The downloader uses the actual release URLs and verifies every SHA-256; it refuses to overwrite a differing existing checkpoint. The complete manifest and SHA256SUMS.txt are also release assets. Obtain the ChemBERTa backbone separately and place all inputs at the paths documented in TRAINING_AND_INFERENCE.md. 

## File checks

Use python -B tools/check_model_artifacts.py after placing files at the recorded paths. This checks SHA-256 without executing pickle/model code. A PASS confirms agreement with the published file identities.
