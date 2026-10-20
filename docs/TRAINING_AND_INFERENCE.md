# Training, evaluation, and inference

The commands below cover archived-data verification, model resources, training, evaluation, and point inference. Use separate output folders for generated files.

## 1. Archived-data verification

~~~bash
python -B tools/verify_source_package.py
python -B supporting_information/verify_package.py
python -B tools/verify_saved_predictions.py --json-output ../verification/saved_predictions.json
python -B tools/prepare_reproduction_data.py --output-dir ../generated_inputs
~~~

These steps check hashes, schemas, stored structural-key separation, and saved metrics, then materialize the existing scaffold/PredRet assignments. They neither re-split data nor train a model. Full molecular canonicalization is not repeated.

Saved-prediction metrics use the fixed assignment-table RT labels, after checking agreement with embedded float32 prediction labels to 0.001 s. This reproduces the SI convention. Using embedded labels directly can change far-decimal metrics slightly; do not silently substitute labels or round predictions.

## 2. Environment

Create an isolated environment and install a suitable PyTorch build for the chosen hardware. requirements.txt records a Python 3.10–3.12 supplement dependency set; it does not lock every historical experiment.

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
~~~

Install the matching PyTorch build separately using its official installation instructions. Environment activation depends on the shell. Check the SI operation-specific records before choosing an environment:

| Operation | Recorded Python | PyTorch | RDKit | GPU |
| --- | --- | --- | --- | --- |
| Release verification | 3.9.25 | 2.3.0+cu118 | 2022.09.5 | RTX 4070 Laptop |
| CCI reevaluation | 3.9.25 | 2.3.0+cu118 | 2025.09.2 | RTX 4070 Laptop |
| Scaffold run | 3.12.3 | 2.7.1+cu128 | 2025.09.2 | RTX 4090 |

Choose the software environment according to the operation-specific records above and the hardware in use.

## 3. Resource preflight

Download the pinned ChemBERTa resource as documented in CHEMBERTA_PRETRAINED_MODEL.md. For the separate asset-root examples below, set the pinned download command's local_dir to ../model_assets/pretrained/ChemBERTa-zinc-base-v1. Download the active trained artifacts from the verified GitHub Release using `python -B tools/download_model_artifacts.py --model-root ../model_assets`. The separate artifact root avoids adding large generated files to the source snapshot.

~~~bash
python -B tools/check_model_artifacts.py --model-root ../model_assets
~~~

This command verifies the required model files against the recorded SHA-256 values.

## 4. CCI training and checkpoint evaluation

The archived configurations retain six candidates, seeds 1234/2025/777, augmentation probabilities, dropout, auxiliary-loss weights, learning rates, and target-transform settings. The exact777 member has zero final weight.

Generate commands from those recorded arguments without executing them:

~~~bash
python -B tools/print_archived_commands.py --model-root ../model_assets --mode train
python -B tools/print_archived_commands.py --model-root ../model_assets --mode train --member exact1234
python -B tools/print_archived_commands.py --model-root ../model_assets --mode eval --member exact1234 --split val
python -B tools/print_archived_commands.py --model-root ../model_assets --mode eval --member exact1234 --split test
~~~

The helper prints commands only. To actually train or evaluate, run a printed command separately. Evaluation mode uses --eval-only-checkpoint, retains the original training data for fitting the target transformation, and redirects --test-data to the chosen validation/test partition. It does not resume training.

Main training settings are batch 16/evaluation 32, maximum 200 epochs, patience 20, AdamW learning rates 1e-4/1e-5, weight decay 1e-3, train-fitted Yeo–Johnson targets, and 0.8 MAE + 0.2 Smooth L1 with beta 1. RTCL is the negative ranking ratio, not its negative-log variant. Member-specific differences are in the SI configuration CSV/JSON.

## 5. PureGIN training

Run separately for each seed after checking the recorded settings:

~~~bash
python -B code/train_pure_gnn_for_ensemble.py --train-data data/unified/train.csv --val-data data/unified/val.csv --test-data data/unified/test.csv --output-dir ../reproduction_runs/gin2026 --cache-dir ../reproduction_runs/graph_cache --seed 2026 --layers 5 --hidden 256 --dropout 0.15 --epochs 180 --patience 25 --batch-size 128 --lr 0.0008
~~~

Repeat with seeds 3407 and 777 and separate output directories. Cache inputs must match the selected split. The model uses explicit hydrogens, normalized weighted adjacency, mean/max pooling, and ten auxiliary features. It is trained independently of CCI and RTCL.

## 6. Complete fixed-weight MCRL point inference

The new standalone entry point loads five active CCI and three PureGIN models. It fits the Yeo–Johnson transformation on the fixed SMRT training RTs and inversely transforms each CCI output to seconds before weighting. It uses only molecular inputs for query molecules.

~~~bash
python -B tools/infer_mcrl_from_checkpoints.py --model-root ../model_assets --smiles CCO --device cpu --output ../inference/query.csv
python -B tools/infer_mcrl_from_checkpoints.py --model-root ../model_assets --input-csv data/unified/test.csv --device cuda --output ../inference/smrt_test_new_predictions.csv
~~~

The input file needs smiles and optionally sample_id; RT columns are not read for input molecules. The training RT file is required only to reconstruct the archived target transform. Existing output files are not overwritten. Original SMILES strings are retained after trimming; no new SMILES augmentation is performed.

The archived-prediction fusion program, scripts/run_final_fusion_ccrl_gnn.py, searches validation weights and writes result files. For query prediction with the published fixed weights, use the point-inference commands above.

## 7. PredRet target-system adaptation

Prepare the published assignments with the input helper. Use ../generated_inputs/predret/by_system_10fold as the data root. Use the published nested train, validation, and outer-test assignments.

The actual target variant comprises:
1. Four CCI initializations: exact1234, exact777, exact2025, and dropout015_seed777; all parameters are fine-tuned.
2. Three new PureGIN models per target fold, seeds 2026/3407/777.
3. A direct RT predictor averaging HistGradientBoosting and ExtraTrees on twelve descriptors.
4. Eight-candidate weights selected only on inner validation, with 8,000 Dirichlet candidates plus uniform/vertices.

For example, the following is a CCI training instruction for a single outer fold. Other members require their corresponding initialization and seed.

~~~bash
python -B code/src/train_multimodal.py --train-data ../generated_inputs/predret/by_system_10fold/Cao_HILIC/fold_00/train.csv --val-data ../generated_inputs/predret/by_system_10fold/Cao_HILIC/fold_00/val.csv --test-data ../generated_inputs/predret/by_system_10fold/Cao_HILIC/fold_00/test.csv --smiles-backbone custom --smiles-model ../model_assets/pretrained/ChemBERTa-zinc-base-v1 --resume-checkpoint ../model_assets/checkpoints/supcr_tuning_aliases/exact1234_best.pt --output-dir ../predret_runs/base/exact1234/Cao_HILIC/fold_00 --checkpoint-dir ../predret_runs/checkpoints/exact1234/Cao_HILIC/fold_00 --fusion cross_attention_residual --seed 1234 --epochs 40 --early-stopping-patience 6 --batch-size 8 --eval-batch-size 16 --lr 0.00005 --encoder-lr 0.000005 --dropout 0.15 --train-smiles-augmentation --smiles-augment-prob 0.2 --target-transform power --task-loss mae --temperature 0.1 --omega-align 0.01 --omega-fusion 0.005 --similarity-weight 0.1 --no-supcr-use-log
~~~

Export validation predictions from the selected checkpoint using --eval-only-checkpoint and --test-data pointing to that fold's val.csv; retain that fold's train.csv for fitting the target transform. The fit run supplies predictions_cross_attention_residual.csv; name the validation export predictions_val_cross_attention_residual.csv in the same member/system/fold output tree.

Train each PureGIN on the same inner train/validation/outer test inputs, with its output under ../predret_runs/gnn/seed2026/Cao_HILIC/fold_00 (and the other seeds). Its standard trainer exports both val and test predictions.

A local ChemBERTa–BiLSTM baseline is adapted from B_chemberta_bilstm_seed2026/best.pt, whose archived SMRT MAE is 32.7605 s:

~~~bash
python -B code/predret_transformer_lstm_strict.py --data-root ../generated_inputs/predret/by_system_10fold --pretrained-checkpoint ../model_assets/checkpoints/paper_ablation_20260825/B_chemberta_bilstm_seed2026/best.pt --smiles-model ../model_assets/pretrained/ChemBERTa-zinc-base-v1 --output-dir ../predret_runs/transformer_lstm --systems Cao_HILIC --folds 0
~~~

After all seven neural branches have matched validation/test exports, the existing code/predret_stack_full_ensemble_strict.py fits the descriptor predictors and selects the eight-candidate weights. Configure its --data-root, --base-root, --gnn-root, --output-dir, --checkpoint-names, --systems and --folds explicitly. Its final assembly for the example is:

~~~bash
python -B code/predret_stack_full_ensemble_strict.py --data-root ../generated_inputs/predret/by_system_10fold --base-root ../predret_runs/base --gnn-root ../predret_runs/gnn --output-dir ../predret_runs/proposed --checkpoint-names exact1234 exact777 exact2025 div_dropout015_seed777 --systems Cao_HILIC --folds 0 --blend-candidates 8000
~~~

The dropout member's output folder must use the same div_dropout015_seed777 name in --checkpoint-names and the branch export tree. These are fitting instructions, not actions run by the saved-record verifier. Training every target branch or regenerating predictions was not part of this preparation.
