# Reproduction data (2026-10-19)

This version publishes the fixed experimental inputs and the machine-readable records accompanying the Supporting Information. RT values and absolute errors are in seconds. These files preserve the recorded partitions; preparing them does not train a model or generate new predictions.

## Included files

| Location | Contents |
| --- | --- |
| `data/unified/{train,val,test}.csv` | Original SMRT random split: 63,163 training, 7,019 validation, 7,798 test records; fields `sample_id,smiles,rt,split` |
| `data/unified/all.csv`, `summary.csv` | Combined SMRT input and split summary |
| `supporting_information/data/S1_smrt_scaffold_assignments.csv` | Fixed scaffold assignment for the same 77,980 molecules: 63,088 training, 7,035 validation, 7,857 test records |
| `supporting_information/data/S1_predret_cleaned.csv` | 2,197 cleaned observations from ten chromatographic systems, including stable IDs, SMILES and RT |
| `supporting_information/data/S1_predret_fold_assignments.csv` | Exact inner train, inner validation and outer test membership for all 100 system/fold combinations |
| `supporting_information/data/S2_*.csv` | Member configurations, ensemble metrics and saved SMRT fusion predictions |
| `supporting_information/data/S3_*.csv` | Retained baseline and ablation metrics with saved predictions |
| `supporting_information/data/S4_*.csv` | Scaffold results and PredRet fold/system results, fold coefficients and outer-test predictions |
| `supporting_information/configs/` | CCI/PureGIN arguments, adopted fusion coefficients, selection rules and evaluation protocols |
| `supporting_information/metadata/` | Field definitions, processing protocol, recorded environments, source hashes and validation records |
| `supporting_information/MCRL_Supporting_Information.pdf` | The 19-page supplement in black and white |

The full companion package has 25 CSV tables and 15 JSON files, including its manifest. The data dictionary documents column meanings and joins. `sample_id` prefixes retain the original random-split naming: always use the explicit `split` field for scaffold membership. PredRet fold numbers are zero-based. Fold-mean metrics and pooled metrics are separately named.

## Verify and prepare inputs

Run from the repository root using Python; no machine-learning packages are needed:

```bash
python -B tools/verify_source_package.py
python -B supporting_information/verify_package.py
python -B tools/prepare_reproduction_data.py
```

The last command reads the published assignments directly, without generating another random split or recomputing canonical SMILES. It creates these ignored local files:

```text
data/reproduction/smrt_scaffold/{train,val,test}.csv
data/reproduction/predret/all_cleaned.csv
data/reproduction/predret/by_system_10fold/<system>/fold_00/{train,val,test,outer_train,outer_test}.csv
...
data/reproduction/predret/by_system_10fold/<system>/fold_09/{train,val,test,outer_train,outer_test}.csv
```

Choose a different destination with `--output-dir PATH`. The helper checks the companion hashes and within-fold identity separation and refuses to replace different existing files.

The main random-split trainers already default to `data/unified/`. For scaffold training, set `--train-data`, `--val-data`, and `--test-data` to the corresponding generated scaffold files. For PredRet programs with `--data-root`, use `data/reproduction/predret/by_system_10fold` and the settings in `supporting_information/configs/predret_adaptation_protocol.json`. PredRet results describe target-system supervised adaptation with inner validation and held-out outer testing.

Pretrained ChemBERTa/tokenizer files and neural-network checkpoints remain separate model artifacts. The published CSV predictions support metric recomputation without neural-network inference; fresh training uses the recorded arguments and environments.

## Original data sources

- SMRT: [Figshare dataset, DOI 10.6084/m9.figshare.8038913](https://doi.org/10.6084/m9.figshare.8038913); [original article, DOI 10.1038/s41467-019-13680-7](https://doi.org/10.1038/s41467-019-13680-7).
- PredRet: [original resource](https://predret.org/); [original article, DOI 10.1021/acs.analchem.5b02287](https://doi.org/10.1021/acs.analchem.5b02287).

The processed inputs retain their original data attribution and terms. No new license is assigned to third-party data by this repository. PredRet companion records include molecular, chromatographic-system and RT information; contributor account fields are excluded.
