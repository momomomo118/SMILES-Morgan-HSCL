# Unified Multimodal Data

This folder is the single data entry point for the SMILES-Morgan HSCL model.

## Files

- `train.csv`: shared training split.
- `val.csv`: shared validation split.
- `test.csv`: shared test split.
- `all.csv`: concatenation of all splits.
- `summary.csv`: split-level sample counts and RT ranges.

## Columns

| column | meaning |
| --- | --- |
| `sample_id` | unique id within the unified project |
| `smiles` | molecular SMILES string |
| `rt` | experimental retention time in seconds |
| `split` | `train`, `val`, or `test` |

## Usage

Both branches must read these same files:

- SMILES branch: tokenize the `smiles` column.
- Morgan branch: generate Morgan fingerprints from the same `smiles` column.
- Prediction label: use the same `rt` column.

Do not let the Morgan branch run a separate random split. The multimodal model
requires both branches to see the same molecule and label at each index.

## Published version

These CSV inputs are included in the repository. The fixed split has 63,163 training, 7,019 validation and 7,798 test records (77,980 total). Scaffold and PredRet assignments, source citations and export commands are documented in [REPRODUCTION_DATA.md](../../docs/REPRODUCTION_DATA.md).
