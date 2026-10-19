# PredRet condition-aware workflow

This workflow adds explicit chromatography-system inputs around the existing
SMILES/Morgan transfer model.

## 1. Build standardized condition metadata

```powershell
python code\prepare_predret_condition_metadata.py
```

Outputs:

- `data/external_predret/condition_metadata/system_conditions.csv`
- `data/external_predret/condition_metadata/row_conditions.csv`
- `data/external_predret/condition_metadata/condition_schema.json`

The current PredRet transfer CSV exposes reliable `system` labels but not full
LC method metadata. Missing method fields are therefore present as standard
`cond_*` columns with matching `mask_* = 0` columns. This lets the model learn
from available system-level inputs now and accept richer column/mobile-phase
metadata later without changing the training interface.

## 2. Build evaluation protocols

```powershell
python code\prepare_predret_condition_protocols.py
```

Outputs:

- `data/external_predret/condition_protocols/leave_one_system_out/<system>/`
- `data/external_predret/condition_protocols/same_system_scaffold/<system>/fold_00/`
- `data/external_predret/condition_protocols/few_shot_target/<system>/<shots>_shot/`
- `data/external_predret/condition_protocols/protocol_summary.csv`

These protocols separate random debugging from the main cross-system question:
leave-one-system-out, within-system scaffold generalization, and target-system
few-shot adaptation.

## 3. Run the minimum viable condition baselines

```powershell
python code\predret_hscl_bilstm_transfer.py `
  --data-root data/external_predret/transfer_10_systems_cleaned/by_system_10fold `
  --output-dir outputs/predret_condition_baselines `
  --variants molecule_only early_fusion early_fusion_mask `
  --systems LIFE_old LIFE_new Eawag_XBridgeC18 `
  --folds 0 1 2
```

The three default variants are:

- `molecule_only`: HSCL-initialized SMILES + Morgan model without conditions.
- `early_fusion`: adds a `condition_encoder` fed by standardized `cond_*` values.
- `early_fusion_mask`: additionally feeds `mask_*` missingness indicators.

Condition dropout is controlled by `--condition-dropout-prob` and defaults to
`0.25`. If all condition fields are dropped or absent, the model uses learned
null condition embeddings and falls back to molecule-only behavior.
