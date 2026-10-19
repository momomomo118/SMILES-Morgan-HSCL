# Model Mechanism

```mermaid
flowchart TD
    A["Input molecule: SMILES"] --> B1["SMILES branch: ChemBERTa / Transformer encoding"]
    A --> B2["Morgan fingerprint branch"]
    A --> B3["RDKit descriptor branch: physicochemical refinement candidate"]
    A --> B4["Molecular graph branch: atoms + bonds + adjacency"]

    B1 --> C1["CCRL SMILES-Morgan multimodal model pool"]
    B2 --> C1
    C1 --> D1["Top6 CCRL ensemble: pred_ensemble"]

    B3 --> E1["Residual correction / RDKit-GBDT refinement candidates"]
    D1 --> E1
    E1 --> E2["Auxiliary calibration and ablation module"]

    B4 --> G1["PureGIN seed2026"]
    B4 --> G2["PureGIN seed3407"]
    B4 --> G3["PureGIN seed777"]

    D1 --> H["Validation-optimized PTDF decision fusion"]
    E2 -. "candidate, not mandatory" .-> H
    G1 --> H
    G2 --> H
    G3 --> H

    H --> I["Predicted retention time"]
    I --> J["Final Scheme A test MAE = 25.8893 s"]
```

## Core Idea

The final system uses complementary molecular representations:

- SMILES semantic representation from ChemBERTa-style sequence encoding.
- Morgan fingerprint representation for local circular substructures.
- Molecular graph representation for atom-bond topology.
- RDKit physicochemical descriptors as optional residual refinement candidates.

The CCRL SMILES-Morgan models remain the neural backbone. The Top6 CCRL ensemble provides the final base branch through `pred_ensemble`. Three PureGIN graph models provide complementary topological predictors.

## Final PTDF Definition

PTDF is now defined as validation-optimized decision fusion:

```text
CCRL ensemble
+ physicochemical refinement candidates
+ topological GNN predictors
-> validation optimized decision fusion
-> RT prediction
```

The final deployed Scheme A uses:

```text
y_RT =
0.30 * CCRL pred_ensemble
+ 0.24 * PureGIN seed2026
+ 0.18 * PureGIN seed3407
+ 0.28 * PureGIN seed777
```

Residual correction is retained as a physicochemical residual refinement module and ablation analysis, not as the final mandatory base.
