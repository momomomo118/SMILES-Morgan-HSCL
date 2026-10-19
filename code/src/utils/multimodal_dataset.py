"""Shared dataset for SMILES-Morgan multimodal RT prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


@dataclass
class MorganConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


class MultimodalRTDataset(Dataset):
    """Return synchronized SMILES tokens, Morgan fingerprints, and RT labels.

    The input CSV must contain:
    - sample_id
    - smiles
    - rt
    - split

    Morgan fingerprints are generated from the same SMILES used by the SMILES
    branch, ensuring both modalities always refer to the same molecule.
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Optional[Any] = None,
        max_length: int = 256,
        morgan: MorganConfig | None = None,
        add_special_tokens: bool = True,
        morgan_mode: str = "normal",
        morgan_shuffle_seed: int = 1234,
        smiles_augmentation: bool = False,
        smiles_augment_prob: float = 0.0,
        smiles_augment_seed: int = 1234,
        target_transformer: Optional[Any] = None,
        condition_metadata_path: str | Path | None = None,
        condition_dropout_prob: float = 0.0,
        condition_dropout_seed: int = 1234,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.morgan = morgan or MorganConfig()
        self.add_special_tokens = add_special_tokens
        self.morgan_mode = morgan_mode
        self.smiles_augmentation = smiles_augmentation
        self.smiles_augment_prob = smiles_augment_prob
        self._smiles_rng = random.Random(smiles_augment_seed)
        self.condition_dropout_prob = condition_dropout_prob
        self._condition_rng = random.Random(condition_dropout_seed)
        self.target_values_original = self.data["rt"].astype(float).to_numpy(dtype=np.float32)
        if target_transformer is None:
            self.target_values = self.target_values_original
        else:
            transformed = target_transformer.transform(self.target_values_original.reshape(-1, 1)).reshape(-1)
            self.target_values = transformed.astype(np.float32)

        if self.morgan_mode not in {"normal", "shuffled", "zero"}:
            raise ValueError("morgan_mode must be one of: normal, shuffled, zero")
        if not 0.0 <= self.smiles_augment_prob <= 1.0:
            raise ValueError("smiles_augment_prob must be between 0 and 1")

        required = {"sample_id", "smiles", "rt", "split"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

        if self.morgan_mode == "shuffled":
            rng = np.random.default_rng(morgan_shuffle_seed)
            self._morgan_indices = rng.permutation(len(self.data))
        else:
            self._morgan_indices = None

        self.condition_value_columns: list[str] = []
        self.condition_mask_columns: list[str] = []
        self.condition_values = np.zeros((len(self.data), 0), dtype=np.float32)
        self.condition_masks = np.zeros((len(self.data), 0), dtype=np.float32)
        if condition_metadata_path is not None:
            self._attach_condition_metadata(Path(condition_metadata_path))

    @property
    def condition_dim(self) -> int:
        return int(self.condition_values.shape[1])

    def _attach_condition_metadata(self, metadata_path: Path) -> None:
        meta = pd.read_csv(metadata_path)
        if "system" not in self.data.columns:
            raise ValueError(f"{self.csv_path} needs a system column for condition metadata")
        if "system" not in meta.columns:
            raise ValueError(f"{metadata_path} needs a system column")

        value_cols = sorted([c for c in meta.columns if c.startswith("cond_")])
        mask_cols = sorted([c for c in meta.columns if c.startswith("mask_")])
        if not value_cols:
            raise ValueError(f"{metadata_path} has no cond_* columns")
        expected_masks = ["mask_" + c[len("cond_") :] for c in value_cols]
        missing_masks = [c for c in expected_masks if c not in mask_cols]
        if missing_masks:
            raise ValueError(f"{metadata_path} is missing condition masks: {missing_masks[:8]}")

        merged = self.data[["system"]].merge(
            meta[["system", *value_cols, *expected_masks]],
            on="system",
            how="left",
            validate="many_to_one",
        )
        if merged[value_cols].isna().any().any():
            missing = sorted(self.data.loc[merged[value_cols].isna().any(axis=1), "system"].astype(str).unique())
            raise ValueError(f"Missing condition metadata for systems: {missing}")

        self.condition_value_columns = value_cols
        self.condition_mask_columns = expected_masks
        self.condition_values = merged[value_cols].astype(float).to_numpy(dtype=np.float32)
        self.condition_masks = merged[expected_masks].astype(float).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def _encode_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            return {}

        encoded = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
        }

    def _randomize_smiles(self, smiles: str) -> str:
        if not self.smiles_augmentation or self._smiles_rng.random() > self.smiles_augment_prob:
            return smiles

        try:
            from rdkit import Chem
        except ImportError:
            return smiles

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True)

    def _morgan_fingerprint(self, smiles: str) -> torch.Tensor:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for Morgan fingerprints. "
                "Run this dataset in the same environment as the Morgan baseline."
            ) from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.morgan.n_bits, dtype=torch.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.morgan.radius,
            nBits=self.morgan.n_bits,
            useChirality=self.morgan.use_chirality,
        )
        return torch.tensor(list(fp), dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data.iloc[index]
        smiles = str(row["smiles"])
        encoder_smiles = self._randomize_smiles(smiles)
        if self.morgan_mode == "zero":
            morgan_fp = torch.zeros(self.morgan.n_bits, dtype=torch.float32)
        elif self.morgan_mode == "shuffled":
            morgan_row = self.data.iloc[int(self._morgan_indices[index])]
            morgan_fp = self._morgan_fingerprint(str(morgan_row["smiles"]))
        else:
            # Keep Morgan on canonical SMILES while the SMILES branch may see
            # randomized enumerations during training.
            morgan_fp = self._morgan_fingerprint(smiles)

        item: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "smiles": smiles,
            "encoder_smiles": encoder_smiles,
            "split": row["split"],
            "labels": torch.tensor(float(self.target_values[index]), dtype=torch.float32),
            "labels_original": torch.tensor(float(self.target_values_original[index]), dtype=torch.float32),
            "morgan_fp": morgan_fp,
        }
        if self.condition_dim:
            condition_x = self.condition_values[index].copy()
            condition_mask = self.condition_masks[index].copy()
            if self.condition_dropout_prob > 0 and self._condition_rng.random() < self.condition_dropout_prob:
                condition_x[:] = 0.0
                condition_mask[:] = 0.0
            item["condition_x"] = torch.tensor(condition_x, dtype=torch.float32)
            item["condition_mask"] = torch.tensor(condition_mask, dtype=torch.float32)
        item.update(self._encode_smiles(encoder_smiles))
        return item
