from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official ABCoRT checkpoint on a fixed SMRT CSV split."
    )
    parser.add_argument("--abcort-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_official_modules(repo: Path):
    repo = Path(repo).resolve()
    for filename in ("model.py", "feature_ops.py"):
        if not (repo / filename).is_file():
            raise FileNotFoundError(f"Missing official ABCoRT file: {repo / filename}")
    sys.path.insert(0, str(repo))
    from feature_ops import atom_featurizer, bond_featurizer
    from model import MyNet

    return MyNet, atom_featurizer, bond_featurizer


def molecule_to_data(smiles: str, rt: float, atom_featurizer, bond_featurizer) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    node_features = np.asarray(
        [atom_featurizer(atom) for atom in mol.GetAtoms()], dtype=np.float32
    )
    if node_features.ndim != 2 or node_features.shape[1] != 46:
        raise ValueError(
            f"Unexpected ABCoRT atom feature shape {node_features.shape} for {smiles}"
        )

    row: list[int] = []
    col: list[int] = []
    edge_features: list[np.ndarray] = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_featurizer(bond)
        row.extend((start, end))
        col.extend((end, start))
        edge_features.extend((feature, feature))

    edge_index = torch.tensor([row, col], dtype=torch.long)
    if edge_features:
        edge_attr = torch.tensor(np.asarray(edge_features), dtype=torch.float32)
    else:
        edge_attr = torch.empty((0, 21), dtype=torch.float32)
    if edge_attr.ndim != 2 or edge_attr.shape[1] != 21:
        raise ValueError(
            f"Unexpected ABCoRT bond feature shape {tuple(edge_attr.shape)} for {smiles}"
        )

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([rt], dtype=torch.float32),
    )


def read_fixed_test(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "smiles", "rt"}
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {path}")
    return rows


def normalize_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)!r}")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    state = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    MyNet, atom_featurizer, bond_featurizer = load_official_modules(args.abcort_repo)
    rows = read_fixed_test(args.test_csv)

    graphs: list[Data] = []
    for row in tqdm(rows, desc="ABCoRT featurization"):
        graphs.append(
            molecule_to_data(
                row["smiles"], float(row["rt"]), atom_featurizer, bond_featurizer
            )
        )
    if len(graphs) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} graphs, built {len(graphs)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyNet()
    try:
        raw_checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
    except TypeError:
        raw_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(normalize_state_dict(raw_checkpoint), strict=True)
    model.to(device).eval()

    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: list[float] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"ABCoRT inference on {device}"):
            batch = batch.to(device, non_blocking=True)
            pred = model(batch).reshape(-1)
            predictions.extend(pred.detach().cpu().double().tolist())

    if len(predictions) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} predictions, received {len(predictions)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "smiles", "true_rt", "pred_rt"]
        )
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "smiles": row["smiles"],
                    "true_rt": float(row["rt"]),
                    "pred_rt": prediction,
                }
            )

    print(f"device={device}")
    print(f"n={len(predictions)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
