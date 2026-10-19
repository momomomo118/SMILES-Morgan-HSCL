"""PredRet transfer experiment with HSCL-initialized condition baselines.

This script is intentionally small-scope and fast to run before launching a
full 10-system/10-fold job. The default comparison now covers:

- molecule_only: HSCL-initialized SMILES + Morgan without chromatography input
- early_fusion: molecule features plus condition_encoder output
- early_fusion_mask: early_fusion with explicit metadata missingness masks

The older paper_roberta_bilstm, hscl_smiles_bilstm, and hscl_morgan_bilstm
variants remain available for backward-compatible checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from net.morgan_encoder import MorganEncoderConfig, MorganFingerprintEncoder  # noqa: E402
from net.smiles_encoder import SMILESEncoder  # noqa: E402
from utils.multimodal_dataset import MultimodalRTDataset  # noqa: E402


PAPER_MAE = {
    "Eawag_XBridgeC18": 28.594,
    "FEM_Lipids": 62.072,
    "FEM_long": 121.703,
    "FEM_short": 49.629,
    "IPB_Halle": 17.372,
    "LIFE_new": 10.913,
    "LIFE_old": 9.682,
    "UniToyama_Atlantis": 43.161,
    "Cao_HILIC": 38.121,
    "MTBLS87": 50.963,
}


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PredRetTransferModel(nn.Module):
    def __init__(
        self,
        smiles_model: str,
        smiles_layers: Iterable[int],
        use_morgan: bool,
        dropout: float,
        freeze_roberta: bool,
        freeze_morgan: bool,
        condition_dim: int = 0,
        condition_use_mask: bool = False,
        condition_hidden_dim: int = 128,
        condition_output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_morgan = use_morgan
        self.condition_dim = condition_dim
        self.smiles_encoder = SMILESEncoder(
            model_name_or_path=smiles_model,
            layers_to_use=smiles_layers,
            lstm_hidden_size=128,
            bidirectional=True,
            dropout=dropout,
            pooling="mean",
            freeze_roberta=freeze_roberta,
        )
        input_dim = self.smiles_encoder.output_dim
        if use_morgan:
            self.morgan_encoder = MorganFingerprintEncoder(
                MorganEncoderConfig(
                    n_bits=2048,
                    chunk_size=16,
                    hidden_dim=256,
                    num_layers=2,
                    num_heads=8,
                    feedforward_dim=512,
                    dropout=dropout,
                )
            )
            if freeze_morgan:
                for param in self.morgan_encoder.parameters():
                    param.requires_grad = False
            input_dim += self.morgan_encoder.output_dim
        else:
            self.morgan_encoder = None

        if condition_dim > 0:
            condition_input_dim = condition_dim * (2 if condition_use_mask else 1)
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_input_dim, condition_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(condition_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(condition_hidden_dim, condition_output_dim),
                nn.GELU(),
            )
            self.condition_null = nn.Parameter(torch.zeros(condition_dim))
            input_dim += condition_output_dim
        else:
            self.condition_encoder = None
            self.condition_null = None

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morgan_fp: torch.Tensor,
        condition_x: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smiles = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask).pooled
        if self.use_morgan:
            assert self.morgan_encoder is not None
            morgan = self.morgan_encoder(morgan_fp=morgan_fp).pooled
            features = torch.cat([smiles, morgan], dim=-1)
        else:
            features = smiles
        if self.condition_encoder is not None:
            if condition_x is None or condition_mask is None:
                batch_size = features.shape[0]
                device = features.device
                condition_x = torch.zeros(batch_size, self.condition_dim, device=device)
                condition_mask = torch.zeros(batch_size, self.condition_dim, device=device)
            assert self.condition_null is not None
            null = self.condition_null.to(condition_x.device).unsqueeze(0).expand_as(condition_x)
            condition_filled = condition_x * condition_mask + null * (1.0 - condition_mask)
            expected = self.condition_encoder[0].in_features
            if expected == self.condition_dim * 2:
                condition_input = torch.cat([condition_filled, condition_mask], dim=-1)
            else:
                condition_input = condition_filled
            condition_features = self.condition_encoder(condition_input)
            features = torch.cat([features, condition_features], dim=-1)
        return self.adapter(features).squeeze(-1)


def load_matching_prefix(target: nn.Module, checkpoint_path: Path, prefixes: tuple[str, ...]) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target_state = target.state_dict()
    matched = {}
    for key, value in source.items():
        for prefix in prefixes:
            source_prefix = prefix + "."
            if key.startswith(source_prefix):
                new_key = key[len(source_prefix) :]
                if new_key in target_state and target_state[new_key].shape == value.shape:
                    matched[new_key] = value
                break
    target_state.update(matched)
    target.load_state_dict(target_state)
    return {"matched": len(matched), "available": len(target_state)}


def build_target_scaler(train_csv: Path, mode: str):
    if mode == "none":
        return None
    train = pd.read_csv(train_csv)
    values = train["rt"].astype(float).to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def make_loader(
    csv_path: Path,
    tokenizer,
    args: argparse.Namespace,
    target_scaler,
    train: bool,
    seed: int,
) -> DataLoader:
    dataset = MultimodalRTDataset(
        csv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        smiles_augmentation=train and args.smiles_augment_prob > 0,
        smiles_augment_prob=args.smiles_augment_prob,
        smiles_augment_seed=seed,
        target_transformer=target_scaler,
        condition_metadata_path=args.condition_metadata if args.condition_metadata else None,
        condition_dropout_prob=args.condition_dropout_prob if train else 0.0,
        condition_dropout_seed=seed + 991,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


def model_flags_for_variant(variant: str) -> dict[str, object]:
    condition_variants = {
        "molecule_only": {"use_morgan": True, "condition": False, "mask": False},
        "early_fusion": {"use_morgan": True, "condition": True, "mask": False},
        "early_fusion_mask": {"use_morgan": True, "condition": True, "mask": True},
        "paper_roberta_bilstm": {"use_morgan": False, "condition": False, "mask": False},
        "hscl_smiles_bilstm": {"use_morgan": False, "condition": False, "mask": False},
        "hscl_morgan_bilstm": {"use_morgan": True, "condition": False, "mask": False},
    }
    if variant not in condition_variants:
        raise ValueError(f"Unknown variant={variant}. Available: {sorted(condition_variants)}")
    return condition_variants[variant]


def move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return values
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, scaler) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    y_true, y_pred, sample_ids, smiles_values = [], [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        pred = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
            condition_x=batch.get("condition_x"),
            condition_mask=batch.get("condition_mask"),
        )
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(batch["labels_original"].detach().cpu().numpy())
        sample_ids.extend(raw_batch["sample_id"])
        smiles_values.extend(raw_batch["smiles"])
    y = np.concatenate(y_true)
    p = inverse_target(np.concatenate(y_pred), scaler)
    error = p - y
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else 0.0,
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0,
    }
    pred_df = pd.DataFrame({"sample_id": sample_ids, "smiles": smiles_values, "true_rt": y, "pred_rt": p})
    return metrics, pred_df


def train_one(
    variant: str,
    system: str,
    fold: int,
    args: argparse.Namespace,
    tokenizer,
) -> dict[str, object]:
    set_seed(args.seed + fold)
    fold_dir = args.data_root / system / f"fold_{fold:02d}"
    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"
    scaler = build_target_scaler(train_csv, args.target_transform)

    flags = model_flags_for_variant(variant)
    use_morgan = bool(flags["use_morgan"])
    use_condition = bool(flags["condition"])
    condition_use_mask = bool(flags["mask"])
    model = PredRetTransferModel(
        smiles_model=args.smiles_model,
        smiles_layers=args.smiles_layers,
        use_morgan=use_morgan,
        dropout=args.dropout,
        freeze_roberta=args.freeze_roberta,
        freeze_morgan=args.freeze_morgan,
        condition_dim=args.condition_dim if use_condition else 0,
        condition_use_mask=condition_use_mask,
        condition_hidden_dim=args.condition_hidden_dim,
        condition_output_dim=args.condition_output_dim,
    )

    load_report = {}
    if variant.startswith("hscl_") or variant in {"molecule_only", "early_fusion", "early_fusion_mask"}:
        load_report["smiles"] = load_matching_prefix(model.smiles_encoder, args.checkpoint, ("smiles_encoder",))
        if use_morgan:
            load_report["morgan"] = load_matching_prefix(model.morgan_encoder, args.checkpoint, ("morgan_encoder",))

    model.to(args.device)
    train_loader = make_loader(train_csv, tokenizer, args, scaler, train=True, seed=args.seed + fold)
    val_loader = make_loader(val_csv, tokenizer, args, scaler, train=False, seed=args.seed + fold)
    test_loader = make_loader(test_csv, tokenizer, args, scaler, train=False, seed=args.seed + fold)

    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("smiles_encoder") or name.startswith("morgan_encoder"):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw_batch in tqdm(train_loader, desc=f"{variant} {system} fold_{fold:02d} epoch {epoch}", leave=False):
            batch = move_batch(raw_batch, args.device)
            pred = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                morgan_fp=batch["morgan_fp"],
                condition_x=batch.get("condition_x"),
                condition_mask=batch.get("condition_mask"),
            )
            loss = F.l1_loss(pred, batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, args.device, scaler)
        scheduler.step(val_metrics["mae"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"{variant} {system} fold_{fold:02d} epoch={epoch} "
            f"train={np.mean(losses):.4f} val_mae={val_metrics['mae']:.4f}",
            flush=True,
        )
        if val_metrics["mae"] + 1e-9 < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop {variant} {system} fold_{fold:02d} best_val={best_val:.4f}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(args.device)
    test_metrics, pred_df = evaluate(model, test_loader, args.device, scaler)

    run_dir = args.output_dir / variant / system / f"fold_{fold:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(run_dir / "predictions_predret_transfer.csv", index=False)
    payload = {
        "variant": variant,
        "system": system,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "test": test_metrics,
        "load_report": load_report,
        "condition": {
            "enabled": use_condition,
            "use_mask": condition_use_mask,
            "metadata": str(args.condition_metadata) if args.condition_metadata else None,
            "condition_dim": args.condition_dim if use_condition else 0,
            "condition_dropout_prob": args.condition_dropout_prob,
        },
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir), "checkpoint": str(args.checkpoint)},
    }
    (run_dir / "metrics_predret_transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history_predret_transfer.csv", index=False)
    return {"variant": variant, "system": system, "fold": fold, "best_epoch": best_epoch, "best_val_mae": best_val, **test_metrics}


def summarize(rows: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "detail_predret_hscl_bilstm_transfer.csv", index=False)
    summary = (
        detail.groupby(["variant", "system"])
        .agg(
            n=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            pearson_mean=("pearson", "mean"),
        )
        .reset_index()
    )
    summary["paper_mae"] = summary["system"].map(PAPER_MAE)
    summary["delta_mae_mean_minus_paper"] = summary["mae_mean"] - summary["paper_mae"]
    summary["improved_vs_paper"] = summary["mae_mean"] < summary["paper_mae"]
    summary.to_csv(output_dir / "summary_predret_hscl_bilstm_transfer.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_protocol_10fold/by_system_10fold"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predret_hscl_bilstm_transfer_probe"))
    parser.add_argument("--smiles-model", default="pretrained/ChemBERTa-zinc-base-v1")
    parser.add_argument("--smiles-layers", type=parse_int_list, default=parse_int_list("2,4,6"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/supcr_tuning_aliases/exact2025_best.pt"))
    parser.add_argument("--systems", nargs="+", default=["LIFE_old", "LIFE_new", "Eawag_XBridgeC18"])
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("0,1,2"))
    parser.add_argument("--variants", nargs="+", default=["molecule_only", "early_fusion", "early_fusion_mask"])
    parser.add_argument("--condition-metadata", type=Path, default=Path("data/external_predret/condition_metadata/system_conditions.csv"))
    parser.add_argument("--condition-dim", type=int, default=0)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--condition-output-dim", type=int, default=64)
    parser.add_argument("--condition-dropout-prob", type=float, default=0.25)
    parser.add_argument("--target-transform", choices=["none", "standard"], default="none")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smiles-augment-prob", type=float, default=0.6)
    parser.add_argument("--freeze-roberta", action="store_true")
    parser.add_argument("--freeze-morgan", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.smiles_model)
    uses_condition = any(v in {"early_fusion", "early_fusion_mask"} for v in args.variants)
    if args.condition_metadata and args.condition_metadata.exists():
        schema_df = pd.read_csv(args.condition_metadata, nrows=1)
        args.condition_dim = len([c for c in schema_df.columns if c.startswith("cond_")])
    elif uses_condition:
        raise FileNotFoundError(f"Condition metadata not found: {args.condition_metadata}")
    else:
        args.condition_metadata = None

    rows = []
    for variant in args.variants:
        for system in args.systems:
            for fold in args.folds:
                rows.append(train_one(variant, system, fold, args, tokenizer))
                summary = summarize(rows, args.output_dir)
                print("\n===== CURRENT SUMMARY =====")
                print(
                    summary[
                        [
                            "variant",
                            "system",
                            "n",
                            "paper_mae",
                            "mae_mean",
                            "mae_std",
                            "delta_mae_mean_minus_paper",
                            "improved_vs_paper",
                        ]
                    ].to_string(index=False)
                )


if __name__ == "__main__":
    main()
