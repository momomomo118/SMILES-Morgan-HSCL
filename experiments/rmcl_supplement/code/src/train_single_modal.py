"""Train single-modality RT prediction baselines.

This file is intentionally separate from train_multimodal.py so the existing
multimodal experiment remains untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from net.morgan_encoder import MorganEncoderConfig, MorganFingerprintEncoder
from net.smiles_encoder import SMILESEncoder
from utils.multimodal_dataset import MultimodalRTDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class SMILESOnlyModel(nn.Module):
    def __init__(self, smiles_encoder: SMILESEncoder, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.regression_head = RegressionHead(smiles_encoder.output_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.smiles_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.regression_head(encoded.pooled)


class MorganOnlyModel(nn.Module):
    def __init__(
        self,
        morgan_encoder: MorganFingerprintEncoder,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.morgan_encoder = morgan_encoder
        self.regression_head = RegressionHead(morgan_encoder.output_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.morgan_encoder(morgan_fp=batch["morgan_fp"])
        return self.regression_head(encoded.pooled)


def parse_args(default_mode: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SMILES-only or Morgan-only RT baseline.")
    parser.add_argument("--mode", choices=["smiles", "morgan"], default=default_mode or "smiles")
    parser.add_argument("--train-data", default=str(PROJECT_ROOT / "data" / "unified" / "train.csv"))
    parser.add_argument("--val-data", default=str(PROJECT_ROOT / "data" / "unified" / "val.csv"))
    parser.add_argument("--test-data", default=str(PROJECT_ROOT / "data" / "unified" / "test.csv"))
    parser.add_argument(
        "--smiles-model",
        default=str(
            PROJECT_ROOT
            / "smiles_baseline"
            / "RT-TR-main"
            / "RT-TR-main"
            / "seyonec"
            / "ChemBERTa-zinc-base-v1"
        ),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "outputs" / "single_modal"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "src" / "checkpoints" / "single_modal"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-smiles", action="store_true")
    parser.add_argument("--morgan-hidden-dim", type=int, default=256)
    parser.add_argument("--morgan-layers", type=int, default=2)
    parser.add_argument("--morgan-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device: str) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(args.smiles_model) if args.mode == "smiles" else None
    train_dataset = MultimodalRTDataset(args.train_data, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = MultimodalRTDataset(args.val_data, tokenizer=tokenizer, max_length=args.max_length)
    test_dataset = MultimodalRTDataset(args.test_data, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.mode == "smiles":
        smiles_encoder = SMILESEncoder(
            model_name_or_path=args.smiles_model,
            layers_to_use=(2, 4, 6),
            lstm_hidden_size=128,
            pooling="first_token",
            dropout=args.dropout,
            freeze_roberta=args.freeze_smiles,
        )
        return SMILESOnlyModel(smiles_encoder=smiles_encoder, dropout=args.dropout)

    morgan_encoder = MorganFingerprintEncoder(
        MorganEncoderConfig(
            n_bits=2048,
            chunk_size=16,
            hidden_dim=args.morgan_hidden_dim,
            num_layers=args.morgan_layers,
            num_heads=args.morgan_heads,
            feedforward_dim=args.morgan_hidden_dim * 2,
            dropout=args.dropout,
        )
    )
    return MorganOnlyModel(morgan_encoder=morgan_encoder, dropout=args.dropout)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    diff = y_pred - y_true
    abs_diff = diff.abs()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(1e-12)
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    pearson = ((yt * yp).sum() / torch.sqrt((yt**2).sum() * (yp**2).sum()).clamp_min(1e-12)).item()
    return {
        "mae": abs_diff.mean().item(),
        "rmse": torch.sqrt((diff**2).mean()).item(),
        "medae": abs_diff.median().item(),
        "r2": (1.0 - ss_res / ss_tot).item(),
        "pearson": pearson,
        "within_30s": (abs_diff <= 30).float().mean().item(),
        "within_60s": (abs_diff <= 60).float().mean().item(),
        "within_120s": (abs_diff <= 120).float().mean().item(),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"{args.mode} train", leave=False):
        batch = move_batch(batch, args.device)
        labels = batch["labels"]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = F.l1_loss(prediction, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return {"mae_loss": total_loss / max(len(loader), 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args: argparse.Namespace, desc: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch(batch, args.device)
        labels = batch["labels"]
        prediction = model(batch)
        loss = F.l1_loss(prediction, labels)
        total_loss += float(loss.detach().cpu())
        y_true.append(labels.detach().cpu())
        y_pred.append(prediction.detach().cpu())

    metrics = {"mae_loss": total_loss / max(len(loader), 1)}
    metrics.update(regression_metrics(torch.cat(y_true), torch.cat(y_pred)))
    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def log_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    keys = ["mae_loss", "mae", "rmse", "r2", "pearson", "within_120s"]
    return f"{prefix}: " + " | ".join(f"{key}={metrics[key]:.6g}" for key in keys if key in metrics)


def run(default_mode: str | None = None) -> None:
    args = parse_args(default_mode=default_mode)
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(args)
    model = build_model(args).to(args.device)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if args.smoke_test:
        batch = move_batch(next(iter(train_loader)), args.device)
        model.eval()
        with torch.no_grad():
            prediction = model(batch)
        print("Forward pass OK")
        print("mode:", args.mode)
        print("prediction shape:", tuple(prediction.shape))
        return

    history = []
    best_val_mae = math.inf
    best_path = Path(args.checkpoint_dir) / f"best_{args.mode}_only.pt"
    print(f"Training mode={args.mode} on device={args.device}")
    print(f"train batches={len(train_loader)}, val batches={len(val_loader)}, test batches={len(test_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args)
        val_metrics = evaluate(model, val_loader, args, desc=f"{args.mode} val")
        scheduler.step(val_metrics["mae"])

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}")
        print(log_metrics("  train", train_metrics))
        print(log_metrics("  val", val_metrics))

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, args)
            print(f"  saved best checkpoint: {best_path}")

        history_path = Path(args.output_dir) / f"history_{args.mode}_only.json"
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, args, desc=f"{args.mode} test")
    print(log_metrics("test", test_metrics))

    result_path = Path(args.output_dir) / f"test_metrics_{args.mode}_only.json"
    result_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote test metrics: {result_path}")


if __name__ == "__main__":
    run()
