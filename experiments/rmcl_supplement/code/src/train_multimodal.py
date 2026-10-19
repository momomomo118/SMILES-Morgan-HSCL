"""Train the SMILES-Morgan multimodal RT prediction model.

This script supports two stages:

1. `--smoke-test`: one forward pass to verify the data/model/loss pipeline.
2. default training: MAE task loss + cross-modal SupCR + optional fusion SupCR.

The defaults are tuned for a fair 20-epoch supervised contrastive comparison:
residual cross-attention fusion with light SupCR regularization. Strong
log-form SupCR should be treated as a sweep candidate rather than the default,
because it can overpower the RT regression objective on this dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm  # 鏂板锛氬鍏qdm杩涘害鏉?
from losses.supcr import CrossModalSupCRLoss, FusionSupCRLoss
from net.morgan_encoder import MorganEncoderConfig, MorganFingerprintEncoder
from net.multimodal_model import (
    SMILESMorganConcatModel,
    SMILESMorganCrossAttentionModel,
    SMILESMorganCrossAttentionResidualModel,
)
from net.smiles_encoder import SMILESEncoder
from utils.multimodal_dataset import MultimodalRTDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SMILES_BACKBONE_PRESETS = {
    "chemberta_zinc": PROJECT_ROOT
    / "smiles_baseline"
    / "RT-TR-main"
    / "RT-TR-main"
    / "seyonec"
    / "ChemBERTa-zinc-base-v1",
    "chemberta_10m_mtr": PROJECT_ROOT
    / "smiles_baseline"
    / "RT-TR-main"
    / "RT-TR-main"
    / "models"
    / "ChemBERTa-10M-MTR_pretrained",
    "chemberta_77m_mtr": PROJECT_ROOT
    / "smiles_baseline"
    / "RT-TR-main"
    / "RT-TR-main"
    / "models"
    / "ChemBERTa-77M-MTR_pretrained",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMILES-Morgan HSCL RT predictor.")

    parser.add_argument("--train-data", default=str(PROJECT_ROOT / "data" / "unified" / "train.csv"))
    parser.add_argument("--val-data", default=str(PROJECT_ROOT / "data" / "unified" / "val.csv"))
    parser.add_argument("--test-data", default=str(PROJECT_ROOT / "data" / "unified" / "test.csv"))
    parser.add_argument(
        "--smiles-backbone",
        choices=[*SMILES_BACKBONE_PRESETS.keys(), "custom"],
        default="chemberta_zinc",
        help="Convenience preset for the SMILES encoder backbone.",
    )
    parser.add_argument(
        "--smiles-model",
        default=None,
        help="Custom Hugging Face/local RoBERTa path. Used when --smiles-backbone custom, or to override a preset.",
    )
    parser.add_argument(
        "--smiles-layers",
        default="2,4,6",
        help="Comma-separated hidden-state layers for SMILES layer fusion, e.g. 2,4,6 or -1,-3,-5.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "outputs" / "multimodal"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "src" / "checkpoints" / "multimodal"))
    parser.add_argument(
        "--prediction-output",
        default=None,
        help="CSV path for per-sample test predictions. Defaults to output-dir/predictions_<fusion>.csv.",
    )
    parser.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Load a checkpoint, evaluate the test set, export metrics and predictions, then exit.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Load model weights from a checkpoint, then continue training on the provided train/val/test data.",
    )

    parser.add_argument(
        "--fusion",
        choices=["concat", "cross_attention", "cross_attention_residual"],
        default="cross_attention_residual",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--morgan-mode", choices=["normal", "shuffled", "zero"], default="normal")
    parser.add_argument("--train-smiles-augmentation", action="store_true")
    parser.add_argument("--smiles-augment-prob", type=float, default=0.0)
    parser.add_argument(
        "--target-transform",
        choices=["none", "power", "standard"],
        default="none",
        help="Train the regression head on transformed RT and report metrics after inverse transform.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-smiles", action="store_true")
    parser.add_argument("--freeze-morgan", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--similarity-weight", type=float, default=0.1)
    parser.add_argument("--omega-align", type=float, default=0.01)
    parser.add_argument("--omega-fusion", type=float, default=0.005)
    parser.add_argument(
        "--task-loss",
        choices=["mae", "mae_huber", "weighted_mae_huber"],
        default="mae",
        help="Regression loss for RT prediction.",
    )
    parser.add_argument("--huber-weight", type=float, default=0.3)
    parser.add_argument("--huber-beta", type=float, default=30.0)
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.8,
        help="Training RT quantile used as the high-RT threshold for --task-loss weighted_mae_huber.",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=2.0,
        help="Loss multiplier for samples above --tail-quantile when using --task-loss weighted_mae_huber.",
    )
    parser.add_argument(
        "--supcr-use-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the stable -log SupCR objective. Disable with --no-supcr-use-log for the exact ratio form.",
    )
    parser.add_argument("--max-pairs-per-anchor", type=int, default=None)
    parser.add_argument(
        "--supcr-positive-mode",
        choices=["rank", "rt_bucket"],
        default="rank",
        help="Use original ranking SupCR or RT-threshold positives.",
    )
    parser.add_argument("--supcr-strong-threshold", type=float, default=30.0)
    parser.add_argument("--supcr-weak-threshold", type=float, default=60.0)
    parser.add_argument("--supcr-weak-weight", type=float, default=0.5)

    parser.add_argument("--morgan-hidden-dim", type=int, default=256)
    parser.add_argument("--morgan-layers", type=int, default=2)
    parser.add_argument("--morgan-heads", type=int, default=8)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument(
        "--tta-smiles",
        type=int,
        default=1,
        help="Number of SMILES views averaged at test time. 1 disables test-time augmentation.",
    )
    parser.add_argument(
        "--tta-seed",
        type=int,
        default=2026,
        help="Seed used when generating randomized SMILES for test-time augmentation.",
    )
    parser.add_argument(
        "--no-tta-canonical",
        action="store_true",
        help="Use only randomized SMILES views during TTA instead of including the canonical input as one view.",
    )

    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/loss pass and exit.")
    return parser.parse_args()


def resolve_smiles_model(args: argparse.Namespace) -> str:
    if args.smiles_model:
        return str(args.smiles_model)
    if args.smiles_backbone == "custom":
        raise ValueError("--smiles-model is required when --smiles-backbone custom")
    return str(SMILES_BACKBONE_PRESETS[args.smiles_backbone])


def parse_smiles_layers(value: str) -> Tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("--smiles-layers must contain at least one layer index")
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict, device: str) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_target_transformer(args: argparse.Namespace):
    if args.target_transform == "none":
        return None
    train_rt = pd.read_csv(args.train_data)["rt"].astype(float).to_numpy().reshape(-1, 1)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True) if args.target_transform == "power" else StandardScaler()
    transformer.fit(train_rt)
    return transformer


def resolve_tail_rt_threshold(args: argparse.Namespace) -> float:
    quantile = min(max(float(args.tail_quantile), 0.0), 1.0)
    train_rt = pd.read_csv(args.train_data)["rt"].astype(float)
    return float(train_rt.quantile(quantile))


def inverse_target_tensor(values: torch.Tensor, target_transformer) -> torch.Tensor:
    if target_transformer is None:
        return values.detach().cpu()
    array = values.detach().cpu().numpy().reshape(-1, 1)
    inverted = target_transformer.inverse_transform(array).reshape(-1)
    return torch.tensor(inverted, dtype=torch.float32)


def build_dataloaders(args: argparse.Namespace, tokenizer, target_transformer=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MultimodalRTDataset(
        args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        morgan_mode=args.morgan_mode,
        morgan_shuffle_seed=args.seed,
        smiles_augmentation=args.train_smiles_augmentation,
        smiles_augment_prob=args.smiles_augment_prob,
        smiles_augment_seed=args.seed,
        target_transformer=target_transformer,
    )
    val_dataset = MultimodalRTDataset(
        args.val_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        morgan_mode=args.morgan_mode,
        morgan_shuffle_seed=args.seed + 1,
        target_transformer=target_transformer,
    )
    test_dataset = MultimodalRTDataset(
        args.test_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        morgan_mode=args.morgan_mode,
        morgan_shuffle_seed=args.seed + 2,
        target_transformer=target_transformer,
    )

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


def build_model(args: argparse.Namespace):
    smiles_encoder = SMILESEncoder(
        model_name_or_path=args.smiles_model,
        layers_to_use=parse_smiles_layers(args.smiles_layers),
        lstm_hidden_size=128,
        pooling="first_token",
        dropout=args.dropout,
        freeze_roberta=args.freeze_smiles,
    )
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

    if args.freeze_morgan:
        for param in morgan_encoder.parameters():
            param.requires_grad = False

    if args.fusion == "concat":
        model = SMILESMorganConcatModel(
            smiles_encoder=smiles_encoder,
            morgan_encoder=morgan_encoder,
            dropout=args.dropout,
        )
    elif args.fusion == "cross_attention":
        model = SMILESMorganCrossAttentionModel(
            smiles_encoder=smiles_encoder,
            morgan_encoder=morgan_encoder,
            fusion_dim=args.fusion_dim,
            fusion_heads=args.fusion_heads,
            fusion_feedforward_dim=args.fusion_dim * 2,
            dropout=args.dropout,
        )
    elif args.fusion == "cross_attention_residual":
        model = SMILESMorganCrossAttentionResidualModel(
            smiles_encoder=smiles_encoder,
            morgan_encoder=morgan_encoder,
            fusion_dim=args.fusion_dim,
            fusion_heads=args.fusion_heads,
            fusion_feedforward_dim=args.fusion_dim * 2,
            dropout=args.dropout,
        )
    return model


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("smiles_encoder") or name.startswith("morgan_encoder"):
            encoder_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def compute_losses(
    output,
    labels: torch.Tensor,
    align_loss_fn: CrossModalSupCRLoss,
    fusion_loss_fn: FusionSupCRLoss,
    args: argparse.Namespace,
    supcr_labels: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    supcr_labels = labels if supcr_labels is None else supcr_labels
    task_loss = regression_loss(output.prediction, labels, args, original_labels=supcr_labels)

    alignment = align_loss_fn(output.z_smiles, output.z_morgan, supcr_labels)
    fusion_loss = labels.new_tensor(0.0)
    if output.z_fusion is not None:
        fusion_loss = fusion_loss_fn(output.z_fusion, supcr_labels)

    total_loss = task_loss + args.omega_align * alignment.total + args.omega_fusion * fusion_loss
    return {
        "total": total_loss,
        "task": task_loss,
        "align_total": alignment.total,
        "align_supcr": alignment.supcr,
        "align_similarity": alignment.similarity,
        "fusion_supcr": fusion_loss,
    }


def regression_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    original_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    abs_error = (prediction - labels).abs()
    mae_loss = abs_error.mean()
    if args.task_loss == "mae":
        return mae_loss
    huber_values = F.smooth_l1_loss(prediction, labels, beta=args.huber_beta, reduction="none")
    if args.task_loss == "mae_huber":
        huber_loss = huber_values.mean()
        return (1.0 - args.huber_weight) * mae_loss + args.huber_weight * huber_loss
    if args.task_loss == "weighted_mae_huber":
        if original_labels is None:
            weights = torch.ones_like(labels)
        else:
            threshold = labels.new_tensor(float(args.tail_rt_threshold))
            weights = torch.where(original_labels >= threshold, args.tail_weight, 1.0)
        weighted_mae = (weights * abs_error).sum() / weights.sum().clamp_min(1e-12)
        weighted_huber = (weights * huber_values).sum() / weights.sum().clamp_min(1e-12)
        return (1.0 - args.huber_weight) * weighted_mae + args.huber_weight * weighted_huber
    raise ValueError(f"Unsupported task loss: {args.task_loss}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    align_loss_fn: CrossModalSupCRLoss,
    fusion_loss_fn: FusionSupCRLoss,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_keys = ["total", "task", "align_total", "align_supcr", "align_similarity", "fusion_supcr"]
    totals = {key: 0.0 for key in loss_keys}
    num_batches = 0

    # 鏂板锛氱敤tqdm鍖呰loader锛屾坊鍔犺繘搴︽潯
    for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} - Training", leave=False):
        batch = move_batch(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        losses = compute_losses(
            output,
            batch["labels"],
            align_loss_fn,
            fusion_loss_fn,
            args,
            supcr_labels=batch.get("labels_original"),
        )
        losses["total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    align_loss_fn: CrossModalSupCRLoss,
    fusion_loss_fn: FusionSupCRLoss,
    args: argparse.Namespace,
    desc: str = "Evaluating",
    tokenizer=None,
    use_tta: bool = False,
    target_transformer=None,
) -> Dict[str, float]:
    model.eval()
    loss_keys = ["total", "task", "align_total", "align_supcr", "align_similarity", "fusion_supcr"]
    totals = {key: 0.0 for key in loss_keys}
    num_batches = 0
    y_true = []
    y_pred = []

    # 鏂板锛氱敤tqdm鍖呰loader锛屾坊鍔犺繘搴︽潯
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch(batch, args.device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        losses = compute_losses(
            output,
            batch["labels"],
            align_loss_fn,
            fusion_loss_fn,
            args,
            supcr_labels=batch.get("labels_original"),
        )
        prediction = output.prediction
        if use_tta:
            prediction = predict_with_tta(model, batch, tokenizer, args)
            task_loss = regression_loss(prediction, batch["labels"], args, original_labels=batch.get("labels_original"))
            losses["task"] = task_loss
            losses["total"] = task_loss + args.omega_align * losses["align_total"] + args.omega_fusion * losses["fusion_supcr"]
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        num_batches += 1
        y_true.append(batch.get("labels_original", batch["labels"]).detach().cpu())
        y_pred.append(inverse_target_tensor(prediction, target_transformer))

    metrics = {key: value / max(num_batches, 1) for key, value in totals.items()}
    metrics.update(regression_metrics(torch.cat(y_true), torch.cat(y_pred)))
    return metrics


def as_plain_list(value) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def randomize_smiles(smiles: str, rng: random.Random) -> str:
    try:
        from rdkit import Chem
    except ImportError:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def build_smiles_views(smiles_list: List[str], args: argparse.Namespace) -> List[List[str]]:
    num_views = max(int(args.tta_smiles), 1)
    include_canonical = not args.no_tta_canonical
    rng = random.Random(args.tta_seed)
    views = []
    for smiles in smiles_list:
        sample_views = [smiles] if include_canonical else []
        while len(sample_views) < num_views:
            sample_views.append(randomize_smiles(smiles, rng))
        views.append(sample_views[:num_views])
    return views


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    batch: Dict,
    tokenizer,
    args: argparse.Namespace,
) -> torch.Tensor:
    smiles_list = [str(smiles) for smiles in as_plain_list(batch.get("smiles", ""))]
    views = build_smiles_views(smiles_list, args)
    flat_smiles = [smiles for sample_views in views for smiles in sample_views]
    encoded = tokenizer(
        flat_smiles,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    repeat_count = len(views[0]) if views else 1
    morgan_fp = batch["morgan_fp"].repeat_interleave(repeat_count, dim=0)
    output = model(input_ids=input_ids, attention_mask=attention_mask, morgan_fp=morgan_fp)
    return output.prediction.view(len(smiles_list), repeat_count).mean(dim=1)


@torch.no_grad()
def save_test_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    output_path: str | Path,
    tokenizer=None,
    use_tta: bool = False,
    target_transformer=None,
) -> None:
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch in tqdm(loader, desc="Saving test predictions", leave=False):
        batch = move_batch(batch, args.device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        sample_ids = as_plain_list(batch.get("sample_id", ""))
        smiles_list = as_plain_list(batch.get("smiles", ""))
        true_values = batch.get("labels_original", batch["labels"]).detach().cpu().tolist()
        prediction = predict_with_tta(model, batch, tokenizer, args) if use_tta else output.prediction
        pred_values = inverse_target_tensor(prediction, target_transformer).tolist()

        for sample_id, smiles, true_rt, pred_rt in zip(sample_ids, smiles_list, true_values, pred_values):
            error = float(pred_rt) - float(true_rt)
            rows.append(
                {
                    "sample_id": sample_id,
                    "smiles": smiles,
                    "true_rt": float(true_rt),
                    "pred_rt": float(pred_rt),
                    "error": error,
                    "abs_error": abs(error),
                }
            )

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "smiles", "true_rt", "pred_rt", "error", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    diff = y_pred - y_true
    abs_diff = diff.abs()
    mae = abs_diff.mean().item()
    rmse = torch.sqrt((diff**2).mean()).item()
    medae = abs_diff.median().item()
    mape = (abs_diff / y_true.abs().clamp_min(1e-12)).mean().mul(100.0).item()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    pearson = pearson_corr(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
        "r2": r2,
        "pearson": pearson,
        "within_30s": (abs_diff <= 30).float().mean().item(),
        "within_60s": (abs_diff <= 60).float().mean().item(),
        "within_120s": (abs_diff <= 120).float().mean().item(),
    }


def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = torch.sqrt((yt**2).sum() * (yp**2).sum()).clamp_min(1e-12)
    return ((yt * yp).sum() / denom).item()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
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
    keys = ["total", "task", "mae", "rmse", "r2", "pearson", "align_supcr", "fusion_supcr"]
    parts = [f"{key}={metrics[key]:.6g}" for key in keys if key in metrics]
    return f"{prefix}: " + " | ".join(parts)


def build_loss_fns(args: argparse.Namespace) -> Tuple[CrossModalSupCRLoss, FusionSupCRLoss]:
    align_loss_fn = CrossModalSupCRLoss(
        temperature=args.temperature,
        similarity_weight=args.similarity_weight,
        use_log=args.supcr_use_log,
        max_pairs_per_anchor=args.max_pairs_per_anchor,
        positive_mode=args.supcr_positive_mode,
        strong_positive_threshold=args.supcr_strong_threshold,
        weak_positive_threshold=args.supcr_weak_threshold,
        weak_positive_weight=args.supcr_weak_weight,
    )
    fusion_loss_fn = FusionSupCRLoss(
        temperature=args.temperature,
        use_log=args.supcr_use_log,
        max_pairs_per_anchor=args.max_pairs_per_anchor,
        positive_mode=args.supcr_positive_mode,
        strong_positive_threshold=args.supcr_strong_threshold,
        weak_positive_threshold=args.supcr_weak_threshold,
        weak_positive_weight=args.supcr_weak_weight,
    )
    return align_loss_fn, fusion_loss_fn


def run_smoke_test(args: argparse.Namespace, model: torch.nn.Module, train_loader: DataLoader) -> None:
    align_loss_fn, fusion_loss_fn = build_loss_fns(args)
    batch = move_batch(next(iter(train_loader)), args.device)
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        losses = compute_losses(
            output,
            batch["labels"],
            align_loss_fn,
            fusion_loss_fn,
            args,
            supcr_labels=batch.get("labels_original"),
        )

    print("Forward pass OK")
    print("fusion mode:", args.fusion)
    print("prediction shape:", tuple(output.prediction.shape))
    print("smiles pooled shape:", tuple(output.smiles_pooled.shape))
    print("morgan pooled shape:", tuple(output.morgan_pooled.shape))
    if output.fusion_pooled is not None:
        print("fusion pooled shape:", tuple(output.fusion_pooled.shape))
    print("z_smiles shape:", tuple(output.z_smiles.shape))
    print("z_morgan shape:", tuple(output.z_morgan.shape))
    if output.z_fusion is not None:
        print("z_fusion shape:", tuple(output.z_fusion.shape))
    for key, value in losses.items():
        print(f"{key} loss:", float(value))


def main() -> None:
    args = parse_args()
    args.smiles_model = resolve_smiles_model(args)
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.smiles_model)
    target_transformer = build_target_transformer(args)
    args.tail_rt_threshold = resolve_tail_rt_threshold(args)
    train_loader, val_loader, test_loader = build_dataloaders(args, tokenizer, target_transformer)
    model = build_model(args).to(args.device)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded resume checkpoint weights: {args.resume_checkpoint}")
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    align_loss_fn, fusion_loss_fn = build_loss_fns(args)

    if args.smoke_test:
        run_smoke_test(args, model, train_loader)
        return

    if args.eval_only_checkpoint:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        use_tta = args.tta_smiles > 1
        test_metrics = evaluate(
            model,
            test_loader,
            align_loss_fn,
            fusion_loss_fn,
            args,
            desc="Test",
            tokenizer=tokenizer,
            use_tta=use_tta,
            target_transformer=target_transformer,
        )
        print(log_metrics("test", test_metrics))

        suffix = f"{args.fusion}_tta{args.tta_smiles}" if use_tta else args.fusion
        result_path = Path(args.output_dir) / f"test_metrics_{suffix}.json"
        result_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote test metrics: {result_path}")

        prediction_path = (
            Path(args.prediction_output)
            if args.prediction_output
            else Path(args.output_dir) / f"predictions_{suffix}.csv"
        )
        save_test_predictions(
            model,
            test_loader,
            args,
            prediction_path,
            tokenizer=tokenizer,
            use_tta=use_tta,
            target_transformer=target_transformer,
        )
        print(f"Wrote test predictions: {prediction_path}")
        return

    history = []
    best_val_mae = math.inf
    epochs_without_improvement = 0
    best_path = Path(args.checkpoint_dir) / f"best_{args.fusion}.pt"

    print(f"Training fusion={args.fusion} morgan_mode={args.morgan_mode} on device={args.device}")
    print(f"SMILES backbone={args.smiles_backbone} layers={args.smiles_layers} path={args.smiles_model}")
    print(f"train batches={len(train_loader)}, val batches={len(val_loader)}, test batches={len(test_loader)}")
    print(
        "Training augmentation/loss: "
        f"train_smiles_augmentation={args.train_smiles_augmentation}, "
        f"smiles_augment_prob={args.smiles_augment_prob}, "
        f"task_loss={args.task_loss}, huber_weight={args.huber_weight}, huber_beta={args.huber_beta}"
    )
    if args.task_loss == "weighted_mae_huber":
        print(
            "Tail loss config: "
            f"tail_quantile={args.tail_quantile}, tail_rt_threshold={args.tail_rt_threshold:.6g}, "
            f"tail_weight={args.tail_weight}"
        )
    print(f"Target transform: {args.target_transform}")
    print(
        "SupCR config: "
        f"use_log={args.supcr_use_log}, temperature={args.temperature}, "
        f"omega_align={args.omega_align}, omega_fusion={args.omega_fusion}, "
        f"similarity_weight={args.similarity_weight}, dropout={args.dropout}, "
        f"positive_mode={args.supcr_positive_mode}, "
        f"strong_threshold={args.supcr_strong_threshold}, "
        f"weak_threshold={args.supcr_weak_threshold}, weak_weight={args.supcr_weak_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        # 淇敼锛氫紶鍏poch鍙傛暟缁檛rain_one_epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, align_loss_fn, fusion_loss_fn, args, epoch)
        val_metrics = evaluate(
            model,
            val_loader,
            align_loss_fn,
            fusion_loss_fn,
            args,
            desc=f"Epoch {epoch}/{args.epochs} - Validation",
            target_transformer=target_transformer,
        )
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
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, args)
            print(f"  saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1

        history_path = Path(args.output_dir) / f"history_{args.fusion}.json"
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"validation MAE improvement. Best val MAE={best_val_mae:.6g}."
            )
            break

    checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    use_tta = args.tta_smiles > 1
    if use_tta:
        print(f"Final test uses SMILES TTA with {args.tta_smiles} views.")
    test_metrics = evaluate(
        model,
        test_loader,
        align_loss_fn,
        fusion_loss_fn,
        args,
        desc="Final Test",
        tokenizer=tokenizer,
        use_tta=use_tta,
        target_transformer=target_transformer,
    )
    print(log_metrics("test", test_metrics))

    suffix = f"{args.fusion}_tta{args.tta_smiles}" if use_tta else args.fusion
    result_path = Path(args.output_dir) / f"test_metrics_{suffix}.json"
    result_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote test metrics: {result_path}")

    prediction_path = (
        Path(args.prediction_output)
        if args.prediction_output
        else Path(args.output_dir) / f"predictions_{suffix}.csv"
    )
    save_test_predictions(
        model,
        test_loader,
        args,
        prediction_path,
        tokenizer=tokenizer,
        use_tta=use_tta,
        target_transformer=target_transformer,
    )
    print(f"Wrote test predictions: {prediction_path}")


if __name__ == "__main__":
    main()
