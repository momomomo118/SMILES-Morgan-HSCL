"""Strict PredRet transfer for the Transformer-LSTM reproduction baseline.

This entry point is intentionally named after the Mazraedoost et al. 2025
Transformer-LSTM style baseline used in this project: ChemBERTa/RoBERTa hidden
state fusion followed by a bidirectional LSTM and dense RT regression head.
It is not Xue et al. RT-Transformer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


class TransformerLSTMModel(nn.Module):
    def __init__(
        self,
        smiles_model: str,
        layers_to_use: Iterable[int],
        lstm_hidden_size: int,
        pooling: str,
        hidden_dim: int,
        dropout: float,
        freeze_roberta: bool,
    ) -> None:
        super().__init__()
        self.encoder = SMILESEncoder(
            model_name_or_path=smiles_model,
            layers_to_use=layers_to_use,
            lstm_hidden_size=lstm_hidden_size,
            bidirectional=True,
            dropout=dropout,
            pooling=pooling,
            freeze_roberta=freeze_roberta,
        )
        self.regression_head = RegressionHead(self.encoder.output_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.regression_head(encoded.pooled)


def build_target_scaler(train_csv: Path, mode: str):
    if mode == "none":
        return None
    train = pd.read_csv(train_csv)
    y = train["rt"].astype(float).to_numpy().reshape(-1, 1)
    scaler = StandardScaler() if mode == "standard" else PowerTransformer(method="yeo-johnson", standardize=True)
    scaler.fit(y)
    return scaler


def inverse(values: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return values
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_loader(csv_path: Path, tokenizer, args: argparse.Namespace, target_scaler, train: bool, seed: int) -> DataLoader:
    dataset = MultimodalRTDataset(
        csv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        smiles_augmentation=train and args.smiles_augment_prob > 0,
        smiles_augment_prob=args.smiles_augment_prob,
        smiles_augment_seed=seed,
        target_transformer=target_scaler,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


def build_model(args: argparse.Namespace) -> TransformerLSTMModel:
    return TransformerLSTMModel(
        smiles_model=str(args.smiles_model),
        layers_to_use=args.layers_to_use,
        lstm_hidden_size=args.lstm_hidden_size,
        pooling=args.pooling,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        freeze_roberta=args.freeze_roberta,
    )


def load_pretrained(model: nn.Module, checkpoint: Path | None) -> dict[str, object]:
    if checkpoint is None:
        return {"checkpoint": None, "loaded": False, "reason": "no checkpoint requested"}
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    report = model.load_state_dict(state, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "loaded": True,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "loaded_keys": len(state),
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
        "checkpoint_args": payload.get("args") if isinstance(payload, dict) else None,
    }


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, scaler) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    y_true, y_pred, sample_ids, smiles_values = [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        pred = model(batch)
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(batch["labels_original"].detach().cpu().numpy())
        sample_ids.extend(raw["sample_id"])
        smiles_values.extend(raw["smiles"])
    y = np.concatenate(y_true)
    p = inverse(np.concatenate(y_pred), scaler)
    err = p - y
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "medae": float(np.median(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.clip(np.abs(y), 1e-12, None)) * 100.0),
        "r2": float(r2_score(y, p)) if len(y) > 1 else 0.0,
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0,
    }
    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "smiles": smiles_values,
            "true_rt": y,
            "pred_rt": p,
            "error": err,
            "abs_error": np.abs(err),
        }
    )
    return metrics, pred_df


def train_one(system: str, fold: int, args: argparse.Namespace, tokenizer) -> dict[str, object]:
    seed = args.seed + fold
    set_seed(seed)
    fold_name = f"fold_{fold:02d}"
    fold_dir = args.data_root / system / fold_name
    run_dir = args.output_dir / system / fold_name
    run_dir.mkdir(parents=True, exist_ok=True)

    done = run_dir / "metrics_transformer_lstm.json"
    if done.exists() and not args.force:
        payload = json.loads(done.read_text(encoding="utf-8"))
        return {
            "variant": "transformer_lstm",
            "system": system,
            "fold": fold,
            "best_epoch": payload["best_epoch"],
            "best_val_mae": payload["best_val_mae"],
            **payload["test"],
        }

    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"
    scaler = build_target_scaler(train_csv, args.target_transform)

    model = build_model(args)
    load_report = load_pretrained(model, args.pretrained_checkpoint)
    model.to(args.device)

    train_loader = make_loader(train_csv, tokenizer, args, scaler, train=True, seed=seed)
    val_loader = make_loader(val_csv, tokenizer, args, scaler, train=False, seed=seed + 1)
    test_loader = make_loader(test_csv, tokenizer, args, scaler, train=False, seed=seed + 2)

    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    best_state = None
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw in tqdm(train_loader, desc=f"transformer_lstm {system} {fold_name} epoch {epoch}", leave=False):
            batch = move_batch(raw, args.device)
            pred = model(batch)
            loss = F.l1_loss(pred, batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, args.device, scaler)
        scheduler.step(val_metrics["mae"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"transformer_lstm {system} {fold_name} epoch={epoch} train={row['train_loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f}",
            flush=True,
        )
        if val_metrics["mae"] + 1e-9 < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop transformer_lstm {system} {fold_name} best_val={best_val:.4f}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(args.device)
    val_metrics, val_pred = evaluate(model, val_loader, args.device, scaler)
    test_metrics, test_pred = evaluate(model, test_loader, args.device, scaler)

    val_pred.to_csv(run_dir / "predictions_val_transformer_lstm.csv", index=False)
    test_pred.to_csv(run_dir / "predictions_test_transformer_lstm.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history_transformer_lstm.csv", index=False)
    payload = {
        "variant": "transformer_lstm",
        "system": system,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "validation": val_metrics,
        "test": test_metrics,
        "load_report": load_report,
        "protocol": {
            "outer_test_used_for_scheduler": False,
            "outer_test_used_for_early_stopping": False,
            "outer_test_used_for_checkpoint_selection": False,
            "outer_test_evaluation": "once_after_loading_best_validation_checkpoint",
        },
        "args": {
            **vars(args),
            "smiles_model": str(args.smiles_model),
            "pretrained_checkpoint": str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
        },
    }
    done.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"variant": "transformer_lstm", "system": system, "fold": fold, "best_epoch": best_epoch, "best_val_mae": best_val, **test_metrics}


def summarize(rows: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    detail = pd.DataFrame(rows).sort_values(["system", "fold"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "transformer_lstm_detail.csv", index=False)
    summary = (
        detail.groupby(["variant", "system"])
        .agg(
            n=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            medae_mean=("medae", "mean"),
            medae_std=("medae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
        )
        .reset_index()
    )
    summary["paper_mae"] = summary["system"].map(PAPER_MAE)
    summary["delta_mae_mean_minus_paper"] = summary["mae_mean"] - summary["paper_mae"]
    summary.to_csv(output_dir / "transformer_lstm_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict PredRet transfer for Transformer-LSTM reproduction.")
    parser.add_argument("--smiles-model", type=Path, default=Path("pretrained/ChemBERTa-zinc-base-v1"))
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path("checkpoints/paper_ablation_20260825/B_chemberta_bilstm_seed2026/best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold/by_system_10fold"))
    parser.add_argument("--output-dir", type=Path, default=Path("server_results/predret_compare_strict_20260909/transformer_lstm"))
    parser.add_argument("--systems", nargs="+", default=list(PAPER_MAE))
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--layers-to-use", type=parse_int_list, default=parse_int_list("2,4,6"))
    parser.add_argument("--target-transform", choices=["none", "standard", "power"], default="power")
    parser.add_argument("--pooling", choices=["first_token", "last_token", "mean"], default="first_token")
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--smiles-augment-prob", type=float, default=0.0)
    parser.add_argument("--freeze-roberta", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.smiles_model)

    rows = []
    for system in args.systems:
        for fold in args.folds:
            rows.append(train_one(system, fold, args, tokenizer))
            summary = summarize(rows, args.output_dir)
            print("\n===== TRANSFORMER-LSTM CURRENT SUMMARY =====")
            cols = ["variant", "system", "n", "paper_mae", "mae_mean", "mae_std", "delta_mae_mean_minus_paper"]
            print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
