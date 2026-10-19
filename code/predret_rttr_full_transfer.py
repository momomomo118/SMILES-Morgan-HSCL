"""PredRet transfer using the original RT-TR RoBERTa-BiLSTM checkpoint.

This script mirrors the public RT-TR architecture more closely than
predret_hscl_bilstm_transfer.py:

- RobertaForRegression-style backbone.
- Hidden states from layers [-1, -3, -5] concatenated.
- LSTMClassificationHead with hidden_size=config.hidden_size.
- mean pooling over the LSTM sequence.
- tokenizer add_special_tokens=False and max_length=89 by default.

It loads the full `model.safetensors` checkpoint when available, including the
LSTM/head weights, then fine-tunes on each PredRet system/fold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, RobertaModel

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover - environment dependent
    load_safetensors = None


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


def resolve_layers(num_hidden_layers: int, layers: Iterable[int]) -> list[int]:
    out = []
    for layer in layers:
        if layer < 0:
            layer = num_hidden_layers + 1 + layer
        if layer < 0 or layer > num_hidden_layers:
            raise ValueError(f"Layer {layer} outside hidden-state range 0..{num_hidden_layers}")
        out.append(layer)
    return out


class RTTRHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lstm_hidden_size: int,
        dropout: float,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            dropout=dropout,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1),
        )

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(features)
        pooled = self.mean_pooling(sequence, attention_mask)
        return self.head(pooled).squeeze(-1)


class RTTRTransferModel(nn.Module):
    def __init__(self, checkpoint_dir: Path, layers_to_use: Iterable[int], dropout: float) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(checkpoint_dir)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.layers_to_use = resolve_layers(config.num_hidden_layers, layers_to_use)
        self.classifier = RTTRHead(
            input_dim=config.hidden_size * len(self.layers_to_use),
            lstm_hidden_size=config.hidden_size,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        features = torch.cat([outputs.hidden_states[i] for i in self.layers_to_use], dim=-1)
        return self.classifier(features, attention_mask)


class RTDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_length: int, scaler=None, add_special_tokens: bool = False) -> None:
        self.data = pd.read_csv(csv_path)
        required = {"sample_id", "smiles", "rt"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.y_original = self.data["rt"].astype(float).to_numpy(dtype=np.float32)
        if scaler is None:
            self.y = self.y_original
        else:
            self.y = scaler.transform(self.y_original.reshape(-1, 1)).reshape(-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        smiles = str(row["smiles"])
        encoded = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        return {
            "sample_id": row["sample_id"],
            "smiles": smiles,
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "labels": torch.tensor(float(self.y[idx]), dtype=torch.float32),
            "labels_original": torch.tensor(float(self.y_original[idx]), dtype=torch.float32),
        }


def build_scaler(train_csv: Path, mode: str):
    if mode == "none":
        return None
    train = pd.read_csv(train_csv)
    y = train["rt"].astype(float).to_numpy().reshape(-1, 1)
    scaler = StandardScaler() if mode == "standard" else PowerTransformer(method="box-cox", standardize=True)
    scaler.fit(y)
    return scaler


def inverse(values: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return values
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def move_batch(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_loader(csv_path: Path, tokenizer, args: argparse.Namespace, scaler, train: bool) -> DataLoader:
    ds = RTDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        scaler=scaler,
        add_special_tokens=args.add_special_tokens,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )


def load_rttr_checkpoint(model: nn.Module, checkpoint_dir: Path) -> dict[str, object]:
    path = checkpoint_dir / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    if load_safetensors is None:
        raise ImportError("safetensors is required to load model.safetensors")
    state = load_safetensors(str(path), device="cpu")
    report = model.load_state_dict(state, strict=False)
    return {
        "checkpoint": str(path),
        "loaded_keys": len(state),
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, scaler) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    ys, ps, sample_ids, smiles_values = [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        pred = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        ps.append(pred.detach().cpu().numpy())
        ys.append(batch["labels_original"].detach().cpu().numpy())
        sample_ids.extend(raw["sample_id"])
        smiles_values.extend(raw["smiles"])
    y = np.concatenate(ys)
    p = inverse(np.concatenate(ps), scaler)
    err = p - y
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "medae": float(np.median(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.clip(np.abs(y), 1e-12, None)) * 100.0),
        "r2": float(r2_score(y, p)) if len(y) > 1 else 0.0,
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0,
    }
    pred_df = pd.DataFrame({"sample_id": sample_ids, "smiles": smiles_values, "true_rt": y, "pred_rt": p})
    return metrics, pred_df


def train_one(system: str, fold: int, args: argparse.Namespace, tokenizer) -> dict[str, object]:
    set_seed(args.seed + fold)
    fold_dir = args.data_root / system / f"fold_{fold:02d}"
    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"
    scaler = build_scaler(train_csv, args.target_transform)

    model = RTTRTransferModel(args.checkpoint_dir, args.layers_to_use, args.dropout)
    load_report = load_rttr_checkpoint(model, args.checkpoint_dir)
    model.to(args.device)

    train_loader = make_loader(train_csv, tokenizer, args, scaler, train=True)
    val_loader = make_loader(val_csv, tokenizer, args, scaler, train=False)
    test_loader = make_loader(test_csv, tokenizer, args, scaler, train=False)

    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("roberta"):
            encoder_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_state = None
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw in tqdm(train_loader, desc=f"rttr_full {system} fold_{fold:02d} epoch {epoch}", leave=False):
            batch = move_batch(raw, args.device)
            pred = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.l1_loss(pred, batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, args.device, scaler)
        scheduler.step(val_metrics["mae"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"rttr_full {system} fold_{fold:02d} epoch={epoch} train={np.mean(losses):.4f} val_mae={val_metrics['mae']:.4f}", flush=True)
        if val_metrics["mae"] + 1e-9 < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop rttr_full {system} fold_{fold:02d} best_val={best_val:.4f}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(args.device)
    val_metrics, val_pred_df = evaluate(model, val_loader, args.device, scaler)
    test_metrics, pred_df = evaluate(model, test_loader, args.device, scaler)

    run_dir = args.output_dir / "rttr_full" / system / f"fold_{fold:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    val_pred_df.to_csv(run_dir / "predictions_val_rttr_full_transfer.csv", index=False)
    pred_df.to_csv(run_dir / "predictions_rttr_full_transfer.csv", index=False)
    payload = {
        "variant": "rttr_full",
        "system": system,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "validation": val_metrics,
        "test": test_metrics,
        "load_report": load_report,
        "args": {
            **vars(args),
            "checkpoint_dir": str(args.checkpoint_dir),
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
        },
    }
    (run_dir / "metrics_rttr_full_transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history_rttr_full_transfer.csv", index=False)
    return {"variant": "rttr_full", "system": system, "fold": fold, "best_epoch": best_epoch, "best_val_mae": best_val, **test_metrics}


def summarize(rows: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    detail = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "detail_predret_rttr_full_transfer.csv", index=False)
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
        )
        .reset_index()
    )
    summary["paper_mae"] = summary["system"].map(PAPER_MAE)
    summary["delta_mae_mean_minus_paper"] = summary["mae_mean"] - summary["paper_mae"]
    summary["improved_vs_paper"] = summary["mae_mean"] < summary["paper_mae"]
    summary.to_csv(output_dir / "summary_predret_rttr_full_transfer.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("code/smiles_baseline/RT-TR/models/best_checkpoint/checkpoint-380982"))
    parser.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_protocol_10fold/by_system_10fold"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predret_rttr_full_transfer_probe"))
    parser.add_argument("--systems", nargs="+", default=["LIFE_old", "LIFE_new", "Eawag_XBridgeC18"])
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("0,1,2"))
    parser.add_argument("--layers-to-use", type=parse_int_list, default=parse_int_list("-1,-3,-5"))
    parser.add_argument("--target-transform", choices=["none", "standard", "power"], default="power")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=89)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    rows = []
    for system in args.systems:
        for fold in args.folds:
            rows.append(train_one(system, fold, args, tokenizer))
            summary = summarize(rows, args.output_dir)
            print("\n===== CURRENT SUMMARY =====")
            cols = ["variant", "system", "n", "paper_mae", "mae_mean", "mae_std", "delta_mae_mean_minus_paper", "improved_vs_paper"]
            print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
