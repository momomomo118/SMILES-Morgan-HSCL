from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel


ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "code" / "src"
sys.path.insert(0, str(CODE_SRC))

from losses.supcr import CrossModalSupCRLoss, FusionSupCRLoss  # noqa: E402
from net.bimodal_fusion import BimodalCrossAttentionFusion  # noqa: E402
from net.morgan_encoder import MorganEncoderConfig, MorganFingerprintEncoder, MorganProjectionHead  # noqa: E402
from net.projection_head import ProjectionHead  # noqa: E402
from net.smiles_encoder import EncoderOutput, SMILESEncoder, SMILESProjectionHead  # noqa: E402
from utils.multimodal_dataset import MultimodalRTDataset  # noqa: E402


OUT_ROOT = ROOT / "outputs" / "paper_ablation_20260825"
CKPT_ROOT = ROOT / "checkpoints" / "paper_ablation_20260825"
PRETRAINED = ROOT / "pretrained" / "ChemBERTa-zinc-base-v1"
SPLIT_FILES = {
    "train": ROOT / "data" / "unified" / "train.csv",
    "val": ROOT / "data" / "unified" / "val.csv",
    "test": ROOT / "data" / "unified" / "test.csv",
}
SCAFFOLD_DIR = OUT_ROOT / "04_scaffold_generalization"
SCAFFOLD_SPLIT_FILES = {
    "train": SCAFFOLD_DIR / "scaffold_train.csv",
    "val": SCAFFOLD_DIR / "scaffold_val.csv",
    "test": SCAFFOLD_DIR / "scaffold_test.csv",
}
SHOW_PROGRESS = os.environ.get("PAPER_ABLATION_PROGRESS", "").lower() in {"1", "true", "yes"}


def progress(iterable, **kwargs):
    return tqdm(iterable, disable=not SHOW_PROGRESS, **kwargs)


@dataclass
class RunConfig:
    run_id: str
    configuration: str
    seed: int
    chemberta: bool = True
    bilstm: bool = True
    morgan: bool = True
    fusion: str = "cross_attention_residual"
    supcr: bool = False
    target_transform: str = "power"
    train_data: str = str(SPLIT_FILES["train"])
    val_data: str = str(SPLIT_FILES["val"])
    test_data: str = str(SPLIT_FILES["test"])
    epochs: int = 200
    early_stopping_patience: int = 20
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-3
    dropout: float = 0.1
    grad_clip: float = 1.0
    max_length: int = 256
    smiles_layers: str = "2,4,6"
    morgan_hidden_dim: int = 256
    morgan_layers: int = 2
    morgan_heads: int = 8
    fusion_dim: int = 256
    fusion_heads: int = 8
    task_loss: str = "mae_huber"
    huber_weight: float = 0.2
    huber_beta: float = 1.0
    temperature: float = 0.07
    similarity_weight: float = 0.05
    omega_align: float = 0.01
    omega_fusion: float = 0.005
    smiles_augment_prob: float = 0.35
    train_smiles_augmentation: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ChemBERTaNoLSTMEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, layers_to_use: Iterable[int], dropout: float = 0.1) -> None:
        super().__init__()
        try:
            self.roberta = RobertaModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'get'" not in str(exc):
                raise
            self.roberta = RobertaModel.from_pretrained(
                model_name_or_path,
                add_pooling_layer=False,
                use_safetensors=False,
            )
        self.layers_to_use = tuple(layers_to_use)
        self.output_dim = self.roberta.config.hidden_size * len(self.layers_to_use)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> EncoderOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        fused = torch.cat([outputs.hidden_states[i] for i in self.layers_to_use], dim=-1)
        fused = self.dropout(fused)
        return EncoderOutput(sequence=fused, pooled=fused[:, 0, :], mask=attention_mask)


class SingleBranchRTModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.regression_head = RegressionHead(encoder.output_dim, dropout=dropout)

    def forward(self, input_ids=None, attention_mask=None, morgan_fp=None):
        if morgan_fp is not None and input_ids is None:
            out = self.encoder(morgan_fp=morgan_fp)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "prediction": self.regression_head(out.pooled),
            "smiles_pooled": out.pooled,
            "morgan_pooled": out.pooled,
            "fusion_pooled": None,
            "z_smiles": out.pooled,
            "z_morgan": out.pooled,
            "z_fusion": None,
        }


class MultiModalRTModel(nn.Module):
    def __init__(self, smiles_encoder, morgan_encoder, fusion: str, dropout: float, fusion_dim: int, fusion_heads: int):
        super().__init__()
        self.smiles_encoder = smiles_encoder
        self.morgan_encoder = morgan_encoder
        self.smiles_projection = SMILESProjectionHead(smiles_encoder.output_dim, dropout=dropout)
        self.morgan_projection = MorganProjectionHead(morgan_encoder.output_dim, dropout=dropout)
        self.fusion_kind = fusion
        if fusion == "concat":
            self.fusion = None
            self.fusion_projection = None
            self.regression_head = RegressionHead(smiles_encoder.output_dim + morgan_encoder.output_dim, dropout=dropout)
        else:
            self.fusion = BimodalCrossAttentionFusion(
                query_dim=smiles_encoder.output_dim,
                context_dim=morgan_encoder.output_dim,
                fusion_dim=fusion_dim,
                num_heads=fusion_heads,
                feedforward_dim=fusion_dim * 2,
                dropout=dropout,
            )
            self.fusion_projection = ProjectionHead(self.fusion.output_dim, dropout=dropout)
            if fusion == "cross_attention":
                self.regression_head = RegressionHead(self.fusion.output_dim, dropout=dropout)
            elif fusion == "cross_attention_residual":
                self.regression_head = RegressionHead(
                    smiles_encoder.output_dim + self.fusion.output_dim + morgan_encoder.output_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, input_ids, attention_mask, morgan_fp):
        smiles_out = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask)
        morgan_out = self.morgan_encoder(morgan_fp=morgan_fp)
        z_fusion = None
        fusion_pooled = None
        if self.fusion_kind == "concat":
            feature = torch.cat([smiles_out.pooled, morgan_out.pooled], dim=-1)
        else:
            fusion_out = self.fusion(
                query_sequence=smiles_out.sequence,
                context_sequence=morgan_out.sequence,
                query_mask=smiles_out.mask,
                context_mask=morgan_out.mask,
            )
            fusion_pooled = fusion_out.pooled
            z_fusion = self.fusion_projection(fusion_out.pooled)
            if self.fusion_kind == "cross_attention":
                feature = fusion_out.pooled
            else:
                feature = torch.cat([smiles_out.pooled, fusion_out.pooled, morgan_out.pooled], dim=-1)
        return {
            "prediction": self.regression_head(feature),
            "smiles_pooled": smiles_out.pooled,
            "morgan_pooled": morgan_out.pooled,
            "fusion_pooled": fusion_pooled,
            "z_smiles": self.smiles_projection(smiles_out.pooled),
            "z_morgan": self.morgan_projection(morgan_out.pooled),
            "z_fusion": z_fusion,
        }


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_target_transformer(cfg: RunConfig):
    if cfg.target_transform != "power":
        return None
    values = pd.read_csv(cfg.train_data)["rt"].astype(float).to_numpy().reshape(-1, 1)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformer.fit(values)
    return transformer


def inverse_values(values: torch.Tensor, transformer) -> torch.Tensor:
    values = values.detach().cpu()
    if transformer is None:
        return values
    arr = transformer.inverse_transform(values.numpy().reshape(-1, 1)).reshape(-1)
    return torch.tensor(arr, dtype=torch.float32)


def build_loaders(cfg: RunConfig, tokenizer, transformer):
    kwargs = dict(tokenizer=tokenizer, max_length=cfg.max_length, target_transformer=transformer)
    train = MultimodalRTDataset(
        cfg.train_data,
        smiles_augmentation=cfg.train_smiles_augmentation,
        smiles_augment_prob=cfg.smiles_augment_prob,
        smiles_augment_seed=cfg.seed,
        **kwargs,
    )
    val = MultimodalRTDataset(cfg.val_data, **kwargs)
    test = MultimodalRTDataset(cfg.test_data, **kwargs)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers),
        DataLoader(test, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers),
    )


def build_model(cfg: RunConfig):
    layers = parse_layers(cfg.smiles_layers)
    if cfg.morgan and cfg.chemberta:
        smiles_encoder = SMILESEncoder(
            model_name_or_path=str(PRETRAINED),
            layers_to_use=layers,
            lstm_hidden_size=128,
            dropout=cfg.dropout,
            pooling="first_token",
        )
        morgan_encoder = MorganFingerprintEncoder(
            MorganEncoderConfig(
                n_bits=2048,
                chunk_size=16,
                hidden_dim=cfg.morgan_hidden_dim,
                num_layers=cfg.morgan_layers,
                num_heads=cfg.morgan_heads,
                feedforward_dim=cfg.morgan_hidden_dim * 2,
                dropout=cfg.dropout,
            )
        )
        return MultiModalRTModel(smiles_encoder, morgan_encoder, cfg.fusion, cfg.dropout, cfg.fusion_dim, cfg.fusion_heads)
    if cfg.chemberta:
        encoder = (
            SMILESEncoder(str(PRETRAINED), layers_to_use=layers, lstm_hidden_size=128, dropout=cfg.dropout)
            if cfg.bilstm
            else ChemBERTaNoLSTMEncoder(str(PRETRAINED), layers_to_use=layers, dropout=cfg.dropout)
        )
        return SingleBranchRTModel(encoder, dropout=cfg.dropout)
    if cfg.morgan:
        encoder = MorganFingerprintEncoder(
            MorganEncoderConfig(
                n_bits=2048,
                chunk_size=16,
                hidden_dim=cfg.morgan_hidden_dim,
                num_layers=cfg.morgan_layers,
                num_heads=cfg.morgan_heads,
                feedforward_dim=cfg.morgan_hidden_dim * 2,
                dropout=cfg.dropout,
            )
        )
        return SingleBranchRTModel(encoder, dropout=cfg.dropout)
    raise ValueError("At least one of chemberta or morgan must be enabled")


def build_optimizer(cfg: RunConfig, model: nn.Module):
    encoder, other = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name or "roberta" in name:
            encoder.append(param)
        else:
            other.append(param)
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": cfg.encoder_lr})
    if other:
        groups.append({"params": other, "lr": cfg.lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def regression_loss(pred, labels, cfg: RunConfig):
    mae = (pred - labels).abs().mean()
    if cfg.task_loss == "mae":
        return mae
    huber = F.smooth_l1_loss(pred, labels, beta=cfg.huber_beta)
    return (1.0 - cfg.huber_weight) * mae + cfg.huber_weight * huber


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    diff = y_pred - y_true
    abs_diff = diff.abs()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(1e-12)
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(torch.sqrt((diff**2).mean())),
        "medae": float(abs_diff.median()),
        "mape": float((abs_diff / y_true.abs().clamp_min(1e-12)).mean() * 100.0),
        "r2": float(1.0 - ss_res / ss_tot),
        "within_30s_pct": float((abs_diff <= 30).float().mean() * 100.0),
        "within_60s_pct": float((abs_diff <= 60).float().mean() * 100.0),
        "within_120s_pct": float((abs_diff <= 120).float().mean() * 100.0),
    }


def compute_loss(out, labels, original_labels, cfg: RunConfig, align_loss, fusion_loss):
    pred = out["prediction"]
    task = regression_loss(pred, labels, cfg)
    if not cfg.supcr:
        return task, {"task": float(task.detach().cpu()), "align_supcr": 0.0, "fusion_supcr": 0.0}
    align = align_loss(out["z_smiles"], out["z_morgan"], original_labels)
    fusion = labels.new_tensor(0.0)
    if out["z_fusion"] is not None:
        fusion = fusion_loss(out["z_fusion"], original_labels)
    total = task + cfg.omega_align * align.total + cfg.omega_fusion * fusion
    return total, {
        "task": float(task.detach().cpu()),
        "align_supcr": float(align.supcr.detach().cpu()),
        "fusion_supcr": float(fusion.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, cfg: RunConfig, transformer, split_name: str, pred_path: Optional[Path] = None):
    model.eval()
    y_true, y_pred, rows = [], [], []
    for batch in progress(loader, desc=f"{cfg.run_id} {split_name}", leave=False):
        batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch.get("input_ids"), batch.get("attention_mask"), batch.get("morgan_fp"))
        pred_sec = inverse_values(out["prediction"], transformer)
        true_sec = batch["labels_original"].detach().cpu()
        y_true.append(true_sec)
        y_pred.append(pred_sec)
        if pred_path is not None:
            sample_ids = list(batch["sample_id"])
            smiles = list(batch["smiles"])
            for sid, smi, yt, yp in zip(sample_ids, smiles, true_sec.tolist(), pred_sec.tolist()):
                err = float(yp) - float(yt)
                rows.append({"sample_id": sid, "smiles": smi, "true_rt": yt, "pred_rt": yp, "error": err, "abs_error": abs(err)})
    result = metrics(torch.cat(y_true), torch.cat(y_pred))
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "smiles", "true_rt", "pred_rt", "error", "abs_error"])
            writer.writeheader()
            writer.writerows(rows)
    return result


def run_one(cfg: RunConfig) -> Dict[str, object]:
    started = time.time()
    set_seed(cfg.seed)
    run_dir = OUT_ROOT / "runs" / cfg.run_id
    ckpt_dir = CKPT_ROOT / cfg.run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(PRETRAINED))
    transformer = build_target_transformer(cfg)
    train_loader, val_loader, test_loader = build_loaders(cfg, tokenizer, transformer)
    model = build_model(cfg).to(cfg.device)
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    align_loss = CrossModalSupCRLoss(
        temperature=cfg.temperature,
        similarity_weight=cfg.similarity_weight,
        use_log=False,
    )
    fusion_loss = FusionSupCRLoss(temperature=cfg.temperature, use_log=False)

    best_val = math.inf
    best_epoch = -1
    stale = 0
    start_epoch = 1
    history = []
    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        latest = torch.load(latest_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(latest["model_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        history = latest.get("history", [])
        best_val = float(latest.get("best_val", math.inf))
        best_epoch = int(latest.get("best_epoch", -1))
        stale = int(latest.get("stale", 0))
        start_epoch = int(latest.get("epoch", 0)) + 1
    log_path = run_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(start_epoch, cfg.epochs + 1):
            model.train()
            losses = []
            for batch in progress(train_loader, desc=f"{cfg.run_id} epoch {epoch}/{cfg.epochs}", leave=False):
                batch = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(batch.get("input_ids"), batch.get("attention_mask"), batch.get("morgan_fp"))
                loss, parts = compute_loss(out, batch["labels"], batch["labels_original"], cfg, align_loss, fusion_loss)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_metrics = evaluate(model, val_loader, cfg, transformer, "val")
            scheduler.step(val_metrics["mae"])
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if val_metrics["mae"] < best_val:
                best_val = val_metrics["mae"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "args": asdict(cfg),
                    },
                    ckpt_dir / "best.pt",
                )
            else:
                stale += 1
                if stale >= cfg.early_stopping_patience:
                    break
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "args": asdict(cfg),
                },
                latest_path,
            )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_pred = run_dir / "predictions_val.csv"
    test_pred = run_dir / "predictions_test.csv"
    val_metrics = evaluate(model, val_loader, cfg, transformer, "val", val_pred)
    test_metrics = evaluate(model, test_loader, cfg, transformer, "test", test_pred)
    output = {
        "run_id": cfg.run_id,
        "configuration": cfg.configuration,
        "seed": cfg.seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_dir / "best.pt"),
        "prediction_val_path": str(val_pred),
        "prediction_test_path": str(test_pred),
        "metrics_path": str(run_dir / "metrics.json"),
        "runtime_seconds": time.time() - started,
        "val": val_metrics,
        "test": test_metrics,
        "config": asdict(cfg),
    }
    (run_dir / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def predefined_configs(quick: bool = False) -> list[RunConfig]:
    epochs = 2 if quick else 200
    patience = 1 if quick else 20
    base = dict(epochs=epochs, early_stopping_patience=patience)
    return [
        RunConfig("A_chemberta_seed2026", "A ChemBERTa", 2026, bilstm=False, morgan=False, fusion="none", supcr=False, **base),
        RunConfig("B_chemberta_bilstm_seed2026", "B ChemBERTa + BiLSTM", 2026, bilstm=True, morgan=False, fusion="none", supcr=False, train_smiles_augmentation=True, **base),
        RunConfig("C_concat_no_supcr_seed2026", "C ChemBERTa + BiLSTM + Morgan concat", 2026, fusion="concat", supcr=False, **base),
        RunConfig("G_cross_attention_no_supcr_seed2026", "G ChemBERTa + BiLSTM + Morgan cross-attention no SupCR", 2026, fusion="cross_attention", supcr=False, **base),
        RunConfig("D_residual_no_supcr_seed2026", "D ChemBERTa + BiLSTM + Morgan residual cross-attention no SupCR", 2026, fusion="cross_attention_residual", supcr=False, **base),
        RunConfig("E_residual_supcr_seed2026", "E ChemBERTa + BiLSTM + Morgan residual cross-attention + SupCR", 2026, fusion="cross_attention_residual", supcr=True, **base),
        RunConfig("D_residual_no_supcr_seed777", "D SupCR strict w/o SupCR", 777, fusion="cross_attention_residual", supcr=False, **base),
        RunConfig("E_residual_supcr_seed777", "E SupCR strict with SupCR", 777, fusion="cross_attention_residual", supcr=True, **base),
        RunConfig("D_residual_no_supcr_seed3407", "D SupCR strict w/o SupCR", 3407, fusion="cross_attention_residual", supcr=False, **base),
        RunConfig("E_residual_supcr_seed3407", "E SupCR strict with SupCR", 3407, fusion="cross_attention_residual", supcr=True, **base),
    ]


def ensure_scaffold_split_files() -> None:
    split_path = SCAFFOLD_DIR / "scaffold_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing scaffold split file: {split_path}")
    df = pd.read_csv(split_path)
    if "scaffold_split" not in df.columns:
        raise ValueError(f"Scaffold split file lacks scaffold_split column: {split_path}")
    keep_cols = [col for col in ["sample_id", "smiles", "rt", "split", "canonical_smiles", "scaffold", "scaffold_split"] if col in df.columns]
    for split, path in SCAFFOLD_SPLIT_FILES.items():
        part = df.loc[df["scaffold_split"] == split, keep_cols].copy()
        if part.empty:
            raise ValueError(f"Scaffold split has no rows for {split}: {split_path}")
        part.to_csv(path, index=False)


def scaffold_config(quick: bool = False) -> RunConfig:
    ensure_scaffold_split_files()
    epochs = 2 if quick else 200
    patience = 1 if quick else 20
    return RunConfig(
        "scaffold_residual_supcr_seed2026",
        "Scaffold split residual cross-attention + SupCR",
        2026,
        fusion="cross_attention_residual",
        supcr=True,
        train_data=str(SCAFFOLD_SPLIT_FILES["train"]),
        val_data=str(SCAFFOLD_SPLIT_FILES["val"]),
        test_data=str(SCAFFOLD_SPLIT_FILES["test"]),
        epochs=epochs,
        early_stopping_patience=patience,
    )


def export_scaffold_artifacts(result: Dict[str, object]) -> None:
    if result.get("status") != "completed":
        return
    if result.get("run_id") != "scaffold_residual_supcr_seed2026":
        return
    scaffold_dir = OUT_ROOT / "04_scaffold_generalization"
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    metrics_src = Path(str(result["metrics_path"]))
    pred_src = Path(str(result["prediction_test_path"]))
    ckpt_src = Path(str(result["checkpoint_path"]))
    (scaffold_dir / "metrics_scaffold.json").write_bytes(metrics_src.read_bytes())
    (scaffold_dir / "predictions_scaffold_test.csv").write_bytes(pred_src.read_bytes())
    (scaffold_dir / "checkpoint_scaffold_best_path.txt").write_text(str(ckpt_src), encoding="utf-8")


def load_relevant_results(configs: list[RunConfig]) -> list[Dict[str, object]]:
    results: list[Dict[str, object]] = []
    for cfg in configs:
        run_dir = OUT_ROOT / "runs" / cfg.run_id
        metrics_path = run_dir / "metrics.json"
        failure_path = run_dir / "failure.json"
        if metrics_path.exists():
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        elif failure_path.exists():
            data = json.loads(failure_path.read_text(encoding="utf-8"))
            if data.get("run_id") == cfg.run_id:
                results.append(data)
    return results


def paired_bootstrap_for_supcr(rows: list[dict]) -> None:
    supcr_dir = OUT_ROOT / "02_supcr_strict_ablation"
    by_seed = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        cfg = row.get("config", {})
        if cfg.get("fusion") != "cross_attention_residual":
            continue
        seed = cfg.get("seed")
        key = "E" if cfg.get("supcr") else "D"
        by_seed.setdefault(seed, {})[key] = row
    out_rows = []
    rng = np.random.default_rng(20260825)
    for seed, pair in sorted(by_seed.items()):
        if "D" not in pair or "E" not in pair:
            continue
        d = pd.read_csv(pair["D"]["prediction_test_path"])
        e = pd.read_csv(pair["E"]["prediction_test_path"])
        merged = d[["sample_id", "true_rt", "pred_rt"]].merge(
            e[["sample_id", "pred_rt"]],
            on="sample_id",
            suffixes=("_D", "_E"),
            validate="one_to_one",
        )
        d_abs = (merged["pred_rt_D"] - merged["true_rt"]).abs().to_numpy()
        e_abs = (merged["pred_rt_E"] - merged["true_rt"]).abs().to_numpy()
        diff = d_abs - e_abs
        boot = np.empty(10000, dtype=float)
        n = len(diff)
        for i in range(len(boot)):
            sample = rng.integers(0, n, size=n)
            boot[i] = float(diff[sample].mean())
        out_rows.append(
            {
                "seed": seed,
                "n": n,
                "D_minus_E_mae_diff_s": float(diff.mean()),
                "ci95_low_s": float(np.quantile(boot, 0.025)),
                "ci95_high_s": float(np.quantile(boot, 0.975)),
                "bootstrap_resamples": len(boot),
            }
        )
    pd.DataFrame(out_rows).to_csv(supcr_dir / "supcr_paired_bootstrap.csv", index=False)


def update_results_package(rows: list[dict]) -> None:
    pkg = OUT_ROOT / "PAPER_RESULTS_PACKAGE"
    pkg.mkdir(parents=True, exist_ok=True)
    copies = [
        (OUT_ROOT / "01_component_ablation" / "component_ablation_results.csv", pkg / "01_component_ablation.csv"),
        (OUT_ROOT / "02_supcr_strict_ablation" / "supcr_seed_results.csv", pkg / "02_supcr_ablation.csv"),
        (OUT_ROOT / "03_fusion_ablation" / "fusion_ablation_results.csv", pkg / "03_fusion_ablation.csv"),
        (OUT_ROOT / "all_metrics_summary.csv", pkg / "all_metrics_summary.csv"),
    ]
    for src, dst in copies:
        if src.exists():
            dst.write_bytes(src.read_bytes())
    manifest = []
    for row in rows:
        cfg = row.get("config", {})
        manifest.append(
            {
                "experiment_id": row.get("run_id"),
                "purpose": row.get("configuration"),
                "dataset": "SMRT unified",
                "split": "train=63163/val=7019/test=7798",
                "seed": row.get("seed", cfg.get("seed")),
                "config": json.dumps(cfg, ensure_ascii=False),
                "script": "scripts/paper_ablation_runner_20260825.py",
                "checkpoint": row.get("checkpoint_path", ""),
                "prediction": row.get("prediction_test_path", ""),
                "metrics": row.get("metrics_path", str(OUT_ROOT / "runs" / str(row.get("run_id")) / "failure.json")),
                "status": row.get("status", "partial"),
            }
        )
    pd.DataFrame(manifest).to_csv(pkg / "experiment_manifest.csv", index=False)


def write_summary(results: list[Dict[str, object]]) -> None:
    comp_dir = OUT_ROOT / "01_component_ablation"
    supcr_dir = OUT_ROOT / "02_supcr_strict_ablation"
    fusion_dir = OUT_ROOT / "03_fusion_ablation"
    comp_dir.mkdir(parents=True, exist_ok=True)
    supcr_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        cfg = r.get("config", {})
        test = r.get("test", {})
        val = r.get("val", {})
        rows.append(
            {
                "run_id": r.get("run_id"),
                "configuration": r.get("configuration"),
                "seed": r.get("seed"),
                "ChemBERTa": cfg.get("chemberta"),
                "BiLSTM": cfg.get("bilstm"),
                "Morgan": cfg.get("morgan"),
                "fusion": cfg.get("fusion"),
                "SupCR": cfg.get("supcr"),
                "MAE_val": val.get("mae"),
                "MAE_test": test.get("mae"),
                "RMSE_test": test.get("rmse"),
                "MedAE_test": test.get("medae"),
                "MAPE_test": test.get("mape"),
                "R2_test": test.get("r2"),
                "best_epoch": r.get("best_epoch"),
                "checkpoint_path": r.get("checkpoint_path"),
                "prediction_path": r.get("prediction_test_path"),
                "status": r.get("status"),
            }
        )
    all_rows = pd.DataFrame(rows)
    component_order = [
        "A_chemberta_seed2026",
        "B_chemberta_bilstm_seed2026",
        "C_concat_no_supcr_seed2026",
        "D_residual_no_supcr_seed2026",
        "E_residual_supcr_seed2026",
    ]
    comp = all_rows[all_rows["run_id"].isin(component_order)].copy()
    if not comp.empty:
        comp["_order"] = comp["run_id"].map({run_id: idx for idx, run_id in enumerate(component_order)})
        comp = comp.sort_values("_order").drop(columns=["_order"])
    comp.to_csv(comp_dir / "component_ablation_results.csv", index=False)
    all_rows.to_csv(OUT_ROOT / "all_metrics_summary.csv", index=False)

    gains = []
    comp_by_run = {row["run_id"]: row for row in comp.to_dict("records")}
    for before, after in zip(component_order, component_order[1:]):
        left = comp_by_run.get(before)
        right = comp_by_run.get(after)
        if not left or not right or left.get("MAE_test") is None or right.get("MAE_test") is None:
            continue
        before_mae = float(left["MAE_test"])
        after_mae = float(right["MAE_test"])
        improvement = before_mae - after_mae
        gains.append(
            {
                "comparison": f"{before} -> {after}",
                "before_run_id": before,
                "after_run_id": after,
                "MAE_test_before": before_mae,
                "MAE_test_after": after_mae,
                "absolute_MAE_improvement": improvement,
                "relative_MAE_improvement_pct": improvement / before_mae * 100.0 if before_mae else None,
            }
        )
    pd.DataFrame(gains).to_csv(comp_dir / "component_stepwise_gains.csv", index=False)

    sup = pd.DataFrame([x for x in rows if str(x["run_id"]).startswith(("D_residual_no_supcr_seed", "E_residual_supcr_seed"))])
    sup.to_csv(supcr_dir / "supcr_seed_results.csv", index=False)
    if not sup.empty:
        summary = sup.groupby("SupCR").agg({"MAE_test": ["mean", "std"], "RMSE_test": ["mean", "std"], "MedAE_test": ["mean", "std"], "MAPE_test": ["mean", "std"], "R2_test": ["mean", "std"]})
        (supcr_dir / "supcr_summary.json").write_text(summary.to_json(indent=2), encoding="utf-8")
    paired_bootstrap_for_supcr(results)

    fus = pd.DataFrame([x for x in rows if x["fusion"] in {"concat", "cross_attention", "cross_attention_residual"} and x["SupCR"] is False and x["seed"] == 2026])
    fus.to_csv(fusion_dir / "fusion_ablation_results.csv", index=False)
    update_results_package(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--quick", action="store_true", help="Use 2 epochs for runtime smoke execution.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scaffold", action="store_true", help="Train the representative scaffold-split final core model.")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.scaffold:
        configs = [scaffold_config(quick=args.quick)]
    else:
        configs = predefined_configs(quick=args.quick)
    if args.run_id:
        configs = [c for c in configs if c.run_id == args.run_id]
    elif not args.all:
        configs = configs[:1]
    results = []
    for cfg in configs:
        try:
            results.append(run_one(cfg))
        except Exception as exc:
            run_dir = OUT_ROOT / "runs" / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            fail = {"run_id": cfg.run_id, "configuration": cfg.configuration, "seed": cfg.seed, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "config": asdict(cfg)}
            (run_dir / "failure.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
            results.append(fail)
        if results:
            export_scaffold_artifacts(results[-1])
        write_summary(load_relevant_results(predefined_configs(quick=args.quick)))
    write_summary(load_relevant_results(predefined_configs(quick=args.quick)))


if __name__ == "__main__":
    main()
