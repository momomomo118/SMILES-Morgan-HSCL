"""Train the current multimodal model repeatedly on one fixed batch."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_multimodal import build_loss_fns, build_model, build_optimizer, compute_losses, move_batch  # noqa: E402
from utils.multimodal_dataset import MultimodalRTDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=PROJECT_ROOT / "data" / "unified" / "train.csv")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = args.stage_dir / "sanity_single_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_args = argparse.Namespace(
        smiles_model=str(PROJECT_ROOT / "smiles_baseline" / "RT-TR-main" / "RT-TR-main" / "seyonec" / "ChemBERTa-zinc-base-v1"),
        freeze_smiles=False,
        freeze_morgan=False,
        morgan_hidden_dim=256,
        morgan_layers=2,
        morgan_heads=8,
        fusion="cross_attention_residual",
        fusion_dim=256,
        fusion_heads=8,
        dropout=0.0,
        lr=1e-4,
        encoder_lr=1e-5,
        weight_decay=1e-3,
        grad_clip=1.0,
        temperature=0.1,
        similarity_weight=0.1,
        omega_align=0.01,
        omega_fusion=0.005,
        supcr_use_log=False,
        max_pairs_per_anchor=None,
        device=args.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_args.smiles_model)
    dataset = MultimodalRTDataset(args.train_data, tokenizer=tokenizer, max_length=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = move_batch(next(iter(loader)), args.device)

    model = build_model(train_args).to(args.device)
    optimizer = build_optimizer(train_args, model)
    align_loss_fn, fusion_loss_fn = build_loss_fns(train_args)

    rows = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            morgan_fp=batch["morgan_fp"],
        )
        losses = compute_losses(output, batch["labels"], align_loss_fn, fusion_loss_fn, train_args)
        losses["total"].backward()
        if train_args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            rows.append(
                {
                    "step": step,
                    "total_loss": float(losses["total"].detach().cpu()),
                    "task_loss": float(losses["task"].detach().cpu()),
                    "align_supcr": float(losses["align_supcr"].detach().cpu()),
                    "fusion_supcr": float(losses["fusion_supcr"].detach().cpu()),
                }
            )

    csv_path = out_dir / "single_batch_loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(7, 4.5))
    plt.plot([r["step"] for r in rows], [r["task_loss"] for r in rows], marker="o", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Task MAE loss")
    plt.title("Single-batch overfit")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "single_batch_loss_curve.png", dpi=180)
    plt.close()
    print(f"Wrote {csv_path}")
    print(f"Initial task loss: {rows[0]['task_loss']:.6f}")
    print(f"Final task loss: {rows[-1]['task_loss']:.6f}")


if __name__ == "__main__":
    main()
