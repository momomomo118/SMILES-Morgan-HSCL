from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from run_abcort_fixed_split import (
    load_official_modules,
    molecule_to_data,
    read_fixed_test,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the official ABCoRT architecture on fixed SMRT splits."
    )
    parser.add_argument("--abcort-repo", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_object(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_or_build_graphs(
    csv_path: Path,
    cache_path: Path,
    rebuild: bool,
    atom_featurizer,
    bond_featurizer,
) -> tuple[list[dict[str, str]], list[Data]]:
    rows = read_fixed_test(csv_path)
    source_hash = sha256(csv_path)
    if cache_path.is_file() and not rebuild:
        payload = load_torch_object(cache_path)
        if (
            payload.get("source_sha256") == source_hash
            and payload.get("sample_ids") == [row["sample_id"] for row in rows]
        ):
            graphs = payload["graphs"]
            if len(graphs) == len(rows):
                print(f"loaded_cache={cache_path} n={len(graphs)}", flush=True)
                return rows, graphs

    graphs: list[Data] = []
    for row in tqdm(rows, desc=f"featurize {csv_path.stem}"):
        graphs.append(
            molecule_to_data(
                row["smiles"], float(row["rt"]), atom_featurizer, bond_featurizer
            )
        )
    if len(graphs) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} graphs, built {len(graphs)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_sha256": source_hash,
            "sample_ids": [row["sample_id"] for row in rows],
            "graphs": graphs,
        },
        cache_path,
    )
    print(f"saved_cache={cache_path} n={len(graphs)}", flush=True)
    return rows, graphs


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    graphs: list[Data],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = torch.nn.SmoothL1Loss()
    loss_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device, non_blocking=True)
        target = batch.y.reshape(-1)
        prediction = model(batch).reshape(-1)
        loss = criterion(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_count = target.numel()
        loss_sum += loss.item() * batch_count
        count += batch_count
    return loss_sum / count


@torch.inference_mode()
def evaluate(model, loader, device, description: str) -> tuple[dict[str, float], list[float]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    for batch in tqdm(loader, desc=description, leave=False):
        batch = batch.to(device, non_blocking=True)
        prediction = model(batch).reshape(-1)
        y_true.extend(batch.y.reshape(-1).detach().cpu().double().tolist())
        y_pred.extend(prediction.detach().cpu().double().tolist())
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    error = pred - true
    abs_error = np.abs(error)
    metrics = {
        "mae_s": float(abs_error.mean()),
        "smooth_l1": float(
            np.where(abs_error < 1.0, 0.5 * error**2, abs_error - 0.5).mean()
        ),
        "rmse_s": float(np.sqrt(np.mean(error**2))),
    }
    return metrics, y_pred


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_mae: float,
    bad_epochs: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "bad_epochs": bad_epochs,
            "config": vars(args),
        },
        path,
    )


def write_predictions(
    path: Path, rows: list[dict[str, str]], predictions: list[float]
) -> None:
    if len(rows) != len(predictions):
        raise RuntimeError(f"rows={len(rows)} predictions={len(predictions)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "graph_cache"
    history_path = args.output_dir / "history.csv"
    best_path = args.output_dir / "checkpoint_best.pt"
    last_path = args.output_dir / "checkpoint_last.pt"

    set_seed(args.seed)
    MyNet, atom_featurizer, bond_featurizer = load_official_modules(args.abcort_repo)
    train_rows, train_graphs = load_or_build_graphs(
        args.train_csv,
        cache_dir / "train.pt",
        args.rebuild_cache,
        atom_featurizer,
        bond_featurizer,
    )
    val_rows, val_graphs = load_or_build_graphs(
        args.val_csv,
        cache_dir / "val.pt",
        args.rebuild_cache,
        atom_featurizer,
        bond_featurizer,
    )
    test_rows, test_graphs = load_or_build_graphs(
        args.test_csv,
        cache_dir / "test.pt",
        args.rebuild_cache,
        atom_featurizer,
        bond_featurizer,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_graphs,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
        pin_memory,
    )
    val_loader = make_loader(
        val_graphs,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        pin_memory,
    )
    test_loader = make_loader(
        test_graphs,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        pin_memory,
    )

    model = MyNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs
    )
    start_epoch = 0
    best_val_mae = float("inf")
    bad_epochs = 0
    if args.resume and last_path.is_file():
        state = load_torch_object(last_path)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_val_mae = float(state["best_val_mae"])
        bad_epochs = int(state["bad_epochs"])
        print(f"resumed_from_epoch={start_epoch}", flush=True)

    print(f"device={device}", flush=True)
    print(
        f"n_train={len(train_rows)} n_val={len(val_rows)} n_test={len(test_rows)}",
        flush=True,
    )
    print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)

    write_header = not history_path.is_file() or start_epoch == 0
    history_mode = "w" if write_header else "a"
    with history_path.open(history_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "learning_rate",
                "train_smooth_l1",
                "val_smooth_l1",
                "val_mae_s",
                "val_rmse_s",
                "best_val_mae_s",
                "bad_epochs",
            ],
        )
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics, _ = evaluate(model, val_loader, device, "validation")
            scheduler.step()
            improved = val_metrics["mae_s"] < best_val_mae
            if improved:
                best_val_mae = val_metrics["mae_s"]
                bad_epochs = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_mae,
                    bad_epochs,
                    args,
                )
            else:
                bad_epochs += 1
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_mae,
                bad_epochs,
                args,
            )
            record = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_smooth_l1": train_loss,
                "val_smooth_l1": val_metrics["smooth_l1"],
                "val_mae_s": val_metrics["mae_s"],
                "val_rmse_s": val_metrics["rmse_s"],
                "best_val_mae_s": best_val_mae,
                "bad_epochs": bad_epochs,
            }
            writer.writerow(record)
            f.flush()
            print(json.dumps(record), flush=True)
            if bad_epochs >= args.patience:
                print(f"early_stopping_epoch={epoch + 1}", flush=True)
                break

    if not best_path.is_file():
        raise FileNotFoundError(f"No best checkpoint was written to {best_path}")
    best_state = load_torch_object(best_path)
    model.load_state_dict(best_state["model"], strict=True)
    test_metrics, test_predictions = evaluate(model, test_loader, device, "test")
    write_predictions(args.predictions, test_rows, test_predictions)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": int(best_state["epoch"]) + 1,
                "best_val_mae_s": float(best_state["best_val_mae"]),
                "test_sanity_metrics": test_metrics,
                "predictions": str(args.predictions),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(test_metrics), flush=True)
    print(f"saved_predictions={args.predictions}", flush=True)


if __name__ == "__main__":
    main()
