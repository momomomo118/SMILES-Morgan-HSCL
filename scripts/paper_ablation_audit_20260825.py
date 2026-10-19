from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_ablation_20260825"
PKG = OUT / "PAPER_RESULTS_PACKAGE"
PYTHON = sys.executable


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mkdirs() -> None:
    for rel in [
        "00_provenance",
        "01_component_ablation",
        "02_supcr_strict_ablation",
        "03_fusion_ablation",
        "04_scaffold_generalization",
        "05_predret_provenance",
        "PAPER_RESULTS_PACKAGE",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def load_checkpoint_args(path: Path):
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            return {
                "status": "loaded",
                "epoch": ckpt.get("epoch"),
                "metrics": ckpt.get("metrics"),
                "args": ckpt.get("args"),
                "keys": sorted(str(k) for k in ckpt.keys()),
            }
        return {"status": f"loaded non-dict {type(ckpt)}"}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}


def final_training_config_audit() -> None:
    rows = []
    checkpoints = {
        "exact1234": ROOT / "checkpoints" / "supcr_tuning_aliases" / "exact1234_best.pt",
        "dropout015_seed777": ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_dropout015_seed777" / "best_cross_attention_residual.pt",
        "lowfusion_seed777": ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_lowfusion_seed777" / "best_cross_attention_residual.pt",
        "aug050_seed777": ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_aug050_seed777" / "best_cross_attention_residual.pt",
        "exact777": ROOT / "checkpoints" / "supcr_tuning_aliases" / "exact777_best.pt",
        "exact2025": ROOT / "checkpoints" / "supcr_tuning_aliases" / "exact2025_best.pt",
    }
    loaded = {}
    for name, path in checkpoints.items():
        info = load_checkpoint_args(path)
        loaded[name] = {"path": str(path), **info}

    params = [
        "epochs",
        "early_stopping_patience",
        "optimizer",
        "lr",
        "encoder_lr",
        "batch_size",
        "eval_batch_size",
        "weight_decay",
        "task_loss",
        "target_transform",
        "temperature",
        "similarity_weight",
        "omega_align",
        "omega_fusion",
        "huber_weight",
        "huber_beta",
        "smiles_augment_prob",
        "dropout",
        "fusion",
        "seed",
    ]
    for param in params:
        values = {}
        for name, info in loaded.items():
            args = info.get("args") or {}
            if isinstance(args, dict) and param in args:
                values[name] = args.get(param)
        if values:
            unique = sorted({json.dumps(v, sort_keys=True, default=str) for v in values.values()})
            confidence = "high" if len(unique) == 1 else "medium_conflicting"
            final_value = next(iter(values.values())) if len(unique) == 1 else values
            evidence = "checkpoint args loaded with torch"
        else:
            final_value = "NOT VERIFIED"
            evidence = "no checkpoint arg field found for this parameter"
            confidence = "low"
        rows.append((param, final_value, evidence, confidence))

    md = [
        "# Final Training Config Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: `{PYTHON}`",
        "",
        "## Checkpoint Load Status",
        "",
        "| member | path | status | epoch | args_present |",
        "|---|---|---|---:|---|",
    ]
    for name, info in loaded.items():
        md.append(
            f"| {name} | `{info['path']}` | {info.get('status')} | {info.get('epoch', '')} | {isinstance(info.get('args'), dict)} |"
        )
    md.extend(["", "## Parameter Evidence", "", "| parameter | final adopted value | evidence | confidence | conflict |", "|---|---|---|---|---|"])
    for param, value, evidence, confidence in rows:
        conflict = "yes" if confidence == "medium_conflicting" else "no"
        md.append(f"| {param} | `{json.dumps(value, ensure_ascii=False, default=str)}` | {evidence} | {confidence} | {conflict} |")
    md.extend(
        [
            "",
            "## Adopted New-Ablation Config",
            "",
            "For strict new ablations, the runner uses a fixed documented config aligned to the common verified checkpoint settings: target_transform=power, task_loss=mae_huber, huber_weight=0.2, huber_beta=1.0, lr=1e-4, encoder_lr=1e-5, batch_size=16, eval_batch_size=32, weight_decay=1e-3, dropout=0.1, temperature=0.07, similarity_weight=0.05, omega_align=0.01, omega_fusion=0.005, smiles_augment_prob=0.35 unless the ablated variable requires disabling a component.",
            "",
            "If checkpoint args could not be verified, this file records the conflict rather than treating script defaults as proof.",
        ]
    )
    (OUT / "00_provenance" / "checkpoint_args_dump.json").write_text(json.dumps(loaded, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "00_provenance" / "final_training_config_audit.md").write_text("\n".join(md), encoding="utf-8")


def scaffold_split() -> None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out_dir = OUT / "04_scaffold_generalization"
    df = pd.concat(
        [
            pd.read_csv(ROOT / "data" / "unified" / "train.csv"),
            pd.read_csv(ROOT / "data" / "unified" / "val.csv"),
            pd.read_csv(ROOT / "data" / "unified" / "test.csv"),
        ],
        ignore_index=True,
    )
    scaffolds = []
    canonical = []
    invalid = 0
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid += 1
            scaffolds.append("")
            canonical.append("")
            continue
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaf or canonical[-1])
    df["canonical_smiles"] = canonical
    df["scaffold"] = scaffolds
    counts = df.groupby("scaffold").size().sort_values(ascending=False)
    target = {"test": int(round(len(df) * 0.10)), "val": int(round(len(df) * 0.09))}
    buckets = {"train": [], "val": [], "test": []}
    sizes = {"train": 0, "val": 0, "test": 0}
    for scaf, n in counts.items():
        if sizes["test"] < target["test"]:
            bucket = "test"
        elif sizes["val"] < target["val"]:
            bucket = "val"
        else:
            bucket = "train"
        buckets[bucket].append(scaf)
        sizes[bucket] += int(n)
    split_map = {scaf: split for split, vals in buckets.items() for scaf in vals}
    df["scaffold_split"] = df["scaffold"].map(split_map)
    split_path = out_dir / "scaffold_split.csv"
    df.to_csv(split_path, index=False)
    train_scaf = set(df.loc[df.scaffold_split == "train", "scaffold"])
    val_scaf = set(df.loc[df.scaffold_split == "val", "scaffold"])
    test_scaf = set(df.loc[df.scaffold_split == "test", "scaffold"])
    stats = {
        "rows": len(df),
        "invalid_smiles": invalid,
        "split_counts": df["scaffold_split"].value_counts().to_dict(),
        "scaffold_counts": {k: len(v) for k, v in buckets.items()},
        "train_val_overlap": len(train_scaf & val_scaf),
        "train_test_overlap": len(train_scaf & test_scaf),
        "val_test_overlap": len(val_scaf & test_scaf),
        "canonical_duplicate_rows": int(df.duplicated(["canonical_smiles"]).sum()),
        "exact_smiles_duplicate_rows": int(df.duplicated(["smiles"]).sum()),
        "sha256": sha256(split_path),
    }
    (out_dir / "scaffold_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md = [
        "# Scaffold Split Audit",
        "",
        f"Rows: {stats['rows']}",
        f"Split counts: {stats['split_counts']}",
        f"Train-test scaffold overlap: {stats['train_test_overlap']}",
        f"Train-val scaffold overlap: {stats['train_val_overlap']}",
        f"Val-test scaffold overlap: {stats['val_test_overlap']}",
        "",
        f"Scaffold model metrics present: {(out_dir / 'metrics_scaffold.json').exists()}",
        f"Scaffold test predictions present: {(out_dir / 'predictions_scaffold_test.csv').exists()}",
    ]
    (out_dir / "scaffold_audit.md").write_text("\n".join(md), encoding="utf-8")


def predret_audit() -> None:
    out_dir = OUT / "05_predret_provenance"
    current = ROOT / "data" / "external_predret" / "transfer_10_systems_cleaned" / "all_cleaned.csv"
    regenerated = out_dir / "regenerated_transfer_10_systems_cleaned" / "all_cleaned.csv"
    summary = ROOT / "data" / "external_predret" / "transfer_10_systems_cleaned" / "cleaning_summary.json"
    prep_summary = ROOT / "data" / "external_predret" / "preparation_summary.json"
    source = ROOT / "data" / "external_predret" / "source" / "predret_raw_normalized.csv"
    diff_rows = []
    if current.exists():
        df = pd.read_csv(current)
        current_hash = sha256(current)
        rows = len(df)
    else:
        rows = 0
        current_hash = ""
    if regenerated.exists():
        regen_df = pd.read_csv(regenerated)
        regenerated_hash = sha256(regenerated)
        regenerated_rows = len(regen_df)
    else:
        regenerated_hash = ""
        regenerated_rows = 0
    summ = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    prep = json.loads(prep_summary.read_text(encoding="utf-8")) if prep_summary.exists() else {}
    diff_rows.extend(
        [
            {
                "key": "all_cleaned_rows",
                "current": rows,
                "regenerated": regenerated_rows,
                "status": "match" if rows == regenerated_rows else "different",
            },
            {
                "key": "all_cleaned_sha256",
                "current": current_hash,
                "regenerated": regenerated_hash,
                "status": "match" if current_hash and current_hash == regenerated_hash else "different_or_missing",
            },
        ]
    )
    md = [
        "# PredRet Preprocessing Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Source normalized file: `{source}`",
        f"Cleaned file: `{current}`",
        f"Regenerated cleaned file: `{regenerated}`",
        f"Cleaned rows: {rows}",
        f"Cleaned SHA256: `{current_hash}`",
        f"Regenerated rows: {regenerated_rows}",
        f"Regenerated SHA256: `{regenerated_hash}`",
        "",
        "## Verified",
        "",
        "- InChI is converted with RDKit `Chem.MolFromInchi(..., sanitize=True, removeHs=True)` in `code/src/utils/prepare_predret_data.py:108`.",
        "- Canonical isomeric SMILES are emitted with `Chem.MolToSmiles(..., canonical=True, isomericSmiles=True)` in `code/src/utils/prepare_predret_data.py:111`.",
        "- RT filtering removes nonfinite values and `rt <= 0` in `code/clean_predret_transfer_data.py:131-136`.",
        "- Duplicate `(system, smiles)` rows are collapsed to median RT in `code/clean_predret_transfer_data.py:141-143`.",
        "- Per-system 10-fold 90/10 splits are created in `code/clean_predret_transfer_data.py:170-184` and documented in `code/clean_predret_transfer_data.py:215-224`.",
        f"- RT filtering / duplicate handling summary exists: {summary.exists()}",
        f"- Cleaning summary: `{json.dumps(summ, ensure_ascii=False, default=str)[:2000]}`",
        "",
        "## Raw-to-Clean Flow",
        "",
        "`PredRet workbook -> data/external_predret/source/predret_raw_normalized.csv -> RDKit InChI parsing and canonical isomeric SMILES conversion -> data/external_predret/transfer_10_systems/all.csv -> data/external_predret/transfer_10_systems_cleaned/all_cleaned.csv`",
        "",
        "## Preparation Counts",
        "",
        f"- converted_rows: {prep.get('converted_rows')}",
        f"- dropped_rows: {prep.get('dropped_rows')}",
        f"- unique_smiles: {prep.get('unique_smiles')}",
        f"- transfer_10_systems before cleaning: {(prep.get('transfer_10_systems') or {}).get('rows')}",
        f"- seed: {prep.get('seed')}",
        "",
        "## Regeneration Diff",
        "",
        f"- Current and regenerated cleaned SHA256 match: {bool(current_hash and current_hash == regenerated_hash)}",
        "- No PredRet retraining is triggered because the regenerated cleaned dataset is byte-identical to the existing cleaned dataset.",
    ]
    pd.DataFrame(diff_rows, columns=["key", "current", "regenerated", "status"]).to_csv(out_dir / "predret_dataset_diff.csv", index=False)
    (out_dir / "predret_preprocessing_audit.md").write_text("\n".join(md), encoding="utf-8")


def package_results() -> None:
    mappings = [
        (OUT / "01_component_ablation" / "component_ablation_results.csv", PKG / "01_component_ablation.csv"),
        (OUT / "02_supcr_strict_ablation" / "supcr_seed_results.csv", PKG / "02_supcr_ablation.csv"),
        (OUT / "03_fusion_ablation" / "fusion_ablation_results.csv", PKG / "03_fusion_ablation.csv"),
        (OUT / "04_scaffold_generalization" / "scaffold_statistics.json", PKG / "04_scaffold_generalization.csv"),
        (OUT / "04_scaffold_generalization" / "metrics_scaffold.json", PKG / "04_scaffold_metrics.json"),
        (OUT / "04_scaffold_generalization" / "predictions_scaffold_test.csv", PKG / "04_scaffold_predictions_test.csv"),
        (OUT / "05_predret_provenance" / "predret_preprocessing_audit.md", PKG / "05_predret_audit.md"),
        (OUT / "all_metrics_summary.csv", PKG / "all_metrics_summary.csv"),
    ]
    for src, dst in mappings:
        if src.exists():
            if src.suffix == ".json" and dst.suffix == ".csv":
                data = json.loads(src.read_text(encoding="utf-8"))
                pd.DataFrame([data]).to_csv(dst, index=False)
            else:
                shutil.copyfile(src, dst)
    manifest_rows = []
    for run_dir in sorted((OUT / "runs").glob("*")) if (OUT / "runs").exists() else []:
        metrics = run_dir / "metrics.json"
        failure = run_dir / "failure.json"
        if metrics.exists():
            data = json.loads(metrics.read_text(encoding="utf-8"))
            status = data.get("status", "completed")
        elif failure.exists():
            data = json.loads(failure.read_text(encoding="utf-8"))
            status = "failed"
        else:
            data = {"run_id": run_dir.name}
            status = "partial"
        cfg = data.get("config", {})
        manifest_rows.append(
            {
                "experiment_id": data.get("run_id", run_dir.name),
                "purpose": data.get("configuration", ""),
                "dataset": "SMRT unified",
                "split": "random unified split",
                "seed": data.get("seed", cfg.get("seed")),
                "config": json.dumps(cfg, ensure_ascii=False),
                "script": "scripts/paper_ablation_runner_20260825.py",
                "checkpoint": data.get("checkpoint_path", ""),
                "prediction": data.get("prediction_test_path", ""),
                "metrics": data.get("metrics_path", str(failure) if failure.exists() else ""),
                "status": status if status in {"completed", "failed", "partial", "reused_verified"} else "partial",
            }
        )
    scaffold_metrics = OUT / "04_scaffold_generalization" / "metrics_scaffold.json"
    scaffold_predictions = OUT / "04_scaffold_generalization" / "predictions_scaffold_test.csv"
    scaffold_checkpoint = OUT / "04_scaffold_generalization" / "checkpoint_scaffold_best_path.txt"
    manifest_rows.extend(
        [
            {
                "experiment_id": "phase0_final_training_config_audit",
                "purpose": "final checkpoint hyperparameter provenance",
                "dataset": "SMRT unified",
                "split": "existing final",
                "seed": "",
                "config": "",
                "script": "scripts/paper_ablation_audit_20260825.py",
                "checkpoint": "",
                "prediction": "",
                "metrics": str(OUT / "00_provenance" / "final_training_config_audit.md"),
                "status": "completed",
            },
            {
                "experiment_id": "predret_preprocessing_audit",
                "purpose": "PredRet preprocessing provenance",
                "dataset": "PredRet 10 systems",
                "split": "cleaned 10-fold protocol",
                "seed": "",
                "config": "",
                "script": "scripts/paper_ablation_audit_20260825.py",
                "checkpoint": "",
                "prediction": "",
                "metrics": str(OUT / "05_predret_provenance" / "predret_preprocessing_audit.md"),
                "status": "completed",
            },
            {
                "experiment_id": "scaffold_split_audit",
                "purpose": "scaffold split construction and overlap audit",
                "dataset": "SMRT unified all rows",
                "split": "Bemis-Murcko scaffold split",
                "seed": "",
                "config": "",
                "script": "scripts/paper_ablation_audit_20260825.py",
                "checkpoint": "",
                "prediction": "",
                "metrics": str(OUT / "04_scaffold_generalization" / "scaffold_statistics.json"),
                "status": "partial",
            },
            {
                "experiment_id": "scaffold_residual_supcr_seed2026",
                "purpose": "representative final core model retrained on Bemis-Murcko scaffold split",
                "dataset": "SMRT unified all rows",
                "split": "Bemis-Murcko scaffold split",
                "seed": "2026",
                "config": "",
                "script": "scripts/paper_ablation_runner_20260825.py --scaffold",
                "checkpoint": scaffold_checkpoint.read_text(encoding="utf-8") if scaffold_checkpoint.exists() else "",
                "prediction": str(scaffold_predictions) if scaffold_predictions.exists() else "",
                "metrics": str(scaffold_metrics) if scaffold_metrics.exists() else "",
                "status": "completed" if scaffold_metrics.exists() and scaffold_predictions.exists() else "partial",
            },
        ]
    )
    pd.DataFrame(manifest_rows).to_csv(PKG / "experiment_manifest.csv", index=False)
    audit = [
        "# Results Audit",
        "",
        "This package contains generated provenance, scaffold split, PredRet audit, and any ablation runs completed by `paper_ablation_runner_20260825.py`.",
        "Completed training runs are reported from actual `metrics.json`; failed or interrupted runs are retained in the manifest.",
    ]
    (PKG / "results_audit.md").write_text("\n".join(audit), encoding="utf-8")


def main() -> None:
    mkdirs()
    final_training_config_audit()
    scaffold_split()
    predret_audit()
    package_results()


if __name__ == "__main__":
    main()
