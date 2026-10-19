"""Prepare PredRet Excel data for the current SMILES-Morgan model.

The model dataset expects CSV files with at least:
sample_id, smiles, rt, split

This script reads the PredRet workbook, converts InChI to canonical SMILES,
normalizes RT from minutes to seconds, and writes reproducible train/val/test
splits. Extra columns such as system and compound name are preserved for
analysis but ignored by the current model loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "external_predret" / "source" / "predret_all.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external_predret"

TRANSFER_SYSTEMS = [
    "FEM_Lipids",
    "FEM_long",
    "FEM_short",
    "LIFE_new",
    "LIFE_old",
    "Eawag_XBridgeC18",
    "IPB_Halle",
    "UniToyama_Atlantis",
    "MTBLS87",
    "Cao_HILIC",
]

SYSTEM_ALIASES = {
    "FEM_lipids": "FEM_Lipids",
    "FEM_Lipids": "FEM_Lipids",
    "UniToyama": "UniToyama_Atlantis",
    "UniToyama_Atlantis": "UniToyama_Atlantis",
}


def _import_pandas():
    try:
        import pandas as pd

        return pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pandas/openpyxl are required to read the Excel file. "
            "Run with the bundled Codex Python or an environment that has pandas."
        ) from exc


def excel_to_raw_csv(input_xlsx: Path, raw_csv: Path) -> None:
    pd = _import_pandas()
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(input_xlsx, sheet_name=0)

    rename = {
        "Unnamed: 0": "source_row",
        "Name": "compound_name",
        "System": "system",
        "Username": "username",
        "RT": "rt_min",
        "Pubchem": "pubchem",
        "InChI": "inchi",
        "Time": "time",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = {"compound_name", "system", "rt_min", "inchi"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Input workbook is missing required columns: {sorted(missing)}")

    keep = [c for c in ["source_row", "compound_name", "system", "username", "rt_min", "pubchem", "inchi", "time"] if c in df.columns]
    df = df[keep].copy()
    df["rt_min"] = pd.to_numeric(df["rt_min"], errors="coerce")
    df = df.dropna(subset=["system", "rt_min", "inchi"])
    df["system"] = df["system"].astype(str).str.strip()
    df["compound_name"] = df["compound_name"].fillna("").astype(str).str.strip()
    df["inchi"] = df["inchi"].astype(str).str.strip()
    df = df[(df["system"] != "") & (df["inchi"] != "") & (df["rt_min"] >= 0)]
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")


def canonicalize_system(system: str) -> str:
    return SYSTEM_ALIASES.get(system, system)


def inchi_to_smiles(inchi: str) -> str:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "RDKit is required for InChI to SMILES conversion. "
            "Run this script with a Python environment that has rdkit."
        ) from exc

    mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    n = len(indexed)
    n_test = round(n * 0.10)
    n_val = round(n * 0.09)
    test_idx = {idx for idx, _ in indexed[:n_test]}
    val_idx = {idx for idx, _ in indexed[n_test : n_test + n_val]}

    out = []
    counters = Counter()
    for idx, row in enumerate(rows):
        split = "test" if idx in test_idx else "val" if idx in val_idx else "train"
        counters[split] += 1
        new = dict(row)
        new["split"] = split
        new["sample_id"] = f"predret_{split}_{counters[split]:06d}"
        out.append(new)
    return out


def write_split_bundle(rows: list[dict[str, str]], output_dir: Path, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = assign_splits(rows, seed)
    fields = [
        "sample_id",
        "smiles",
        "rt",
        "split",
        "system",
        "compound_name",
        "inchi",
        "rt_min",
        "source_row",
        "pubchem",
        "username",
        "time",
    ]

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    for split in ["train", "val", "test"]:
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(by_split.get(split, []))

    with (output_dir / "all.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for split in ["train", "val", "test"]:
        split_rows = by_split.get(split, [])
        rts = [float(r["rt"]) for r in split_rows]
        summary_rows.append(
            {
                "split": split,
                "samples": len(split_rows),
                "systems": len({r["system"] for r in split_rows}),
                "rt_min_sec": min(rts) if rts else "",
                "rt_max_sec": max(rts) if rts else "",
            }
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "samples", "systems", "rt_min_sec", "rt_max_sec"])
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "output_dir": str(output_dir),
        "rows": len(rows),
        "splits": {split: len(by_split.get(split, [])) for split in ["train", "val", "test"]},
        "systems": len({r["system"] for r in rows}),
    }


def convert_raw_csv(raw_csv: Path, output_root: Path, seed: int, dedupe: bool) -> None:
    converted = []
    dropped = []
    seen = set()
    smiles_cache: dict[str, str] = {}

    with raw_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inchi = row.get("inchi", "").strip()
            smiles = smiles_cache.get(inchi)
            if smiles is None:
                smiles = inchi_to_smiles(inchi)
                smiles_cache[inchi] = smiles
            if not smiles:
                row["drop_reason"] = "inchi_to_smiles_failed"
                dropped.append(row)
                continue

            system = canonicalize_system(row["system"].strip())
            try:
                rt_min = float(row["rt_min"])
            except ValueError:
                row["drop_reason"] = "bad_rt"
                dropped.append(row)
                continue

            key = (system, smiles, round(rt_min, 8))
            if dedupe and key in seen:
                row["drop_reason"] = "duplicate_system_smiles_rt"
                dropped.append(row)
                continue
            seen.add(key)

            row["system"] = system
            row["smiles"] = smiles
            row["rt_min"] = f"{rt_min:.8g}"
            row["rt"] = f"{rt_min * 60.0:.8g}"
            converted.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "dropped_rows.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["drop_reason", "source_row", "compound_name", "system", "rt_min", "inchi", "pubchem", "username", "time"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dropped)

    all_summary = write_split_bundle(converted, output_root / "all_systems", seed)
    transfer_set = set(TRANSFER_SYSTEMS)
    transfer_rows = [r for r in converted if r["system"] in transfer_set]
    transfer_summary = write_split_bundle(transfer_rows, output_root / "transfer_10_systems", seed)

    system_counts = Counter(r["system"] for r in converted)
    transfer_counts = Counter(r["system"] for r in transfer_rows)
    summary = {
        "raw_csv": str(raw_csv),
        "rt_unit": "seconds; computed from PredRet RT minutes * 60",
        "split_strategy": "paper-style random split: 10% test, 9% validation, 81% train",
        "seed": seed,
        "dedupe": dedupe,
        "converted_rows": len(converted),
        "dropped_rows": len(dropped),
        "unique_smiles": len({r["smiles"] for r in converted}),
        "all_systems": all_summary,
        "transfer_10_systems": transfer_summary,
        "transfer_10_system_counts": dict(sorted(transfer_counts.items())),
        "top_system_counts": dict(system_counts.most_common(30)),
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PredRet data for model training.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stage",
        choices=["excel-to-raw", "convert", "all"],
        default="all",
        help="Use excel-to-raw with pandas env, convert with RDKit env, or all if one env has both.",
    )
    args = parser.parse_args()

    raw_csv = args.raw_csv or args.output_dir / "source" / "predret_raw_normalized.csv"
    if args.stage in {"excel-to-raw", "all"}:
        excel_to_raw_csv(args.input_xlsx, raw_csv)
        print(f"Wrote raw normalized CSV: {raw_csv}")
    if args.stage in {"convert", "all"}:
        convert_raw_csv(raw_csv, args.output_dir, args.seed, args.dedupe)


if __name__ == "__main__":
    main()
