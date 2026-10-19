"""Build standardized PredRet chromatography-condition metadata.

The available PredRet transfer table contains reliable system labels but not
full column/mobile-phase/gradient fields. This script still creates the target
schema now: known or inferred fields are encoded, unavailable fields are kept
as standard columns with explicit missing masks, so downstream models can use
the same interface when richer metadata is added later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SYSTEMS = [
    "Eawag_XBridgeC18",
    "FEM_Lipids",
    "FEM_long",
    "FEM_short",
    "IPB_Halle",
    "LIFE_new",
    "LIFE_old",
    "UniToyama_Atlantis",
    "Cao_HILIC",
    "MTBLS87",
]

SYSTEM_ALIASES = {
    "FEM_lipids": "FEM_Lipids",
    "FEM_Lipids": "FEM_Lipids",
    "UniToyama": "UniToyama_Atlantis",
    "UniToyama_Atlantis": "UniToyama_Atlantis",
}

MISSING_NUMERIC_FIELDS = [
    "column_length_mm",
    "column_inner_diameter_mm",
    "particle_um",
    "pore_angstrom",
    "temperature_c",
    "flow_ml_min",
    "gradient_initial_b_pct",
    "gradient_final_b_pct",
    "gradient_time_min",
    "mobile_phase_pH",
    "buffer_mM",
]


def alias_system(value: object) -> str:
    text = str(value).strip()
    return SYSTEM_ALIASES.get(text, text)


def source_family(system: str) -> str:
    if system.startswith("FEM"):
        return "FEM"
    if system.startswith("LIFE"):
        return "LIFE"
    if system.startswith("Eawag"):
        return "Eawag"
    if system.startswith("UniToyama"):
        return "UniToyama"
    if system.startswith("Cao"):
        return "Cao"
    return system.split("_", 1)[0]


def chromatographic_mode(system: str) -> str:
    if "HILIC" in system.upper():
        return "HILIC"
    return "RP"


def column_family(system: str) -> str:
    text = system.upper()
    if "XBRIDGEC18" in text or "ATLANTIS" in text:
        return "C18"
    if "HILIC" in text:
        return "HILIC"
    return "unknown"


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - float(values.mean())) / std


def add_one_hot(df: pd.DataFrame, column: str, prefix: str, values: list[str]) -> list[str]:
    cols = []
    for value in values:
        safe = value.replace("-", "_").replace("/", "_").replace(" ", "_")
        col = f"cond_{prefix}_{safe}"
        df[col] = (df[column] == value).astype(float)
        df[f"mask_{prefix}_{safe}"] = 1.0
        cols.append(col)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/external_predret/transfer_10_systems_cleaned/all_cleaned.csv")
    parser.add_argument("--output-dir", default="data/external_predret/condition_metadata")
    parser.add_argument("--systems", nargs="*", default=TARGET_SYSTEMS)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(inp, low_memory=False)
    rows["system"] = rows["system"].map(alias_system)
    rows = rows[rows["system"].isin(args.systems)].copy()
    rows["rt"] = pd.to_numeric(rows["rt"], errors="coerce")

    summary = (
        rows.groupby("system", sort=False)
        .agg(
            n_rows=("rt", "size"),
            rt_min_sec=("rt", "min"),
            rt_max_sec=("rt", "max"),
            rt_mean_sec=("rt", "mean"),
            rt_std_sec=("rt", "std"),
            duplicate_merge_rate=("n_merged", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(1) > 1).mean())),
        )
        .reset_index()
    )
    summary["system"] = pd.Categorical(summary["system"], categories=args.systems, ordered=True)
    summary = summary.sort_values("system").reset_index(drop=True)
    summary["system_id"] = np.arange(len(summary), dtype=int)
    summary["system_id_str"] = summary["system"].astype(str)
    summary["chromatographic_mode"] = summary["system"].map(chromatographic_mode)
    summary["column_family"] = summary["system"].map(column_family)
    summary["source_family"] = summary["system"].map(source_family)
    summary["metadata_source"] = "system_label_inferred"
    summary["metadata_confidence"] = 0.5

    value_cols = []
    value_cols.extend(add_one_hot(summary, "system_id_str", "system", list(summary["system_id_str"])))
    value_cols.extend(add_one_hot(summary, "chromatographic_mode", "mode", sorted(summary["chromatographic_mode"].unique())))
    value_cols.extend(add_one_hot(summary, "column_family", "column_family", sorted(summary["column_family"].unique())))
    value_cols.extend(add_one_hot(summary, "source_family", "source_family", sorted(summary["source_family"].unique())))

    numeric_sources = {
        "n_rows_log": np.log1p(summary["n_rows"].astype(float)),
        "rt_mean_sec_z": zscore(summary["rt_mean_sec"]),
        "rt_std_sec_z": zscore(summary["rt_std_sec"].fillna(0.0)),
        "rt_range_sec_z": zscore(summary["rt_max_sec"] - summary["rt_min_sec"]),
        "duplicate_merge_rate": summary["duplicate_merge_rate"].astype(float),
        "metadata_confidence": summary["metadata_confidence"].astype(float),
    }
    for name, values in numeric_sources.items():
        col = f"cond_{name}"
        summary[col] = values.astype(float)
        summary[f"mask_{name}"] = 1.0
        value_cols.append(col)

    for field in MISSING_NUMERIC_FIELDS:
        col = f"cond_{field}"
        summary[col] = 0.0
        summary[f"mask_{field}"] = 0.0
        value_cols.append(col)

    mask_cols = ["mask_" + col[len("cond_") :] for col in value_cols]
    ordered = [
        "system",
        "system_id",
        "system_id_str",
        "chromatographic_mode",
        "column_family",
        "source_family",
        "metadata_source",
        "metadata_confidence",
        "n_rows",
        "rt_min_sec",
        "rt_max_sec",
        "rt_mean_sec",
        "rt_std_sec",
        "duplicate_merge_rate",
        *sorted(value_cols),
        *sorted(mask_cols),
    ]
    system_conditions = summary[ordered].copy()
    system_conditions.to_csv(out / "system_conditions.csv", index=False)

    row_conditions = rows.merge(system_conditions, on="system", how="left", validate="many_to_one")
    row_conditions.to_csv(out / "row_conditions.csv", index=False)

    schema = {
        "input": str(inp),
        "system_key": "system",
        "rt_unit": "seconds",
        "value_columns": sorted(value_cols),
        "mask_columns": sorted(mask_cols),
        "missing_numeric_fields": MISSING_NUMERIC_FIELDS,
        "notes": [
            "PredRet transfer CSV exposes system labels but not full LC method metadata.",
            "Column family and chromatographic mode are inferred from system names where possible.",
            "Unavailable LC method fields are retained as cond_* zeros with mask_* = 0.",
            "Condition dropout and learned null embeddings are handled by the training model.",
        ],
    }
    (out / "condition_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Wrote {out / 'system_conditions.csv'}")
    print(f"Wrote {out / 'row_conditions.csv'}")
    print(f"condition_dim={len(value_cols)}")


if __name__ == "__main__":
    main()
