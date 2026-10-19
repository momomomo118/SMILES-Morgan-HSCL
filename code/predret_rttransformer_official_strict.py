"""Official RT-Transformer strict PredRet transfer entry point.

This script is intentionally separate from predret_rttr_full_transfer.py.  It is
for Xue et al. RT-Transformer only: molecular graph + 2048D fingerprint + MyNet.
It does not use RoBERTa, ChemBERTa, or LSTM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "external_literature_code" / "rt_transformer"


def import_official_model():
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from model import MyNet
    finally:
        try:
            sys.path.remove(str(OFFICIAL_ROOT))
        except ValueError:
            pass
    return MyNet


def unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise TypeError("Checkpoint is not a recognized RT-Transformer state_dict payload.")


def compatibility_check(pretrained_checkpoint: Path, feat_dim: int, emb_dim: int) -> dict[str, object]:
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(pretrained_checkpoint)
    MyNet = import_official_model()
    model = MyNet(emb_dim=emb_dim, feat_dim=feat_dim)
    expected = model.state_dict()
    payload = torch.load(pretrained_checkpoint, map_location="cpu")
    state = unwrap_state_dict(payload)

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = []
    for key in sorted(set(expected).intersection(state)):
        got = tuple(state[key].shape)
        exp = tuple(expected[key].shape)
        if got != exp:
            shape_mismatch.append({"key": key, "expected": exp, "checkpoint": got})

    compatible = not missing and not unexpected and not shape_mismatch
    return {
        "checkpoint": str(pretrained_checkpoint),
        "official_root": str(OFFICIAL_ROOT),
        "architecture": "Xue et al. RT-Transformer MyNet graph + 2048D fingerprint",
        "expected_key_count": len(expected),
        "checkpoint_key_count": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "compatible": compatible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True, help="Official best_state_dict.pth checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("data/external_predret/transfer_10_systems_cleaned_strict_10fold/by_system_10fold"))
    parser.add_argument("--output-dir", type=Path, default=Path("server_results/predret_compare_strict_20260909/rttransformer_official"))
    parser.add_argument("--compatibility-only", action="store_true")
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=512)
    args = parser.parse_args()

    report = compatibility_check(args.pretrained_checkpoint, feat_dim=args.feat_dim, emb_dim=args.emb_dim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_compatibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["compatible"]:
        raise RuntimeError("Official RT-Transformer checkpoint is not architecture-compatible.")
    if args.compatibility_only:
        return

    raise NotImplementedError(
        "Official strict transfer training is intentionally not launched yet. "
        "Next step is adding the PredRet graph/fingerprint Dataset adapter and inner-validation training loop."
    )


if __name__ == "__main__":
    main()
