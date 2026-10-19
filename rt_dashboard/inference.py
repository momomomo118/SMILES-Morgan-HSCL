from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from sklearn.preprocessing import PowerTransformer
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "code" / "src"
CODE_ROOT = ROOT / "code"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train_multimodal import build_model  # noqa: E402
from train_pure_gnn_for_ensemble import PureGIN, collate, smiles_to_graph  # noqa: E402


RDLogger.DisableLog("rdApp.*")


BASE_MODELS = {
    "target_power_exact_repro_seed1234": (
        ROOT / "checkpoints" / "supcr_tuning_aliases" / "exact1234_best.pt",
        0.30,
    ),
    "target_power_div_dropout015_seed777": (
        ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_dropout015_seed777" / "best_cross_attention_residual.pt",
        0.25,
    ),
    "target_power_div_lowfusion_seed777": (
        ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_lowfusion_seed777" / "best_cross_attention_residual.pt",
        0.10,
    ),
    "target_power_div_aug050_seed777": (
        ROOT / "checkpoints" / "supcr_tuning" / "target_power_div_aug050_seed777" / "best_cross_attention_residual.pt",
        0.20,
    ),
    "target_power_exact_repro_seed2025": (
        ROOT / "checkpoints" / "supcr_tuning_aliases" / "exact2025_best.pt",
        0.15,
    ),
}

FINAL_WEIGHTS = {
    "pred_ccrl_ensemble": 0.30,
    "gnn2026": 0.24,
    "gnn3407": 0.18,
    "gnn777": 0.28,
}


def descriptor_row(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无法解析该 SMILES，请检查括号、环编号和元素符号。")
    return {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "smiles_len": float(len(smiles)),
    }


class ModelService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._loaded = False
        self._loading = False
        self._load_error: str | None = None
        self._base_models: dict[str, torch.nn.Module] = {}
        self._gnn_models: dict[str, torch.nn.Module] = {}
        self._tokenizer = None
        self._target_transformer: PowerTransformer | None = None
        self._reference_fps: list = []
        self._reference_abs_errors = np.array([], dtype=float)
        self._started_at = time.time()

    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "loading": self._loading,
            "error": self._load_error,
            "device": "GPU" if self.device == "cuda" else "CPU",
            "model": "SMILES–Morgan HSCL + PureGIN ensemble",
            "dataset": "SMRT",
        }

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loading = True
            self._load_error = None
            try:
                train_rt = pd.read_csv(ROOT / "data" / "unified" / "train.csv")["rt"].astype(float).to_numpy().reshape(-1, 1)
                self._target_transformer = PowerTransformer(method="yeo-johnson", standardize=True).fit(train_rt)
                model_dir = str((ROOT / "pretrained" / "ChemBERTa-zinc-base-v1").resolve())
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

                for name, (path, _) in BASE_MODELS.items():
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    args_dict = dict(checkpoint["args"])
                    args_dict["smiles_model"] = model_dir
                    args = SimpleNamespace(**args_dict)
                    model = build_model(args)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval().to(self.device)
                    self._base_models[name] = model

                in_dim = len(smiles_to_graph("CC")[0][0])
                for seed in (2026, 3407, 777):
                    path = ROOT / "checkpoints" / "gnn" / f"pure_gin_seed{seed}" / "best_pure_gin.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    model = PureGIN(in_dim=in_dim).to(self.device)
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                    self._gnn_models[f"gnn{seed}"] = model

                self._load_reference_errors()
                self._loaded = True
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._loading = False

    def _load_reference_errors(self) -> None:
        base_dir = ROOT / "outputs" / "final_fusion_ccrl_gnn"
        reference = pd.read_csv(base_dir / "predictions_test_final_ccrl_gnn_ptdf.csv")
        pred_col = "pred_final_ccrl_gnn_ptdf"
        self._reference_abs_errors = (reference[pred_col] - reference["true_rt"]).abs().to_numpy(dtype=float)
        self._reference_fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(s)), 2, nBits=2048, useChirality=True)
            for s in reference["smiles"]
        ]

    def _morgan_tensor(self, mol: Chem.Mol) -> torch.Tensor:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
        return torch.tensor(list(fp), dtype=torch.float32).unsqueeze(0).to(self.device)

    def _predict_ccrl_ensemble(self, smiles: str, mol: Chem.Mol) -> tuple[float, dict[str, float]]:
        encoded = self._tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        morgan = self._morgan_tensor(mol)
        predictions: dict[str, float] = {}
        with torch.inference_mode():
            for name, model in self._base_models.items():
                raw = model(input_ids=input_ids, attention_mask=attention_mask, morgan_fp=morgan).prediction.item()
                actual = float(self._target_transformer.inverse_transform([[raw]])[0, 0])
                predictions[name] = actual

        names = list(BASE_MODELS)
        weights = np.array([BASE_MODELS[name][1] for name in names], dtype=float)
        return float(np.array([predictions[name] for name in names]) @ weights), predictions

    def _predict_gnns(self, smiles: str) -> dict[str, float]:
        x, adj, desc = smiles_to_graph(smiles)
        item = {
            "sample_id": "query",
            "smiles": smiles,
            "x": torch.tensor(x, dtype=torch.float32),
            "adj": torch.tensor(adj, dtype=torch.float32),
            "desc": torch.tensor(desc, dtype=torch.float32),
            "y": torch.tensor(0.0, dtype=torch.float32),
        }
        batch = collate([item])
        x_t = batch["x"].to(self.device)
        adj_t = batch["adj"].to(self.device)
        mask_t = batch["mask"].to(self.device)
        desc_t = batch["desc"].to(self.device)
        output = {}
        with torch.inference_mode():
            for name, model in self._gnn_models.items():
                output[name] = float(model(x_t, adj_t, mask_t, desc_t).item())
        return output

    def _estimate_error(self, smiles: str, components: dict[str, float], prediction: float) -> tuple[float, float, str]:
        values = np.array(list(components.values()), dtype=float)
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._reference_fps), dtype=float)
        top = np.argsort(similarities)[-12:]
        sim = similarities[top]
        weights = np.maximum(sim, 0.05) ** 4
        neighbor_est = float(np.average(self._reference_abs_errors[top], weights=weights))
        nearest = float(sim.max())

        disagreement = float(values.std())
        estimate = float(np.clip(0.75 * neighbor_est + 0.25 * max(13.6490, disagreement), 8.0, 140.0))
        if nearest >= 0.75:
            level = "高"
        elif nearest >= 0.50:
            level = "中"
        else:
            level = "较低"
            estimate = min(140.0, estimate * 1.25)
        return estimate, nearest, level

    @staticmethod
    def _molecule_png(mol: Chem.Mol) -> str:
        image = Draw.MolToImage(mol, size=(480, 300), kekulize=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, smiles: str) -> dict:
        smiles = "".join(str(smiles).strip().split())
        if not smiles:
            raise ValueError("请输入 SMILES 序列。")
        if len(smiles) > 512:
            raise ValueError("SMILES 长度超过 512 个字符，暂不支持。")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无法解析该 SMILES，请检查括号、环编号和元素符号。")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        self._load()
        started = time.perf_counter()
        pred_ccrl_ensemble, base_members = self._predict_ccrl_ensemble(canonical, mol)
        gnn = self._predict_gnns(canonical)
        components = {"pred_ccrl_ensemble": pred_ccrl_ensemble, **gnn}
        prediction = float(sum(FINAL_WEIGHTS[name] * value for name, value in components.items()))
        estimated_error, nearest_similarity, confidence = self._estimate_error(canonical, components, prediction)
        interval_half = max(45.0, estimated_error * 2.25)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        desc = descriptor_row(canonical)

        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "prediction_s": round(prediction, 2),
            "prediction_min": round(prediction / 60.0, 2),
            "estimated_abs_error_s": round(estimated_error, 2),
            "interval_90_s": [round(max(0.0, prediction - interval_half), 2), round(prediction + interval_half, 2)],
            "confidence": confidence,
            "nearest_similarity": round(nearest_similarity, 3),
            "model_disagreement_s": round(float(np.std(list(components.values()))), 2),
            "molecule": {
                "formula": formula,
                "molecular_weight": round(desc["mol_wt"], 2),
                "logp": round(desc["logp"], 2),
                "tpsa": round(desc["tpsa"], 2),
                "rings": int(desc["rings"]),
                "heavy_atoms": int(desc["heavy_atoms"]),
                "image": self._molecule_png(mol),
            },
            "components": {name: round(value, 2) for name, value in components.items()},
            "base_members": {name: round(value, 2) for name, value in base_members.items()},
            "latency_s": round(time.perf_counter() - started, 2),
            "note": "误差值为基于验证集相似分子和集成分歧估计的预计绝对误差；新分子没有实验真值，因此不是实测误差。",
        }


SERVICE = ModelService()


def metrics_payload() -> dict:
    path = ROOT / "outputs" / "final_fusion_ccrl_gnn" / "metrics_final_ccrl_gnn_ptdf.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["final_test_metrics"]
    return {
        "mae": round(m["mae"], 3),
        "rmse": round(m["rmse"], 3),
        "r2": round(m["r2"], 4),
        "medae": round(m["medae"], 3),
        "p95": round(m["p95_abs_error"], 3),
    }
