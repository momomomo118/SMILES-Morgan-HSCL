import argparse, json, math, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H", "Si", "B"]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def onehot_idx(x, values):
    out = [0.0] * (len(values) + 1)
    try:
        out[values.index(x)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out

def atom_features(atom):
    return np.array(
        onehot_idx(atom.GetSymbol(), ATOM_TYPES)
        + [
            atom.GetAtomicNum() / 100.0,
            atom.GetTotalDegree() / 8.0,
            atom.GetFormalCharge() / 5.0,
            atom.GetTotalNumHs() / 8.0,
            atom.GetTotalValence() / 8.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() / 200.0,
        ],
        dtype=np.float32,
    )

def desc_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(10, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol) / 800.0,
        Crippen.MolLogP(mol) / 10.0,
        rdMolDescriptors.CalcTPSA(mol) / 250.0,
        Lipinski.NumHDonors(mol) / 10.0,
        Lipinski.NumHAcceptors(mol) / 20.0,
        Lipinski.NumRotatableBonds(mol) / 20.0,
        rdMolDescriptors.CalcNumRings(mol) / 10.0,
        mol.GetNumHeavyAtoms() / 100.0,
        rdMolDescriptors.CalcFractionCSP3(mol),
        len(str(smiles)) / 200.0,
    ], dtype=np.float32)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        x = np.zeros((1, len(ATOM_TYPES) + 1 + 8), dtype=np.float32)
        adj = np.eye(1, dtype=np.float32)
        return x, adj, desc_features(smiles)

    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    x = np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0)
    adj = np.eye(n, dtype=np.float32)

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        weight = 1.0
        if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            weight = 1.2
        elif b.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            weight = 1.4
        elif b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            weight = 1.1
        adj[i, j] = weight
        adj[j, i] = weight

    deg = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(deg, 1e-6, None)
    return x, adj, desc_features(smiles)

class GraphRTDataset(Dataset):
    def __init__(self, csv_path, cache_path):
        self.csv_path = Path(csv_path)
        self.cache_path = Path(cache_path)
        self.df = pd.read_csv(self.csv_path)
        if self.cache_path.exists():
            self.items = torch.load(self.cache_path)
        else:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.items = []
            for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"graph cache {self.csv_path.name}"):
                x, adj, desc = smiles_to_graph(row["smiles"])
                self.items.append({
                    "sample_id": row["sample_id"],
                    "smiles": row["smiles"],
                    "x": torch.tensor(x, dtype=torch.float32),
                    "adj": torch.tensor(adj, dtype=torch.float32),
                    "desc": torch.tensor(desc, dtype=torch.float32),
                    "y": torch.tensor(float(row["rt"]), dtype=torch.float32),
                })
            torch.save(self.items, self.cache_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    max_n = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    xs, adjs, masks, descs, ys = [], [], [], [], []
    sample_ids, smiles = [], []

    for item in batch:
        n = item["x"].shape[0]
        x = torch.zeros(max_n, feat_dim)
        adj = torch.zeros(max_n, max_n)
        mask = torch.zeros(max_n)
        x[:n] = item["x"]
        adj[:n, :n] = item["adj"]
        mask[:n] = 1.0
        xs.append(x)
        adjs.append(adj)
        masks.append(mask)
        descs.append(item["desc"])
        ys.append(item["y"])
        sample_ids.append(item["sample_id"])
        smiles.append(item["smiles"])

    return {
        "x": torch.stack(xs),
        "adj": torch.stack(adjs),
        "mask": torch.stack(masks),
        "desc": torch.stack(descs),
        "y": torch.stack(ys),
        "sample_id": sample_ids,
        "smiles": smiles,
    }

class PureGIN(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=5, desc_dim=10, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.desc_proj = nn.Sequential(nn.Linear(desc_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, adj, mask, desc):
        h = self.in_proj(x)
        for mlp, norm in zip(self.layers, self.norms):
            m = torch.bmm(adj, h)
            h = norm(h + mlp(m))
            h = F.relu(h)
        mask_u = mask.unsqueeze(-1)
        mean_pool = (h * mask_u).sum(1) / mask_u.sum(1).clamp_min(1.0)
        max_pool = h.masked_fill(mask_u == 0, -1e9).max(1).values
        d = self.desc_proj(desc)
        return self.head(torch.cat([mean_pool, max_pool, d], dim=1)).squeeze(-1)

def metrics(y, p):
    abs_err = np.abs(p - y)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "medae": float(np.median(abs_err)),
        "mape": float(np.mean(abs_err / np.clip(np.abs(y), 1e-12, None)) * 100.0),
        "r2": float(r2_score(y, p)),
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0,
    }

def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="pure gnn train", leave=False):
        batch = move(batch, device)
        pred = model(batch["x"], batch["adj"], batch["mask"], batch["desc"])
        loss = F.smooth_l1_loss(pred, batch["y"], beta=20.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, ids, smiles = [], [], [], []
    for batch in tqdm(loader, desc="pure gnn eval", leave=False):
        raw = batch
        batch = move(batch, device)
        pred = model(batch["x"], batch["adj"], batch["mask"], batch["desc"])
        ys.append(batch["y"].detach().cpu())
        ps.append(pred.detach().cpu())
        ids.extend(raw["sample_id"])
        smiles.extend(raw["smiles"])
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    df = pd.DataFrame({"sample_id": ids, "smiles": smiles, "true_rt": y, "pred_rt": p})
    return metrics(y, p), df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/unified/train.csv")
    ap.add_argument("--val-data", default="data/unified/val.csv")
    ap.add_argument("--test-data", default="data/unified/test.csv")
    ap.add_argument("--output-dir", default="src/outputs/gnn/pure_gin_seed2026")
    ap.add_argument("--cache-dir", default="src/outputs/gnn/pure_gin_cache")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    train_ds = GraphRTDataset(args.train_data, cache / "train.pt")
    val_ds = GraphRTDataset(args.val_data, cache / "val.pt")
    test_ds = GraphRTDataset(args.test_data, cache / "test.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    in_dim = train_ds[0]["x"].shape[1]
    model = PureGIN(in_dim=in_dim, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20)

    best = math.inf
    bad = 0
    history = []
    best_path = out / "best_pure_gin.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device)
        sched.step()
        val_m, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"Epoch {epoch}/{args.epochs} train={train_loss:.4f} val_mae={val_m['mae']:.4f}", flush=True)

        if val_m["mae"] < best:
            best = val_m["mae"]
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_m, "args": vars(args)}, best_path)
        else:
            bad += 1

        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if bad >= args.patience:
            print(f"Early stopping best_val={best:.4f}", flush=True)
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_m, val_pred = evaluate(model, val_loader, device)
    test_m, test_pred = evaluate(model, test_loader, device)
    val_pred.to_csv(out / "predictions_val_gnn.csv", index=False)
    test_pred.to_csv(out / "predictions_test_gnn.csv", index=False)
    summary = {"best_epoch": int(ckpt["epoch"]), "val": val_m, "test": test_m, "best_checkpoint": str(best_path)}
    (out / "metrics_gnn.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
