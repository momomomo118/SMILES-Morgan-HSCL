"""Generate label-free GIN inputs with the descriptor version matching archives."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/generalization_runtime_rdkit202603_py310'),str(ROOT/'code')]
import torch,pandas as pd
from rdkit import rdBase,Chem,RDLogger
from train_pure_gnn_for_ensemble import smiles_to_graph
RDLogger.DisableLog('rdApp.*')

def main():
    assert rdBase.rdkitVersion=='2026.03.3',rdBase.rdkitVersion
    source=Path(sys.argv[1]);dest=Path(sys.argv[2]);d=pd.read_csv(source)
    assert list(d.columns)==['sample_id','smiles'],'Labels are prohibited in feature input'
    items=[]
    for i,r in d.iterrows():
        assert Chem.MolFromSmiles(r.smiles) is not None
        x,a,desc=smiles_to_graph(r.smiles)
        items.append(dict(sample_id=r.sample_id,smiles=r.smiles,x=torch.tensor(x),adj=torch.tensor(a),desc=torch.tensor(desc),y=torch.tensor(0.)))
    torch.save(items,dest)
    dest.with_suffix('.json').write_text(json.dumps(dict(rdkit=rdBase.rdkitVersion,n=len(items),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),labels_used=False,
        features_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
    print('GIN features generated',rdBase.rdkitVersion,len(items),flush=True)

if __name__=='__main__':main()
