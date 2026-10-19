"""ML runtime and the explicitly documented concat-RSCL extension."""
import os
import sys
from dataclasses import fields
from pathlib import Path

from common import ROOT, read_json, run_id

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'code/src')]

import numpy as np
import torch
from transformers import AutoTokenizer
import paper_ablation_runner_20260825 as legacy


class ConcatRSCLModel(legacy.MultiModalRTModel):
    """Keep C's regression path; add a training-only projection for concat features.

    The legacy concat model had z_fusion=None. Simply setting supcr=True would
    silently omit fusion SupCR. This named extension computes both loss terms.
    Projection initialization is isolated from the historical global RNG stream.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fusion_kind != 'concat':
            raise ValueError('ConcatRSCLModel requires concat fusion')
        with torch.random.fork_rng(devices=[]):
            self.fusion_projection = legacy.ProjectionHead(
                self.smiles_encoder.output_dim+self.morgan_encoder.output_dim,
                dropout=kwargs.get('dropout', args[3] if len(args) > 3 else .1))

    def forward(self, input_ids, attention_mask, morgan_fp):
        out = super().forward(input_ids, attention_mask, morgan_fp)
        feature = torch.cat([out['smiles_pooled'], out['morgan_pooled']], dim=-1)
        out['fusion_pooled'] = feature
        out['z_fusion'] = self.fusion_projection(feature)
        return out


def config_for(variant, seed, device='cuda', source=None):
    raw = read_json(source) if source else {}
    allowed = {f.name for f in fields(legacy.RunConfig)}
    raw = {k:v for k,v in raw.items() if k in allowed}
    raw.update(run_id=run_id(variant, seed), configuration=variant, seed=seed,
               fusion='concat' if variant in ('base', 'rscl') else 'cross_attention_residual',
               supcr=variant in ('rscl', 'full'), device=device,
               train_data=str(ROOT/'data/unified/train.csv'),
               val_data=str(ROOT/'data/unified/val.csv'),
               test_data=str(ROOT/'data/unified/test.csv'))
    if variant == 'smiles':
        raw.update(morgan=False, fusion='none', supcr=False)
    return legacy.RunConfig(**raw)


def build_model(cfg):
    if cfg.fusion == 'concat' and cfg.supcr:
        # Temporarily substitute the class during construction, then restore it.
        original = legacy.MultiModalRTModel
        legacy.MultiModalRTModel = ConcatRSCLModel
        try:
            return legacy.build_model(cfg)
        finally:
            legacy.MultiModalRTModel = original
    return legacy.build_model(cfg)


def resources(cfg):
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'pretrained/ChemBERTa-zinc-base-v1'), local_files_only=True)
    transformer = legacy.build_target_transformer(cfg)
    loaders = legacy.build_loaders(cfg, tokenizer, transformer)
    return transformer, loaders


def torch_load(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def capture_rng(dataset):
    from rdkit import rdBase
    return dict(python=__import__('random').getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                dataset=dataset._smiles_rng.getstate())


def restore_rng(state, dataset):
    import random
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    dataset._smiles_rng.setstate(state['dataset'])


def environment():
    import platform, pandas, sklearn, transformers
    from rdkit import rdBase
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__,
                transformers=transformers.__version__, rdkit=rdBase.rdkitVersion,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
