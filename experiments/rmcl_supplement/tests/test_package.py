"""Synthetic contract checks; no dataset training or parameter updates."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import runtime as rt
from common import metrics


class TinyEncoder(torch.nn.Module):
    output_dim=8
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(8,8)
    def forward(self,input_ids=None,attention_mask=None,morgan_fp=None):
        x=morgan_fp if morgan_fp is not None else input_ids
        seq=self.layer(x.float()).unsqueeze(1)
        return rt.legacy.EncoderOutput(sequence=seq,pooled=seq[:,0],mask=torch.ones(len(x),1,dtype=torch.long))


class Contracts(unittest.TestCase):
    def test_concat_extension_preserves_regression(self):
        torch.manual_seed(19)
        base=rt.legacy.MultiModalRTModel(TinyEncoder(),TinyEncoder(),'concat',.1,8,2).eval()
        rng_after_base=torch.get_rng_state().clone()
        torch.manual_seed(19)
        extended=rt.ConcatRSCLModel(TinyEncoder(),TinyEncoder(),'concat',.1,8,2).eval()
        self.assertTrue(torch.equal(rng_after_base,torch.get_rng_state()))
        x=torch.arange(32,dtype=torch.float32).reshape(4,8)/32
        with torch.no_grad():
            a=base(x,None,x);b=extended(x,None,x)
            self.assertTrue(torch.equal(a['prediction'],b['prediction']))
            self.assertIsNone(a['z_fusion'])
            self.assertEqual(tuple(b['z_fusion'].shape),(4,256))
            cfg=rt.config_for('rscl',2026,'cpu')
            al=rt.legacy.CrossModalSupCRLoss(temperature=.07,similarity_weight=.05,use_log=False)
            fl=rt.legacy.FusionSupCRLoss(temperature=.07,use_log=False)
            loss,parts=rt.legacy.compute_loss(b,torch.zeros(4),torch.arange(4).float(),cfg,al,fl)
            self.assertTrue(torch.isfinite(loss))
            self.assertNotEqual(parts['fusion_supcr'],0.)

    def test_metric_contract(self):
        r=metrics([1.,2.,3.],[1.,3.,2.])
        self.assertAlmostEqual(r['MAE'],2/3)
        self.assertAlmostEqual(r['RMSE'],np.sqrt(2/3))
        self.assertEqual(r['MedAE'],1.)


if __name__=='__main__':
    from common import ROOT,write_json
    result=unittest.main(exit=False).result
    write_json(ROOT/'validation/synthetic_contract_checks.json',dict(
        status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        scope='Tiny synthetic encoders only; regression equivalence, RNG isolation, both RSCL losses, metric definitions',
        dataset_training=False,parameter_updates=False))
    sys.exit(0 if result.wasSuccessful() else 1)
