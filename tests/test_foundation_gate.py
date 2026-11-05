import tempfile, unittest
from pathlib import Path
import torch
from training.v16.foundation_gate import check

class FoundationGateTests(unittest.TestCase):
    def test_step0_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pretrain_best.pt'
            torch.save({'mode':'pretrain','step':0,'model_config':{}},path)
            result=check([path]); self.assertIsNone(result['candidate'])
    def test_trained_latest_wins(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for name,step in [('pretrain_best.pt',0),('pretrain_latest.pt',100)]:
                path=Path(d)/name; torch.save({'mode':'pretrain','step':step},path); paths.append(path)
            result=check(paths)
            self.assertEqual(result['candidate']['step'],100)
    def test_sft_not_mistaken_for_pretraining(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sft_best.pt'; torch.save({'mode':'sft','step':500},path)
            self.assertIsNone(check([path])['candidate'])
            self.assertEqual(check([path],require_pretrain=False)['candidate']['step'],500)

if __name__=='__main__': unittest.main()
