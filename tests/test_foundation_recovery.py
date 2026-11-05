import json
import tempfile
import unittest
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.v16.foundation_recovery import inspect, command, make_parser

class FoundationRecoveryTests(unittest.TestCase):
    def test_inspect_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/'data'; data.mkdir()
            (data/'tokenizer.json').write_text('test', encoding='utf-8')
            digest=hashlib.sha256(b'test').hexdigest()
            (data/'build_report.json').write_text(json.dumps({'vocab_size':32,'tokenizer_sha256':digest}))
            np.save(data/'train.npy',np.arange(4000,dtype=np.uint16)%32)
            np.save(data/'val.npy',np.arange(2500,dtype=np.uint16)%32)
            ck=root/'ck.pt'
            torch.save({'model':{'test': torch.ones(1)},'mode':'pretrain','step':2000,'model_config':{'vocab_size':32,'block_size':1024},'tokenizer_sha256':digest},ck)
            r=inspect(ck,data)
            self.assertEqual(r['train']['count'],4000)
            self.assertEqual(r['checkpoint_step'],2000)
            a=make_parser().parse_args(['--checkpoint',str(ck),'--data-dir',str(data)])
            cmd=command(a,r['config'])
            self.assertIn('--restart-from',cmd)
            self.assertNotIn('--resume',cmd)
            self.assertEqual(cmd[cmd.index('--lr')+1],str(8e-5))
    def test_wrong_tokenizer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data';data.mkdir()
            (data/'tokenizer.json').write_text('test')
            (data/'build_report.json').write_text(json.dumps({'vocab_size':32,'tokenizer_sha256':'BAD'}))
            ck=root/'ck.pt'
            torch.save({'model':{},'mode':'pretrain','step':2000,'model_config':{'vocab_size':32,'block_size':1024},'tokenizer_sha256':'BAD'},ck)
            with self.assertRaises(ValueError): inspect(ck,data)

if __name__=='__main__':unittest.main()
