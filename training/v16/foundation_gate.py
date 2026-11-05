"""Guard v16 against fine-tuning on an untrained (step-zero) base.

Reads checkpoint metadata; never modifies checkpoints or runs a model.
It does not establish that a model has learned language or reasoning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def inspect_checkpoint(path):
    ck = torch.load(str(path), map_location='cpu', weights_only=False)  # trusted self-created checkpoints only
    if not isinstance(ck, dict):
        raise ValueError(f'{path}: checkpoint is not a dictionary')
    step = ck.get('step')
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError(f'{path}: missing integer step metadata')
    return {'path':str(path), 'step':step, 'mode':ck.get('mode'),
            'best_val':ck.get('best_val'), 'model_config':ck.get('model_config'),
            'tokenizer_sha256':ck.get('tokenizer_sha256'),
            'has_optimizer': 'optimizer' in ck,
            'valid_trained_step':step>0}


def check(paths, min_steps=1, require_pretrain=True):
    found=[]
    for path in paths:
        if Path(path).is_file():
            item=inspect_checkpoint(path)
            item['passes_gate']=(item['step']>=min_steps and
                                (item['mode']=='pretrain' if require_pretrain else True))
            found.append(item)
    valid=[x for x in found if x['passes_gate']]
    valid.sort(key=lambda x:x['step'],reverse=True)
    return {'checked':found, 'candidate':valid[0] if valid else None,
            'warning':None if valid else 'No eligible trained checkpoint found. Do not initialize new SFT from a baseline step-0 checkpoint.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--min-steps', type=int, default=1)
    p.add_argument('--allow-sft', action='store_true', help='Allow SFT as a candidate (not evidence of adequate pretraining)')
    p.add_argument('--report', default='reports/v16_foundation_gate.json')
    a=p.parse_args()
    if a.min_steps<1: p.error('min-steps must be positive')
    result=check(a.candidates,a.min_steps,not a.allow_sft)
    path=Path(a.report); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
    print(json.dumps(result,indent=2,default=str))
    if not result['candidate']:
        raise SystemExit(2)

if __name__=='__main__': main()
