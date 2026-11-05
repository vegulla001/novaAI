#!/usr/bin/env python3
"""Raw continuation evaluation across held-out names and multiple seeds."""
import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig
from training.v16.evaluate_character import SCENES, HEADER


def count(text,name):
    return len(re.findall(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text,flags=re.I))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,137,2026])
    p.add_argument('--tokens',type=int,default=180)
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--top-k',type=int,default=30)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--output',default='reports/v16_character_dynamic_comparison.json')
    a=p.parse_args()
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    tok=Tokenizer.from_file(a.tokenizer)
    results=[]
    for cp in a.checkpoints:
        c=torch.load(cp,map_location='cpu',weights_only=False)
        model=GPT(GPTConfig(**c['model_config'])).to(device)
        model.load_state_dict(c['model'],strict=True);model.eval()
        for title,scene,names,setting in SCENES:
            prompt=HEADER+f'Characters to preserve: {", ".join(names)}\n\nStory so far:\n{scene}\n\nContinuation:\n'
            xids=tok.encode(prompt).ids
            if len(xids)+a.tokens > c['model_config']['block_size']:
                raise ValueError('Input plus generation exceeds block-size')
            x=torch.tensor([xids],dtype=torch.long,device=device)
            for seed in a.seeds:
                torch.manual_seed(seed)
                if device=='cuda':torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    output=model.generate(x,max_new_tokens=a.tokens,temperature=a.temperature,top_k=a.top_k)
                completion=tok.decode(output[0,len(xids):].tolist(),skip_special_tokens=True)
                mentions={name:count(completion,name) for name in names}
                r={'checkpoint':cp,'scene':title,'seed':seed,'model_only_mentions':mentions,
                   'both_characters_present':all(mentions.values()),
                   'setting_mentioned':setting.lower() in completion.lower(),
                   'json_contamination':bool(re.search(r'"(?:observed_next_text|context_evidence)"\s*:',completion)),
                   'text':completion}
                results.append(r)
                print(f'\n=== {Path(cp).parent.name} | {title} | seed={seed} ===\n{completion}\n')
                print('MODEL_ONLY_MENTIONS',mentions,'BOTH',r['both_characters_present'],'SETTING',r['setting_mentioned'])
        subset=[r for r in results if r['checkpoint']==cp]
        print('SUMMARY',cp, json.dumps({'cases':len(subset),
             'both_rate':sum(r['both_characters_present'] for r in subset)/len(subset),
             'setting_rate':sum(r['setting_mentioned'] for r in subset)/len(subset)},indent=2))
        del model,c
        if device=='cuda':torch.cuda.empty_cache()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print('SAVED',out)

if __name__=='__main__':main()
