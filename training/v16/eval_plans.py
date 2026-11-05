#!/usr/bin/env python3
"""Evaluate structured plan generation, not just 24-token prefix matching."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from tokenizers import Tokenizer
from training.v15.model import GPT,GPTConfig
from training.v16.planning import prompt_for, validate_row


def generate(model, ids, device, max_new_tokens, block_size):
    # Explicit cap avoids silently dropping the grounding context.
    if len(ids)+max_new_tokens>block_size:
        raise ValueError(f'Prompt ({len(ids)}) + output budget ({max_new_tokens}) exceeds context ({block_size})')
    x=torch.tensor([ids],dtype=torch.long,device=device)
    with torch.no_grad():
        y=model.generate(x,max_new_tokens=max_new_tokens,top_k=1)
    return y[0].tolist()[len(ids):]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--source',required=True,help='Curated JSONL: evaluation stories only, not used for training')
    ap.add_argument('--data-dir',default='dataset/v15_prose_expanded')
    ap.add_argument('--max-cases',type=int,default=100)
    ap.add_argument('--max-new-tokens',type=int,default=256)
    ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    if args.max_cases<1 or args.max_new_tokens<1:ap.error('positive limits required')
    data=Path(args.data_dir)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False) # trusted local checkpoint only
    tok_path=data/'tokenizer.json'
    tok_hash=hashlib.sha256(tok_path.read_bytes()).hexdigest()
    if ckpt.get('tokenizer_sha256')!=tok_hash:raise ValueError('Checkpoint tokenizer hash mismatch')
    cfg=GPTConfig(**ckpt['model_config'])
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    model=GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    tok=Tokenizer.from_file(str(tok_path))
    cases=[]
    for line in Path(args.source).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        validate_row(row)
        cases.append(row)
    random.Random(args.seed).shuffle(cases)
    cases=cases[:args.max_cases]
    metrics={'cases':0,'parseable_json':0,'schema_valid':0,'exact_character_set':0,'facts_exact_match':0}
    details=[]
    required={'characters','facts','next_event','beats','constraints'}
    for row in cases:
        ids=tok.encode(prompt_for(row['context']),add_special_tokens=True).ids
        try:
            out=generate(model,ids,device,args.max_new_tokens,cfg.block_size)
            raw=tok.decode(out,skip_special_tokens=True).strip()
            obj=json.loads(raw)
            metrics['parseable_json']+=1
            valid=isinstance(obj,dict) and required.issubset(obj.keys()) and isinstance(obj['characters'],list) and isinstance(obj['facts'],list) and isinstance(obj['beats'],list) and isinstance(obj['constraints'],list) and isinstance(obj['next_event'],str)
            if valid:
                metrics['schema_valid']+=1
                metrics['exact_character_set']+=int(set(obj['characters'])==set(row['characters']))
                metrics['facts_exact_match']+=int(set(obj['facts'])==set(f['fact'] for f in row['facts']))
            error=None
        except Exception as e:
            raw=locals().get('raw','')
            error=str(e)
        metrics['cases']+=1
        details.append({'story_id':row['story_id'],'generated':raw[:4000],'error':error})
        raw=''
    print('METRICS',json.dumps(metrics))
    print('CAUTION: JSON/character/fact string metrics do not validate whether a plan is good or events logically follow; human review required.')
    if args.output:
        output=Path(args.output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'metrics':metrics,'cases':details},ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',output)

if __name__=='__main__':main()
