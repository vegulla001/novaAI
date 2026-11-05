#!/usr/bin/env python3
"""Compile human-reviewed JSONL planning examples to v15-compatible SFT tensors.

Never auto-approves model-generated labels. Uses story_id grouping to prevent split leakage.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from tokenizers import Tokenizer
from training.v16.planning import prompt_for, target_for, validate_row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='JSONL containing curated, reviewed story plans')
    p.add_argument('--data-dir',default='dataset/v15_prose_expanded')
    p.add_argument('--out-dir',default='dataset/v16_planning_sft')
    p.add_argument('--val-ratio',type=float,default=0.2)
    p.add_argument('--max-tokens',type=int,default=1025,help='full ids length: block_size+1 max')
    p.add_argument('--min-train',type=int,default=100)
    p.add_argument('--min-val',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if not 0<a.val_ratio<1 or a.max_tokens<3 or a.min_train<1 or a.min_val<1:
        p.error('Invalid ratio, max tokens, or minimum counts')
    source=Path(a.source)
    data=Path(a.data_dir)
    out=Path(a.out_dir)
    report=json.loads((data/'build_report.json').read_text(encoding='utf-8-sig'))
    tok_path=data/'tokenizer.json'
    actual_sha=hashlib.sha256(tok_path.read_bytes()).hexdigest()
    if actual_sha != report['tokenizer_sha256']:
        raise ValueError('Tokenizer/report hash mismatch')
    tok=Tokenizer.from_file(str(tok_path))
    eos=tok.token_to_id('<eos>')
    if eos is None: raise ValueError('Tokenizer missing <eos>')
    rows=[]
    for line_no,line in enumerate(source.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        try:
            row=json.loads(line)
            validate_row(row)
        except Exception as e:
            raise ValueError(f'{source}:{line_no}: {e}') from e
        rows.append(row)
    if not rows:raise ValueError('Source contains no reviewed records')
    grouped={}
    for row in rows: grouped.setdefault(row['story_id'],[]).append(row)
    keys=sorted(grouped)
    if len(keys)<2:raise ValueError('Need at least two distinct story_id values for leakage-safe split')
    random.Random(a.seed).shuffle(keys)
    val_count=max(1,min(len(keys)-1,round(len(keys)*a.val_ratio)))
    val_ids=set(keys[:val_count])
    splits={'train':[], 'val':[]}
    source_splits={'train':[], 'val':[]}
    counts={'train':0,'val':0}
    vocab=int(report['vocab_size'])
    for row in rows:
        prompt=prompt_for(row['context'])
        answer=target_for(row)
        # No postprocessing: preserve training prompt/target identity.
        p_ids=tok.encode(prompt,add_special_tokens=True).ids
        t_ids=tok.encode(answer,add_special_tokens=False).ids+[eos]
        ids=p_ids+t_ids
        if len(ids)>a.max_tokens:
            raise ValueError(f"story_id={row['story_id']}: {len(ids)} tokens > {a.max_tokens}. Shorten context/plan; do not silently truncate.")
        if not p_ids or not t_ids or any(not 0<=int(v)<vocab for v in ids):
            raise ValueError('Invalid token IDs')
        labels=[-100]*len(p_ids)+t_ids
        split='val' if row['story_id'] in val_ids else 'train'
        splits[split].append({'input_ids':torch.tensor(ids,dtype=torch.long), 'labels':torch.tensor(labels,dtype=torch.long)})
        source_splits[split].append(row)
        counts[split]+=1
    if counts['train']<a.min_train or counts['val']<a.min_val:
        raise ValueError(f"Too few examples: {counts}. Need >= {a.min_train} train and >= {a.min_val} val. Collect and review more stories; do not lower minimum for a real training run.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; refusing to overwrite dataset')
    out.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        torch.save(splits[split],out/f'{split}_sft.pt')
        (out/f'{split}_source.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in source_splits[split]),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'source':str(source),'records':counts,'stories':{'train':len(keys)-val_count,'val':val_count},'seed':a.seed,'tokenizer_sha256':actual_sha,'max_tokens':a.max_tokens,'warning':'reviewed=true is an assertion by dataset curator, not automatic semantic verification'},indent=2),encoding='utf-8')
    print('BUILT',out,'records',counts,'story groups',{'train':len(keys)-val_count,'val':val_count})

if __name__=='__main__':main()
