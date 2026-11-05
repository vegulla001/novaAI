#!/usr/bin/env python3
"""v16 supervised exact-name copy + named-scene curriculum. No injected generation.

Synthetic examples are task tests, not evidence of general novel coherence.
"""
import argparse, json, random, re, hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer

HELDOUT = ('Arin','Elara','Mira','Jonas','Nila','Dev')
STARTS = ('Al','Bel','Cor','Dar','Fen','Gal','Hal','Jen','Kel','Lor','Mar','Nor','Or','Pel','Quen','Ral','Sel','Tor','Ul','Val','Wen','Yor','Zan')
ENDS = ('adin','ara','en','ion','ella','ric','orin','essa','ian','ora','eth','ira','ielle','enor','isa','alis')
OPENINGS = (
    '{a} stopped beside {b} near the locked door.',
    '{b} held the lantern while {a} examined the sealed letter.',
    'Outside the stone archway, {a} warned {b} to stay close.',
    '{a} heard footsteps. {b} looked toward the dark passage.',
    'The rain grew stronger as {b} followed {a} into the courtyard.',
    '{b} found a missing page and handed it to {a}.',
)
SCENES = (
    '{a} glanced at {b}. "We should wait until the footsteps stop," {a} said.',
    '{b} stepped beside {a} and lowered the lantern. "I agree," {b} whispered.',
    '{a} folded the letter while {b} watched the closed door.',
    '"Stay close," {b} told {a}. Neither approached the dark passage.',
    '{a} asked {b} to listen. The footsteps came from the corridor ahead.',
    '{b} nodded to {a}. Together they moved away from the locked door.',
)

def make_record(tok, prompt, answer, max_tokens):
    # Preserve causal mask boundary by encoding separately.
    p = tok.encode(prompt, add_special_tokens=False).ids
    a = tok.encode(answer, add_special_tokens=False).ids
    if not p or not a or len(p)+len(a)>max_tokens: return None
    return {'input_ids':p+a,'labels':[-100]*len(p)+a}

def name_pool():
    return [a+b for a in STARTS for b in ENDS if a+b not in HELDOUT]

def make_synthetic(tok, pool, count, seed, max_tokens):
    rng=random.Random(seed)
    rec=[]; roles={'copy':0,'scene':0}
    for i in range(count):
        a,b=rng.sample(pool,2)
        opening=rng.choice(OPENINGS).format(a=a,b=b)
        if i%2 == 0:
            # Exact copy task with no ambiguous output field.
            prompt=f'Copy exactly the two character names from this story. Do not invent names.\nStory: {opening}\nCharacters: '
            target=f'{a}, {b}\n'
            role='copy'
        else:
            # Short supervised prose; both required names in target.
            prompt=(f'Continue this story in one sentence using both named characters.\n'
                    f'Characters to preserve: {a}, {b}\nStory so far:\n{opening}\nContinuation:\n')
            target=rng.choice(SCENES).format(a=a,b=b)+'\n'
            role='scene'
        row=make_record(tok,prompt,target,max_tokens)
        if not row: raise RuntimeError('Synthetic record does not fit maximum tokens')
        rec.append(row);roles[role]+=1
    rng.shuffle(rec)
    return rec,roles

def build_replay(tok, source_path, count, seed, max_tokens, excluded):
    src=torch.load(source_path,map_location='cpu',weights_only=True)
    rng=random.Random(seed); idx=list(range(len(src)));rng.shuffle(idx)
    out=[]; stats={'excluded':0,'overlong':0,'invalid':0}
    bad=re.compile(r'(?<!\w)(?:'+ '|'.join(map(re.escape,excluded)) + r')(?!\w)',re.I)
    for i in idx:
        if len(out)==count:break
        row=src[i]; ids=list(row['input_ids']);labels=list(row['labels'])
        if len(ids)!=len(labels) or not any(v!=-100 for v in labels):stats['invalid']+=1;continue
        if len(ids)>max_tokens:stats['overlong']+=1;continue
        if bad.search(tok.decode(ids,skip_special_tokens=True)):stats['excluded']+=1;continue
        out.append(row)
    if len(out)<count:raise RuntimeError(f'Replay short: got {len(out)}/{count}, stats={stats}')
    return out,stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--replay-dir',default='dataset/v15_prose_expanded/continuation_sft')
    p.add_argument('--out-dir',default='dataset/v16_character_copy_sft')
    p.add_argument('--train-synthetic',type=int,default=8000)
    p.add_argument('--val-synthetic',type=int,default=800)
    p.add_argument('--train-replay',type=int,default=2000)
    p.add_argument('--val-replay',type=int,default=200)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--seed',type=int,default=20261105)
    a=p.parse_args()
    if min(a.train_synthetic,a.val_synthetic,a.train_replay,a.val_replay)<1:raise ValueError('All record counts must be positive')
    tok=Tokenizer.from_file(a.tokenizer)
    rng=random.Random(a.seed)
    all_names=name_pool();rng.shuffle(all_names)
    train_names=all_names[:240]
    val_names=all_names[240:]
    if len(val_names)<20:raise ValueError('Too few validation names')
    if set(train_names)&set(val_names) or set(HELDOUT)&set(train_names):raise RuntimeError('Name split leaked')
    train,tc=make_synthetic(tok,train_names,a.train_synthetic,a.seed,a.max_tokens)
    val,vc=make_synthetic(tok,val_names,a.val_synthetic,a.seed+1,a.max_tokens)
    tr,trs=build_replay(tok,Path(a.replay_dir)/'train_sft.pt',a.train_replay,a.seed+2,a.max_tokens,HELDOUT)
    vr,vrs=build_replay(tok,Path(a.replay_dir)/'val_sft.pt',a.val_replay,a.seed+3,a.max_tokens,HELDOUT)
    train+=tr;val+=vr
    random.Random(a.seed+4).shuffle(train);random.Random(a.seed+5).shuffle(val)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save(train,out/'train_sft.pt');torch.save(val,out/'val_sft.pt')
    report={'train':len(train),'val':len(val),'train_synthetic_tasks':tc,'val_synthetic_tasks':vc,
            'train_replay':len(tr),'val_replay':len(vr),'train_name_pool_size':len(train_names),
            'val_name_pool_size':len(val_names),'name_pool_overlap':len(set(train_names)&set(val_names)),
            'heldout_probe_names':HELDOUT,'replay_train_filter':trs,'replay_val_filter':vrs,
            'limitations':['Synthetic task templates are limited and do not prove novel coherence',
                           'Replay samples are inherited; book-level train/validation separation not established',
                           'Heldout test names excluded from synthetic train names and selected replay samples, not necessarily from prior model pretraining']}
    (out/'build_report_copy.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('CHARACTER_COPY_DATASET_BUILT',json.dumps(report,indent=2))
if __name__=='__main__':main()
