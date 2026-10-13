"""Character/plot grounding curriculum for V8. Standalone; original checkpoints untouched.
Run: python -m training.v12.train_grounding --steps 600
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

# Synthetic supervision tests variable binding, NOT natural novel quality.
NAMES = ['Rhea','Pax','Oren','Lysa','Ilse','Rowan','Dara','Luis','Mira','Tovin',
         'Selka','Bram','Neris','Calen','Veya','Jorin','Tessa','Hale','Kira','Soren',
         'Aven','Bela','Ciro','Demi','Eren','Fara','Galen','Hina','Ivo','Jessa',
         'Kelan','Luma','Niko','Orla','Pera','Quin','Rumi','Sela','Taro','Una',
         'Vico','Wren','Xena','Yara','Zeno','Asha','Borin','Celia','Doran','Elin',
         'Fenn','Greta','Hugo','Inara','Jace','Kael','Lina','Marek','Nora','Olin']
SETTINGS = [
    ('an underwater archive','a damaged navigation chart','rising seawater','copy the chart to a mechanical recorder'),
    ('a mountain observatory','a missing star map','a failing telescope','recover the map from the backup drawer'),
    ('a riverside workshop','a broken signal lantern','an approaching storm','repair the lantern before nightfall'),
    ('an abandoned station','a sealed supply crate','a jammed door','open the crate with the emergency key'),
    ('a forest watchtower','a lost messenger note','a spreading wildfire','carry the note down the safe trail'),
    ('a desert library','a faded treaty','a collapsing roof','save the treaty inside a metal case'),
    ('a harbor warehouse','a stolen manifest','a locked loading gate','retrieve the manifest through the side entrance'),
    ('a snowbound cabin','a damaged radio','a depleted battery','power the radio with a hand crank'),
    ('a village infirmary','a medicine ledger','a leaking ceiling','move the ledger into a dry cabinet'),
    ('an underground greenhouse','a rare seed collection','a failed water pump','seal the seeds in protective jars'),
]

def record(rng, names, settings):
    a,b = rng.sample(names,2)
    setting, objective, obstacle, outcome = rng.choice(settings)
    return {'characters':[a,b], 'setting':setting,'objective':objective,'obstacle':obstacle,'outcome':outcome}

def prefix(r):
    a,b=r['characters']
    return (f'<INSTRUCTION>\nCharacters: {a}, {b}\nSetting: {r["setting"]}\n'
            f'Goal: {r["objective"]}\nObstacle: {r["obstacle"]}\n'
            f'Outcome: {r["outcome"]}\n<STORY>\n')

def target(r):
    a,b=r['characters']
    return (f'{a} and {b} entered {r["setting"]}. '
            f'They needed to recover {r["objective"]}. '
            f'But {r["obstacle"]} threatened their work. '
            f'{a} kept watch while {b} found a way forward. '
            f'Together they managed to {r["outcome"]}.<END_STORY>')

def encode_sample(r,tok,block):
    # Segment tokenization is consistent with inference; next-token targets are shifted.
    p=tok.encode(prefix(r)); s=tok.encode(target(r))
    ids=p+s
    if len(ids)>block+1:
        raise ValueError(f'Sample length {len(ids)} exceeds {block+1}; reduce prompt or increase model context safely')
    x=torch.tensor(ids[:-1],dtype=torch.long).unsqueeze(0)
    y=torch.tensor(ids[1:],dtype=torch.long).unsqueeze(0)
    y[:,:len(p)-1]=-1
    assert int((y!=-1).sum())==len(s)
    return x,y

def generate(model,tok,r,block,max_new=150):
    ids=tok.encode(prefix(r))
    produced=[]
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            logits,_=model(torch.tensor(ids[-block:],dtype=torch.long).unsqueeze(0))
            nxt=int(torch.argmax(logits[0,-1]).item())
            ids.append(nxt); produced.append(nxt)
            if '<END_STORY>' in tok.decode(produced):break
    return tok.decode(produced).split('<END_STORY>')[0]

def eval_samples(model,samples):
    model.eval(); total=0.
    with torch.no_grad():
        for x,y in samples:total+=float(model(x,y)[1])
    return total/len(samples)

def test_generation(model,tok,records,block):
    rows=[]
    for r in records:
        s=generate(model,tok,r,block)
        names={n:bool(re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',s,re.I)) for n in r['characters']}
        rows.append({'required':r,'generated':s,'characters':names,'both_names':all(names.values()),
                     'setting_mentioned':r['setting'].lower() in s.lower(),
                     'outcome_mentioned':r['outcome'].lower() in s.lower()})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='checkpoints/storyai_v8_best.pt')
    ap.add_argument('--tokenizer',default='dataset/processed/tokenizer.json')
    ap.add_argument('--output',default='output/v12_grounding')
    ap.add_argument('--save-checkpoint',default='checkpoints/storyai_v12_grounding.pt')
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--lr',type=float,default=3e-5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-size',type=int,default=1200)
    ap.add_argument('--val-size',type=int,default=60)
    args=ap.parse_args()
    if args.steps<1 or args.train_size<1 or args.val_size<1:ap.error('positive steps and sizes required')
    if Path(args.save_checkpoint).resolve()==Path(args.checkpoint).resolve():ap.error('Refusing to overwrite V8 checkpoint')
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    rng=random.Random(args.seed);torch.manual_seed(args.seed)
    tokpath=Path(args.tokenizer);tok=BPETokenizer.load(str(tokpath))
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    sha=hashlib.sha256(tokpath.read_bytes()).hexdigest()
    if ckpt.get('tokenizer_sha256') and ckpt['tokenizer_sha256']!=sha:raise ValueError('Tokenizer mismatch')
    config=ckpt['config'];config=GPTConfig(**config) if isinstance(config,dict) else config
    model=GPT(config);model.load_state_dict(ckpt['model'],strict=True)
    # Name-disjoint split tests copying unseen names, not memorizing fixed names.
    train_names=NAMES[:40];val_names=NAMES[40:]
    train=[record(rng,train_names,SETTINGS) for _ in range(args.train_size)]
    val=[record(rng,val_names,SETTINGS) for _ in range(args.val_size)]
    train_samples=[encode_sample(r,tok,config.block_size) for r in train]
    val_samples=[encode_sample(r,tok,config.block_size) for r in val]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    before=eval_samples(model,val_samples)
    print(f'Baseline held-out loss: {before:.4f} | {len(train)} train, {len(val)} validation')
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    for step in range(1,args.steps+1):
        model.train();x,y=train_samples[rng.randrange(len(train_samples))]
        optim.zero_grad(set_to_none=True);_,loss=model(x,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        if step==1 or step%50==0 or step==args.steps:
            print(f'{step}/{args.steps} train_loss={float(loss):.4f}')
    after=eval_samples(model,val_samples)
    examples=test_generation(model,tok,val[:12],config.block_size)
    report={'status':'synthetic_grounding_experiment_not_novel_training',
            'held_out_names':val_names,'steps':args.steps,'val_loss_before':before,'val_loss_after':after,
            'exact_both_names_rate':sum(x['both_names'] for x in examples)/len(examples),
            'exact_setting_rate':sum(x['setting_mentioned'] for x in examples)/len(examples),
            'exact_outcome_rate':sum(x['outcome_mentioned'] for x in examples)/len(examples),
            'examples':examples}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    save=Path(args.save_checkpoint);save.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'config':config,'tokenizer_sha256':sha,
                'training_stage':'synthetic_grounding','steps':args.steps},save)
    print('Held-out loss:',round(before,4),'->',round(after,4))
    print('Held-out both-character recall:',f"{report['exact_both_names_rate']:.0%}")
    print('Report:',out/'report.json');print('Candidate checkpoint:',save)
    print('V8 untouched. Synthetic grounding does not establish novel quality.')

if __name__=='__main__':main()
