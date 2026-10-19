"""Train StoryAI on reviewed instruction/story pairs. No pretrained APIs.
Uses correct next-token targets and prompt masking; splits must be independent.
"""
import argparse, hashlib, json, math, random
from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def format_prompt(row):
    names = row.get('characters', [])
    if isinstance(names, list):
        names = ', '.join(str(n.get('name', n) if isinstance(n, dict) else n) for n in names)
    return ('<INSTRUCTION>\nGenre: ' + ', '.join(row.get('genres', [])) +
            '\nCharacters: ' + str(names) + '\nSetting: ' + str(row.get('setting','')) +
            '\nGoal: ' + str(row.get('objective','')) + '\nObstacle: ' + str(row.get('obstacle','')) +
            '\nOutcome: ' + str(row.get('outcome','')) + '\nPremise: ' + str(row.get('premise','')) +
            '\n<STORY>\n')


def load_pairs(path, tokenizer, max_length):
    samples=[]; seen=set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if not row.get('story') or not row.get('characters'): continue
        if row.get('review_status') not in ('approved','human_approved','authored','verified','AUTHOR_WRITTEN_PILOT'):
            continue
        prompt=tokenizer.encode(format_prompt(row)); target=tokenizer.encode(row['story']+'\n<END_STORY>')
        if len(prompt)+2 > max_length: continue
        target=target[:max_length-len(prompt)]
        if len(target)<16: continue
        full=prompt+target
        x=torch.tensor(full[:-1],dtype=torch.long)
        y=torch.tensor(full[1:],dtype=torch.long)
        y[:len(prompt)-1]=-1
        if torch.all(y==-1): continue
        key=hashlib.sha256(row['story'].strip().encode()).hexdigest()
        if key in seen: continue
        seen.add(key); samples.append((x,y,key,row.get('source_family','')))
    return samples


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/storyai_v13_long_context.pt')
    p.add_argument('--tokenizer',default='dataset/processed/tokenizer.json')
    p.add_argument('--train',default='dataset/v12/pilot_expanded/train.jsonl')
    p.add_argument('--val',default='dataset/v12/pilot_expanded/val.jsonl')
    p.add_argument('--output',default='checkpoints/storyai_v14_instruction.pt')
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.steps<1 or args.grad_accum<1: p.error('steps and grad-accum must be positive')
    if Path(args.output).resolve()==Path(args.checkpoint).resolve(): p.error('Cannot overwrite input checkpoint')
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok=BPETokenizer.load(args.tokenizer)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=GPTConfig(**ck['config']) if isinstance(ck['config'],dict) else ck['config']
    if tok.vocab_size!=config.vocab_size: raise ValueError('Tokenizer vocabulary mismatch')
    digest=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256') and ck['tokenizer_sha256']!=digest: raise ValueError('Tokenizer SHA mismatch')
    model=GPT(config); model.load_state_dict(ck['model'],strict=True)
    train=load_pairs(args.train,tok,config.block_size)
    val=load_pairs(args.val,tok,config.block_size)
    if not train or not val: raise ValueError('No approved train/val pairs; review and add data before training')
    train_keys={(s[2],s[3]) for s in train}; val_keys={(s[2],s[3]) for s in val}
    if {s[2] for s in train}&{s[2] for s in val}: raise ValueError('Train/val story overlap')
    train_families={s[3] for s in train if s[3]}; val_families={s[3] for s in val if s[3]}
    if train_families & val_families: raise ValueError('Train/val source_family leakage')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    bf16=device=='cuda' and torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and not bf16)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    def amp(): return torch.autocast('cuda',dtype=dtype) if device=='cuda' else nullcontext()
    def score(samples):
        model.eval(); values=[]
        with torch.no_grad():
            for x,y,*_ in samples:
                with amp(): _,loss=model(x[None].to(device),y[None].to(device))
                values.append(loss.item())
        return sum(values)/len(values)
    best=float('inf'); before=score(val)
    print(f'Instruction pairs train={len(train)} val={len(val)} baseline_val={before:.4f}',flush=True)
    for step in range(1,args.steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x,y,*_=random.choice(train)
            with amp(): _,loss=model(x[None].to(device),y[None].to(device))
            scaler.scale(loss/args.grad_accum).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt); scaler.update()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            value=score(val); print(f'{step}/{args.steps} val={value:.4f}',flush=True)
            if math.isfinite(value) and value<best:
                best=value; dest=Path(args.output); dest.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'model':model.state_dict(),'config':asdict(config),'tokenizer_sha256':digest,
                            'training_stage':'v14_instruction','step':step,'best_val':best,
                            'train_pairs':len(train),'val_pairs':len(val)},dest)
    print(f'Saved {args.output}; val {before:.4f} -> {best:.4f}; evaluate story quality separately')

if __name__=='__main__': main()
