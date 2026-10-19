"""Instruction SFT from reviewed JSONL, best-val checkpoint, prompt-masked causal loss."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT,GPTConfig

def prompt(row):
    chars=', '.join(c['name'] if isinstance(c,dict) else str(c) for c in row['characters'])
    return (f"<INSTRUCTION>\nGenre: {', '.join(row.get('genres',[])) if row.get('genres') else row.get('genre','fiction')}\n"
            f"Characters: {chars}\nSetting: {row['setting']}\nGoal: {row.get('objective',row.get('goal',''))}\n"
            f"Obstacle: {row['obstacle']}\nOutcome: {row['outcome']}\n<STORY>\n")

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='checkpoints/storyai_v15_lm.pt')
    p.add_argument('--tokenizer',default='dataset/v15/tokenizer.json');p.add_argument('--train',default='dataset/v12/pilot_expanded/train.jsonl')
    p.add_argument('--val',default='dataset/v12/pilot_expanded/val.jsonl');p.add_argument('--out',default='checkpoints/storyai_v15_instruction.pt')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--min-pairs',type=int,default=1000)
    a=p.parse_args();random.seed(42);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    ck=torch.load(a.base,map_location='cpu',weights_only=False)
    sha=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256')!=sha:raise ValueError('Tokenizer mismatch')
    tok=Tokenizer.from_file(a.tokenizer)
    def load(path):
        rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        for r in rows:
            for k in ('characters','setting','obstacle','outcome','story'):
                if not r.get(k):raise ValueError(f'Missing {k} in {path}')
        return rows
    train=load(a.train);val=load(a.val)
    if len(train)<a.min_pairs:raise RuntimeError(f'Only {len(train)} training pairs; require {a.min_pairs} reviewed pairs. Do not pay for SFT yet.')
    if len(val)<10:raise RuntimeError('Need >=10 held-out validation pairs')
    model=GPT(GPTConfig(**ck['config'])).cuda();model.load_state_dict(ck['model']);del ck
    def encode(row):
        pref=tok.encode(prompt(row)).ids;story=tok.encode(row['story']+'\n<END_STORY>').ids
        maxlen=model.config.block_size+1
        if len(pref)+len(story)>maxlen:story=story[:maxlen-len(pref)]
        if len(story)<2:raise ValueError('Instruction exceeds context')
        ids=pref+story;x=torch.tensor(ids[:-1],device='cuda')[None,:]
        y=torch.tensor(ids[1:],device='cuda')[None,:];y[:,:max(0,len(pref)-1)]=-1
        return x,y
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    @torch.no_grad()
    def evaluate():
        model.eval();losses=[]
        for row in val:
            x,y=encode(row)
            with torch.autocast('cuda',dtype=dtype):_,loss=model(x,y)
            losses.append(loss.item())
        return sum(losses)/len(losses)
    best=float('inf');out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for step in range(1,a.steps+1):
        model.train();x,y=encode(random.choice(train));opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):_,loss=model(x,y)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update()
        if step==1 or step%a.eval_every==0:
            score=evaluate();print(f'{step} train={loss.item():.4f} val={score:.4f}',flush=True)
            if score<best:
                best=score;torch.save({'model':model.state_dict(),'config':model.config.__dict__,
                   'tokenizer_sha256':sha,'best_val':best,'step':step,'training_stage':'v15_instruction'},out)
if __name__=='__main__':main()
