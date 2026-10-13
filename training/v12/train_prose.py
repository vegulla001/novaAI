"""Optional prose LM continuation of V8. Save separate checkpoint; NEVER overwrite V8."""
import argparse, hashlib, json, math, random
from pathlib import Path
import torch
from model import GPT,GPTConfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='dataset/v12/prose_processed')
    p.add_argument('--checkpoint',default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--tokenizer',default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output',default='checkpoints/storyai_v12_prose_best.pt')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--grad-accum',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.grad_accum,a.eval_every)<1:p.error('Positive counts required')
    if Path(a.checkpoint).resolve()==Path(a.output).resolve():p.error('Output must differ from base checkpoint')
    random.seed(a.seed);torch.manual_seed(a.seed)
    meta=json.loads((Path(a.data)/'meta.json').read_text(encoding='utf-8'))
    if meta['kind']!='prose_lm':raise ValueError('Wrong dataset kind')
    digest=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    if digest!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch with data')
    base=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if base.get('tokenizer_sha256') and digest!=base['tokenizer_sha256']:raise ValueError('Tokenizer mismatch with checkpoint')
    cfg=GPTConfig(**base['config'])
    if meta['block_size']>cfg.block_size:raise ValueError('Block size exceeds model context')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=GPT(cfg).to(device);model.load_state_dict(base['model']);del base
    def read(split):
        rows=[json.loads(s)['ids'] for s in (Path(a.data)/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        if not rows:raise ValueError(f'{split} empty')
        return rows
    train,val=read('train'),read('val')
    def batch(rows):
        chosen=random.choices(rows,k=a.batch_size)
        n=max(len(s)-1 for s in chosen)
        x=torch.zeros((len(chosen),n),dtype=torch.long)
        y=torch.full((len(chosen),n),-1,dtype=torch.long)
        for i,s in enumerate(chosen):
            x[i,:len(s)-1]=torch.tensor(s[:-1]);y[i,:len(s)-1]=torch.tensor(s[1:])
        return x.to(device),y.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.01)
    amp=device=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=float('inf')
    for step in range(1,a.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);running=0
        for _ in range(a.grad_accum):
            x,y=batch(train)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):_,loss=model(x,y)
            scaler.scale(loss/a.grad_accum).backward();running+=loss.item()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        scaler.step(optimizer);scaler.update()
        if step==1 or step%10==0:print(f'step={step} train={running/a.grad_accum:.4f}',flush=True)
        if step%a.eval_every==0 or step==a.steps:
            model.eval();values=[]
            with torch.inference_mode():
                for _ in range(min(10,len(val))):
                    x,y=batch(val)
                    with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):_,loss=model(x,y)
                    values.append(loss.item())
            score=sum(values)/len(values)
            print(f'VAL step={step} loss={score:.4f} ppl={math.exp(min(20,score)):.2f}',flush=True)
            if score<best:
                best=score;out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'model':model.state_dict(),'config':vars(cfg),'tokenizer_sha256':digest,'step':step,'best_val':best,'training_kind':'prose_lm'},out)
                print('Saved',out,flush=True)
if __name__=='__main__':main()
