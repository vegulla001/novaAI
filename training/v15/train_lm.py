"""Fresh random-initialized GPT, AMP, eval, best checkpoint, resume, strict dataset checks."""
import argparse,hashlib,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from model import GPT,GPTConfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='dataset/v15');p.add_argument('--output',default='checkpoints/storyai_v15_lm.pt')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=16);p.add_argument('--block-size',type=int,default=2048)
    p.add_argument('--layers',type=int,default=8);p.add_argument('--heads',type=int,default=8);p.add_argument('--embedding',type=int,default=512)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--eval-batches',type=int,default=10);p.add_argument('--resume',default=None)
    p.add_argument('--min-train-tokens',type=int,default=5000000);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if not torch.cuda.is_available():raise RuntimeError('V15 LM training requires a CUDA GPU')
    if a.embedding%a.heads:raise ValueError('embedding must be divisible by heads')
    root=Path(a.data);info=json.loads((root/'report.json').read_text());train=np.load(root/'train.npy',mmap_mode='r');val=np.load(root/'val.npy',mmap_mode='r')
    if len(train)<a.min_train_tokens:raise RuntimeError('Insufficient training data; refusing costly run')
    if len(val)<a.block_size+2:raise ValueError('Validation data too short')
    if int(max(train.max(),val.max()))>=info['vocab_size']:raise ValueError('Out-of-range token ID')
    sha=hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest()
    if sha!=info['tokenizer_sha256']:raise ValueError('Tokenizer hash mismatch')
    config=GPTConfig(block_size=a.block_size,vocab_size=info['vocab_size'],n_layer=a.layers,n_head=a.heads,n_embd=a.embedding,dropout=0.0)
    model=GPT(config).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.1)
    first=1;best=float('inf')
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        if ck['config']!=asdict(config) or ck['tokenizer_sha256']!=sha:raise ValueError('Resume config/tokenizer mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['step']+1;best=ck['best_val']
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.cuda()
    use_bf16=torch.cuda.is_bf16_supported();dtype=torch.bfloat16 if use_bf16 else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=not use_bf16)
    def batch(data):
        starts=np.random.randint(0,len(data)-a.block_size-1,size=a.batch_size)
        x=np.stack([data[s:s+a.block_size] for s in starts]).astype('int64')
        y=np.stack([data[s+1:s+a.block_size+1] for s in starts]).astype('int64')
        return torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
    @torch.no_grad()
    def evaluate():
        model.eval();values=[]
        for _ in range(a.eval_batches):
            x,y=batch(val)
            with torch.autocast('cuda',dtype=dtype):_,loss=model(x,y)
            values.append(loss.item())
        return sum(values)/len(values)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for step in range(first,a.steps+1):
        model.train();opt.zero_grad(set_to_none=True);total=0
        for _ in range(a.grad_accum):
            x,y=batch(train)
            with torch.autocast('cuda',dtype=dtype):_,loss=model(x,y);loss=loss/a.grad_accum
            scaler.scale(loss).backward();total+=loss.item()
        scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt);scaler.update()
        if step==first or step%a.eval_every==0 or step==a.steps:
            score=evaluate();print(f'{step}/{a.steps} train={total:.4f} val={score:.4f}',flush=True)
            if score<best:
                best=score
                ck={'model':model.state_dict(),'config':asdict(config),'tokenizer_sha256':sha,'step':step,'best_val':best,'training_stage':'v15_lm'}
                torch.save(ck,out)
            # resume state is separate from best inference weights
            torch.save({'model':model.state_dict(),'config':asdict(config),'tokenizer_sha256':sha,'step':step,'best_val':best,'optimizer':opt.state_dict()},out.with_name(out.stem+'_resume.pt'))
if __name__=='__main__':main()
