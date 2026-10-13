import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from model import GPT,GPTConfig


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--data',default='dataset/v11/processed')
    p.add_argument('--tokenizer',default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output',default='checkpoints/storyai_v11_best.pt')
    p.add_argument('--resume',default='checkpoints/storyai_v11_last_train.pt')
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--block-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--eval-batches',type=int,default=20)
    p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    meta=json.loads((Path(args.data)/'meta.json').read_text())
    digest=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if digest!=meta['tokenizer_sha256']:raise ValueError('Dataset tokenizer hash mismatch')
    base=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if base.get('tokenizer_sha256') and digest!=base['tokenizer_sha256']:raise ValueError('Checkpoint tokenizer hash mismatch')
    cfg=GPTConfig(**base['config'])
    if args.block_size>cfg.block_size:raise ValueError('block-size exceeds model context')
    model=GPT(cfg).to(device);model.load_state_dict(base['model']);del base
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),weight_decay=0.1)
    train=np.memmap(Path(args.data)/'train.bin',dtype=np.uint32,mode='r')
    val=np.memmap(Path(args.data)/'val.bin',dtype=np.uint32,mode='r')
    if min(len(train),len(val))<=args.block_size+1:raise ValueError('Dataset too short')
    use_amp=device=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    def batch(arr):
        positions=np.random.randint(0,len(arr)-args.block_size-1,size=args.batch_size)
        xs=np.stack([arr[i:i+args.block_size].astype(np.int64) for i in positions])
        ys=np.stack([arr[i+1:i+args.block_size+1].astype(np.int64) for i in positions])
        return torch.from_numpy(xs).to(device),torch.from_numpy(ys).to(device)
    start=0;best=float('inf')
    if Path(args.resume).exists():
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ck['tokenizer_sha256']!=digest or ck['config']!=vars(cfg):raise ValueError('Resume checkpoint incompatible')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start=ck['step'];best=ck['best_val']
        print('Resuming from',start,flush=True)
    @torch.no_grad()
    def evaluate():
        model.eval();values=[]
        for _ in range(args.eval_batches):
            x,y=batch(val)
            with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):
                _,loss=model(x,y)
            values.append(loss.item())
        model.train();return sum(values)/len(values)
    def save(path,step,training):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload={'model':model.state_dict(),'config':vars(cfg),'tokenizer_sha256':digest,'step':step,'best_val':best}
        if training:payload.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict())
        temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
    model.train();optimizer.zero_grad(set_to_none=True)
    for step in range(start+1,args.steps+1):
        total=0
        for _ in range(args.grad_accum):
            x,y=batch(train)
            with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):
                _,loss=model(x,y)
                scaled=loss/args.grad_accum
            scaler.scale(scaled).backward();total+=loss.item()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        if step%25==0:print(f'step {step}/{args.steps} train_loss={total/args.grad_accum:.4f}',flush=True)
        if step%args.eval_every==0 or step==args.steps:
            value=evaluate();print(f'VAL step={step} loss={value:.4f} ppl={math.exp(min(value,20)):.2f}',flush=True)
            if value<best:best=value;save(args.output,step,False);print('Saved best',args.output,flush=True)
        if step%args.save_every==0 or step==args.steps:save(args.resume,step,True)
    print('Training finished. Best val loss:',best)
if __name__=='__main__':main()
