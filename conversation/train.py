import argparse,json,random
from pathlib import Path
import torch
from .model import TinyGPT

# Byte-level encoding is trained from scratch; IDs 0..255 represent UTF-8 bytes, 256 is EOS.
EOS=256; VOCAB=257

def encode(s): return list(s.encode('utf-8'))
def prompt(user): return f'USER: {user}\nACTION: '
def target(action): return json.dumps(action,ensure_ascii=True,separators=(',',':'))+'\n'
def examples(path,block):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r=json.loads(line); prefix=encode(prompt(r['user'])); full=prefix+encode(target(r['action']))+[EOS]
        if len(full)>block: continue
        rows.append((full,len(prefix)))
    if not rows: raise ValueError('No samples fit block size; increase --block-size')
    return rows

def batch(rows,size,block,device):
    xs=[];ys=[]
    for _ in range(size):
        full,start=random.choice(rows)
        seq=full[:-1]; labels=full[1:]
        labels=[(-100 if i<start-1 else v) for i,v in enumerate(labels)]
        xs.append(seq+[0]*(block-len(seq)))
        ys.append(labels+[-100]*(block-len(labels)))
    return torch.tensor(xs,device=device),torch.tensor(ys,device=device)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset/conversation');p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--block-size',type=int,default=256);p.add_argument('--n-embd',type=int,default=128);p.add_argument('--n-layer',type=int,default=3);p.add_argument('--n-head',type=int,default=4);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--checkpoint',default='checkpoints/storyai_v9_conversation.pt');a=p.parse_args()
    random.seed(42);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=examples(Path(a.data)/'train.jsonl',a.block_size);val=examples(Path(a.data)/'val.jsonl',a.block_size)
    config=dict(vocab_size=VOCAB,block_size=a.block_size,n_embd=a.n_embd,n_layer=a.n_layer,n_head=a.n_head)
    model=TinyGPT(**config).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    path=Path(a.checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
    print('Device:',device,'Parameters:',sum(x.numel() for x in model.parameters()),'Samples:',len(train),len(val),flush=True)
    best=float('inf')
    for step in range(1,a.steps+1):
        model.train();x,y=batch(train,a.batch_size,a.block_size,device);_,loss=model(x,y)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%a.eval_every==0 or step==a.steps:
            model.eval()
            with torch.no_grad():
                losses=[]
                for _ in range(5):
                    x,y=batch(val,a.batch_size,a.block_size,device);_,l=model(x,y);losses.append(l.item())
                vl=sum(losses)/len(losses)
            print(f'step={step} train={loss.item():.3f} val={vl:.3f}',flush=True)
            if vl<best:
                best=vl;torch.save({'config':config,'model':model.state_dict(),'step':step,'val_loss':vl},path)
    print('Saved best checkpoint:',path)
if __name__=='__main__': main()
