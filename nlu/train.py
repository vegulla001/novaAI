import argparse,json,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .model import NLUModel,collate

def read(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def evaluate(model,loader,device):
    model.eval();correct=total=tag_correct=tag_total=0
    with torch.no_grad():
        for x,lengths,labels,tags in loader:
            x,lengths,labels,tags=[t.to(device) for t in (x,lengths,labels,tags)]
            il,tl=model(x,lengths)
            correct+=(il.argmax(-1)==labels).sum().item();total+=len(labels)
            mask=tags!=-100;tag_correct+=((tl.argmax(-1)==tags)&mask).sum().item();tag_total+=mask.sum().item()
    return correct/max(1,total),tag_correct/max(1,tag_total)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--lr',type=float,default=.001);p.add_argument('--data',default='dataset/nlu');p.add_argument('--checkpoint',default='checkpoints/storyai_v10_nlu.pt');args=p.parse_args()
    random.seed(42);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=DataLoader(read(Path(args.data)/'train.jsonl'),batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val=DataLoader(read(Path(args.data)/'val.jsonl'),batch_size=args.batch_size,collate_fn=collate)
    model=NLUModel().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=-1
    print('Device:',device,'Parameters:',sum(p.numel() for p in model.parameters()))
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0
        for x,lengths,labels,tags in train:
            x,lengths,labels,tags=[t.to(device) for t in (x,lengths,labels,tags)]
            il,tl=model(x,lengths)
            loss=torch.nn.functional.cross_entropy(il,labels)+torch.nn.functional.cross_entropy(tl.reshape(-1,tl.shape[-1]),tags.reshape(-1),ignore_index=-100)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=loss.item()
        ia,ta=evaluate(model,val,device)
        print(f'epoch={epoch} loss={loss_sum/len(train):.3f} heldout_intent={ia:.3f} heldout_tag={ta:.3f}')
        score=(ia+ta)/2
        if score>best:
            best=score;path=Path(args.checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'hidden':128,'intent_accuracy':ia,'tag_accuracy':ta},path)
            print('Saved',path)
if __name__=='__main__':main()
