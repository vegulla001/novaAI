import torch
from torch import nn
from .dataset import INTENTS,TAGS

# Byte-level character encoder: supports names never seen during training.
PAD=256
MAX_LEN=192
class NLUModel(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.embedding=nn.Embedding(257,96,padding_idx=PAD)
        self.encoder=nn.GRU(96,hidden,bidirectional=True,batch_first=True,num_layers=2,dropout=.1)
        self.intent_head=nn.Linear(hidden*2,len(INTENTS))
        self.tag_head=nn.Linear(hidden*2,len(TAGS))
    def forward(self,ids,lengths):
        x=self.embedding(ids)
        x,_=self.encoder(x)
        mask=(torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None])
        pooled=(x*mask.unsqueeze(-1)).sum(1)/lengths.clamp(min=1).unsqueeze(-1)
        return self.intent_head(pooled),self.tag_head(x)

def encode(text):
    return list(text.encode('utf-8')[:MAX_LEN])

def collate(rows):
    # Dataset generated from ASCII templates/names; labels align with byte positions.
    seqs=[encode(r['text']) for r in rows]
    length=torch.tensor([len(x) for x in seqs],dtype=torch.long)
    width=max(length.tolist())
    x=torch.full((len(rows),width),PAD,dtype=torch.long)
    y=torch.full((len(rows),width),-100,dtype=torch.long)
    intent=torch.tensor([INTENTS.index(r['intent']) for r in rows])
    for i,(row,seq) in enumerate(zip(rows,seqs)):
        x[i,:len(seq)]=torch.tensor(seq)
        y[i,:len(seq)]=torch.tensor([TAGS.index(tag) for tag in row['tags'][:len(seq)]])
    return x,length,intent,y
