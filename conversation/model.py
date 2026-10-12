import torch
from torch import nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self,vocab_size=128,block_size=256,n_embd=128,n_head=4,n_layer=3):
        super().__init__()
        self.block_size=block_size
        self.emb=nn.Embedding(vocab_size,n_embd)
        self.pos=nn.Embedding(block_size,n_embd)
        layer=nn.TransformerEncoderLayer(n_embd,n_head,dim_feedforward=4*n_embd,dropout=0.1,batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,n_layer,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(n_embd)
        self.head=nn.Linear(n_embd,vocab_size,bias=False)
    def forward(self,ids,targets=None):
        b,t=ids.shape
        if t>self.block_size: raise ValueError('Sequence exceeds context window')
        x=self.emb(ids)+self.pos(torch.arange(t,device=ids.device))
        mask=torch.triu(torch.ones(t,t,device=ids.device,dtype=torch.bool),diagonal=1)
        logits=self.head(self.norm(self.blocks(x,mask=mask)))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100) if targets is not None else None
        return logits,loss
