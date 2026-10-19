"""Validate GPU and StoryAI inputs before paid training."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from model import GPT, GPTConfig

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/storyai_v8_best.pt'); p.add_argument('--data-dir',default='dataset/processed_structured'); p.add_argument('--block-size',type=int,default=2048); p.add_argument('--require-gpu',action='store_true'); a=p.parse_args()
 print('PyTorch:',torch.__version__,'CUDA build:',torch.version.cuda,'CUDA available:',torch.cuda.is_available())
 if a.require_gpu and not torch.cuda.is_available(): sys.exit('ERROR: CUDA GPU not available; do not start paid training.')
 if torch.cuda.is_available():
  d=torch.cuda.get_device_properties(0); print('GPU:',d.name,'VRAM GiB:',round(d.total_memory/2**30,2),'BF16:',torch.cuda.is_bf16_supported())
 folder=Path(a.data_dir); required=[folder/'train.npy',folder/'val.npy',folder/'tokenizer.json',Path(a.checkpoint)]
 missing=[str(f) for f in required if not f.is_file()]
 if missing: sys.exit('MISSING required files:\n'+'\n'.join(missing))
 train=np.load(folder/'train.npy',mmap_mode='r'); val=np.load(folder/'val.npy',mmap_mode='r')
 for name, arr in [('train',train),('val',val)]:
  print(name,'tokens:',len(arr));
  if len(arr)<=a.block_size+1: sys.exit(f'ERROR: {name} too short for {a.block_size} context')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 cfg=GPTConfig(**ck['config']) if isinstance(ck['config'],dict) else ck['config']
 sha=hashlib.sha256((folder/'tokenizer.json').read_bytes()).hexdigest()
 if ck.get('tokenizer_sha256') and ck['tokenizer_sha256']!=sha: sys.exit('ERROR: tokenizer hash differs from checkpoint')
 if a.block_size<cfg.block_size: sys.exit('ERROR: context cannot be smaller than checkpoint')
 model=GPT(cfg); model.load_state_dict(ck['model'],strict=True)
 if max(int(train.max()),int(val.max()))>=cfg.vocab_size: sys.exit('ERROR: dataset token IDs exceed vocabulary')
 print('CHECK PASSED. Baseline parameters:',sum(x.numel() for x in model.parameters()),'old context:',cfg.block_size,'target context:',a.block_size)
 print('Note: context extension is an experiment, not instruction-following training.')
if __name__=='__main__': main()
