"""Create native PyTorch initialized reference checkpoint for pipeline testing."""
import argparse
import torch
from .model import ModelConfig, NativeNovelLM
from .checkpoint import save_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='checkpoints/native_small/model.pt')
    p.add_argument('--seed', type=int, default=42)
    args=p.parse_args()
    torch.manual_seed(args.seed)
    model=NativeNovelLM(ModelConfig())
    save_checkpoint(args.out,model,step=0,metadata={
        'status':'INITIALIZED_UNTRAINED', 'quality':'NOT SUITABLE FOR WRITING NOVELS',
        'weights_origin':'Random initialization by native PyTorch code; no external model weights'})
    print(f'initialized checkpoint saved: {args.out}')
if __name__ == '__main__': main()
