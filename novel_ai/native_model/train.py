"""From-scratch next-token training on USER-OWNED/LICENSED plain-text corpus.

Usage: python -m novel_ai.native_model.train --data-dir data/owned_books --steps 10000
       python -m novel_ai.native_model.train --resume checkpoints/native_small/model.pt --data-dir data/owned_books --steps 20000

No pretrained weights, no third-party model architecture, no external tokenizer.
"""
import argparse
import json
import random
from pathlib import Path
import torch
from .model import ModelConfig, NativeNovelLM
from .tokenizer import ByteTokenizer
from .checkpoint import save_checkpoint, load_checkpoint

def load_corpus(path):
    root = Path(path)
    files = sorted(root.rglob('*.txt'))
    if not files: raise ValueError(f'No .txt files in {root}. Supply licensed/owned novels.')
    tokenizer = ByteTokenizer()
    streams = []
    for f in files:
        b = tokenizer.encode(f.read_text(encoding='utf-8'), add_eos=True)
        if len(b) >= 96: streams.append(torch.tensor(b, dtype=torch.long))
    if not streams: raise ValueError('No input texts are long enough (at least 96 bytes).')
    return streams

def batch(streams, context, batch_size, device):
    eligible = [s for s in streams if len(s) >= context + 2]
    if not eligible: raise ValueError('Input texts shorter than context length; choose --context-length smaller')
    starts = [(s, random.randint(0, len(s)-context-2)) for s in random.choices(eligible, k=batch_size)]
    x = torch.stack([s[i:i+context] for s,i in starts]).to(device)
    y = torch.stack([s[i+1:i+context+1] for s,i in starts]).to(device)
    return x,y

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', default='checkpoints/native_small/model.pt')
    p.add_argument('--resume')
    p.add_argument('--steps', type=int, default=10000, help='Total steps, including resumed steps')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--context-length', type=int, default=512)
    p.add_argument('--d-model', type=int, default=192)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--n-heads', type=int, default=6)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    streams = load_corpus(args.data_dir)
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device, training=True)
        start = payload['step']
    else:
        model = NativeNovelLM(ModelConfig(context_length=args.context_length,
                      d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)).to(device)
        start = 0; payload = {}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    if 'optimizer' in payload:
        opt.load_state_dict(payload['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor): state[key] = value.to(device)
    for step in range(start+1, args.steps+1):
        x,y = batch(streams, model.config.context_length, args.batch_size, device)
        _, loss = model(x,y)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f'step={step}/{args.steps} loss={loss.item():.4f} device={device}', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.out, model, step=step, optimizer=opt,
                 metadata={'training_corpus':'user-supplied; not bundled', 'kind':'causal-language-model'})
            print(f'checkpoint={args.out} step={step}', flush=True)
if __name__ == '__main__': main()
