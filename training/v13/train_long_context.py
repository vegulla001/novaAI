"""Extend RoPE context of a trusted StoryAI checkpoint and continue LM training.
Run from project root: python -m training.v13.train_long_context --block-size 1024 --steps 1000
No pretrained weights or external APIs. Checkpoints are saved separately.
"""
import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from model import GPT, GPTConfig


def load_config(value):
    return GPTConfig(**value) if isinstance(value, dict) else value


def extend_context(model, length):
    """RoPE has no learned position table; rebuild nonpersistent causal masks if needed."""
    if length < model.config.block_size:
        raise ValueError('New context must be >= checkpoint context')
    model.config.block_size = length
    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, 'mask'):
            attn.mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=attn.mask.device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--data-dir', default='dataset/processed', help='Directory with train.npy and val.npy')
    p.add_argument('--output', default='checkpoints/storyai_v13_long_context.pt')
    p.add_argument('--report', default='output/v13_long_context/report.json')
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.block_size, args.steps, args.batch_size, args.grad_accum, args.eval_every, args.eval_batches) < 1:
        p.error('All sizes and counts must be positive')
    if args.lr <= 0: p.error('--lr must be positive')
    if Path(args.checkpoint).resolve() == Path(args.output).resolve(): p.error('Refusing to overwrite input checkpoint')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    tokenizer_file = Path(args.tokenizer)
    digest = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('tokenizer_sha256') and ck['tokenizer_sha256'] != digest:
        raise ValueError('Tokenizer does not match checkpoint')
    config = load_config(ck['config'])
    previous_context = config.block_size
    if args.block_size < previous_context: p.error(f'Context must be >= original {previous_context}')
    model = GPT(config)
    model.load_state_dict(ck['model'], strict=True)
    extend_context(model, args.block_size)
    del ck
    folder = Path(args.data_dir)
    train = np.load(folder / 'train.npy', mmap_mode='r')
    val = np.load(folder / 'val.npy', mmap_mode='r')
    if min(len(train), len(val)) <= args.block_size + 1:
        raise ValueError('Train/val token arrays must each exceed context length')
    if int(max(train.max(), val.max())) >= config.vocab_size:
        raise ValueError('Dataset has token IDs outside checkpoint vocabulary')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        print('WARNING: CPU long-context training will be slow; CUDA GPU recommended.', flush=True)
    model.to(device)
    bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    amp = device == 'cuda'
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and not bf16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    def batch(data):
        starts = np.random.randint(0, len(data) - args.block_size - 1, size=args.batch_size)
        x = np.stack([data[s:s+args.block_size] for s in starts]).astype(np.int64)
        y = np.stack([data[s+1:s+args.block_size+1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    def autocast():
        return torch.autocast(device_type='cuda', dtype=dtype) if amp else nullcontext()
    @torch.no_grad()
    def evaluate():
        model.eval(); scores=[]
        for _ in range(args.eval_batches):
            x, y = batch(val)
            with autocast(): _, loss = model(x, y)
            scores.append(loss.item())
        return sum(scores)/len(scores)
    baseline = evaluate()
    best = float('inf')
    print(f'Context {previous_context} -> {args.block_size}; device={device}; val_before={baseline:.4f}', flush=True)
    for step in range(1, args.steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True); total=0.0
        for _ in range(args.grad_accum):
            x, y = batch(train)
            with autocast(): _, loss = model(x, y)
            total += loss.detach().item()
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = evaluate()
            print(f'{step}/{args.steps} train={total/args.grad_accum:.4f} val={score:.4f}', flush=True)
            if math.isfinite(score) and score < best:
                best = score
                dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model':model.state_dict(), 'config':asdict(config),
                            'tokenizer_sha256':digest, 'step':step, 'best_val':best,
                            'training_stage':'v13_long_context_continuation',
                            'original_context':previous_context}, dest)
    report = {'input_checkpoint':args.checkpoint, 'output_checkpoint':args.output,
              'original_context':previous_context, 'new_context':args.block_size,
              'validation_loss_before':baseline, 'best_validation_loss':best,
              'steps':args.steps, 'device':device,
              'note':'Longer context is not proof of instruction adherence or improved prose.'}
    rp=Path(args.report); rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',args.output,'and',rp)

if __name__ == '__main__': main()
