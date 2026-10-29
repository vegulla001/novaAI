#!/usr/bin/env python3
"""StoryAI V15 trainer: pretrain or response-only SFT. Run from project root with -m."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from training.v15.model import GPT, GPTConfig


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pretrain', 'sft'], required=True)
    p.add_argument('--data-dir', default='dataset/v15')
    p.add_argument('--out-dir', default='checkpoints/v15')
    p.add_argument('--init-from', default=None, help='Pretraining checkpoint for SFT; required for SFT unless --smoke-test')
    p.add_argument('--resume', default=None, help='Resume same-mode checkpoint including optimizer')
    p.add_argument('--restart-from', default=None, help='Load checkpoint model weights only; fresh optimizer and training schedule')
    p.add_argument('--eval-seed', type=int, default=20261029, help='Separate fixed validation sampling seed')
    p.add_argument('--n-layer', type=int, default=8)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--n-embd', type=int, default=512)
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--lr-decay-steps',type=int,default=None,help='Fixed step at which cosine learning-rate decay reaches min-lr')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument("--sft-dir",type=str,default="dataset/v15/novelist_continuation_sft")
    p.add_argument('--min-lr', type=float, default=3e-5)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--smoke-test', action='store_true', help='One optimizer step; no checkpoints or GPU needed')
    return p.parse_args()


def get_pretrain_batch(tokens, batch_size, block_size, device):
    if len(tokens) <= block_size:
        raise ValueError(f'Pretraining split has {len(tokens)} tokens; need > block-size={block_size}')
    starts = torch.randint(0, len(tokens) - block_size, (batch_size,)).tolist()
    # Use copies to avoid writable numpy memmap warnings.
    x = torch.stack([torch.from_numpy(np.array(tokens[i:i + block_size], dtype=np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(np.array(tokens[i + 1:i + block_size + 1], dtype=np.int64)) for i in starts])
    return x.to(device), y.to(device)


def get_sft_batch(rows, batch_size, block_size, device):
    chosen = random.choices(rows, k=batch_size)
    xs, ys = [], []
    for row in chosen:
        ids = list(row['input_ids'])
        labels = list(row['labels'])
        if len(ids) != len(labels):
            raise ValueError('SFT ids/labels length mismatch')
        if len(ids) < 2:
            raise ValueError('SFT record too short')
        # A causal model predicts labels[1:] from ids[:-1].
        # When cropping, retain the completion boundary and some prompt context.
        if len(ids) > block_size + 1:
            supervised = [i for i, v in enumerate(labels) if v != -100]
            if not supervised:
                raise ValueError('SFT record has no completion tokens')
            boundary = supervised[0]
            start = max(0, boundary - max(1, block_size // 2))
            start = min(start, len(ids) - (block_size + 1))
            ids = ids[start:start + block_size + 1]
            labels = labels[start:start + block_size + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([(-1 if v == -100 else int(v)) for v in labels[1:]], dtype=torch.long)
        if not (y != -1).any():
            raise ValueError('SFT sample contains no supervised targets after cropping')
        xs.append(x)
        ys.append(y)
    # GPT uses ignore_index=-1 (verified by V15 real-model smoke test).
    x = pad_sequence(xs, batch_first=True, padding_value=0)
    y = pad_sequence(ys, batch_first=True, padding_value=-1)
    return x.to(device), y.to(device)


def load_checkpoint(path, device):
    # Only load checkpoints produced by your own trusted training code.
    return torch.load(path, map_location=device, weights_only=False)


def main():
    a = get_args()
    if a.lr_decay_steps is None:
        a.lr_decay_steps = a.max_steps

    if a.lr_decay_steps <= a.warmup_steps:
        raise ValueError(
            '--lr-decay-steps must be greater than --warmup-steps'
        )

    if not 0 < a.min_lr <= a.lr:
        raise ValueError('Require 0 < min-lr <= lr')
    if a.smoke_test:
        a.max_steps, a.grad_accum, a.eval_batches = 1, 1, 1
        a.eval_every, a.save_every = 1, 1
    if sum(bool(v) for v in (a.resume, a.init_from, a.restart_from)) > 1:
        raise ValueError('Choose only one of --resume, --init-from, --restart-from')
    if a.restart_from and a.mode != 'pretrain':
        raise ValueError('--restart-from currently supports pretrain only')
    if a.mode == 'sft' and not (a.init_from or a.resume or a.smoke_test):
        raise ValueError('SFT requires --init-from pretrained V15 checkpoint')
    if min(a.block_size, a.batch_size, a.grad_accum, a.max_steps) < 1:
        raise ValueError('block-size, batch-size, grad-accum and max-steps must be positive')
    if a.n_embd % a.n_head:
        raise ValueError('n-embd must be divisible by n-head')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = 'cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    data = Path(a.data_dir)
    report = json.loads((data / 'build_report.json').read_text(encoding='utf-8-sig'))
    tokenizer_path = data / 'tokenizer.json'

    if not tokenizer_path.is_file():
        tokenizer_path = data.parent / 'tokenizer.json'

    tokenizer_hash = sha256(tokenizer_path)
    if tokenizer_hash != report['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match dataset build report')
    vocab = int(report['vocab_size'])
    if a.mode == 'pretrain':
        train = np.load(data / 'train.npy', mmap_mode='r')
        val = np.load(data / 'val.npy', mmap_mode='r')
        if len(train) <= a.block_size or len(val) <= a.block_size:
            raise ValueError('Not enough tokens for chosen block-size')
        if max(int(train.max()), int(val.max())) >= vocab or min(int(train.min()), int(val.min())) < 0:
            raise ValueError('Invalid pretraining token IDs')
        batch = lambda split: get_pretrain_batch(train if split == 'train' else val, a.batch_size, a.block_size, device)
    else:
        sft_dir = Path(a.sft_dir)

        train = torch.load(
            sft_dir / 'train_sft.pt',
            map_location='cpu',
            weights_only=True,
        )
        val = torch.load(
            sft_dir / 'val_sft.pt',
            map_location='cpu',
            weights_only=True,
        )
        if not train or not val:
            raise ValueError('Empty SFT split')
        if not a.smoke_test and (len(train) < 500 or len(val) < 50):
            raise ValueError(
                f"SFT dataset too small: train={len(train)}, val={len(val)}. "
                "Expected at least 500 training and 50 validation examples."
            )

        print(f"SFT dataset loaded: train={len(train)}, val={len(val)}")
        for split in (train, val):
            for row in split:
                if any(int(t) < 0 or int(t) >= vocab for t in row['input_ids']):
                    raise ValueError('Invalid SFT token ID')
        batch = lambda split: get_sft_batch(train if split == 'train' else val, a.batch_size, a.block_size, device)
    cfg = dict(block_size=a.block_size, vocab_size=vocab, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
    ckpt = None
    if a.resume or a.init_from or a.restart_from:
        ckpt = load_checkpoint(a.resume or a.init_from or a.restart_from, device)
        if ckpt.get('tokenizer_sha256') != tokenizer_hash:
            raise ValueError('Checkpoint tokenizer hash mismatch')
        if a.init_from and ckpt.get('mode') != 'pretrain':
            raise ValueError('--init-from must point to a pretraining checkpoint')
        if (a.resume or a.restart_from) and ckpt.get('mode') != a.mode:
            raise ValueError('Cannot resume checkpoint from different mode')
        cfg = ckpt['model_config']
        if cfg['vocab_size'] != vocab:
            raise ValueError('Checkpoint vocabulary mismatch')
        if a.block_size != cfg['block_size']:
            raise ValueError('Use --block-size matching checkpoint model_config')
    model = GPT(GPTConfig(**cfg)).to(device)
    if ckpt:
        model.load_state_dict(ckpt['model'])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    step = 0
    if a.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        step = int(ckpt['step'])
        if 'torch_rng_state' in ckpt:
            torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        if device == 'cuda' and 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state(ckpt['cuda_rng_state'].cpu())
        if 'python_rng_state' in ckpt:
            random.setstate(ckpt['python_rng_state'])
    use_amp = device == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    if a.resume and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    start_step = step
    if a.restart_from:
        print(f'WEIGHTS_ONLY_RESTART source_step={ckpt.get("step")} schedule_step=0 optimizer=fresh')
    print(json.dumps({'mode': a.mode, 'device': device, 'parameters': sum(p.numel() for p in model.parameters()), 'step': step, 'tokenizer_sha256': tokenizer_hash, 'smoke_test': a.smoke_test, 'amp_dtype': str(amp_dtype) if use_amp else 'none'}, indent=2))
    # Validation windows are fixed across all evaluations and never consume training RNG.
    fixed_batches = None
    if a.mode == 'pretrain':
        rng = np.random.default_rng(a.eval_seed)
        fixed_batches = {}
        for split, tokens in (('train', train), ('val', val)):
            starts = rng.integers(0, len(tokens) - a.block_size, size=(a.eval_batches, a.batch_size))
            fixed_batches[split] = []
            for group in starts:
                x = torch.from_numpy(np.stack([np.array(tokens[int(i):int(i)+a.block_size], dtype=np.int64) for i in group])).to(device)
                y = torch.from_numpy(np.stack([np.array(tokens[int(i)+1:int(i)+a.block_size+1], dtype=np.int64) for i in group])).to(device)
                fixed_batches[split].append((x, y))
    best_val = float('inf')
    if a.mode == 'sft' and len(train) < 1000:
        print(f'WARNING: only {len(train)} SFT training examples. Pipeline testing only; high overfitting risk.')

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = {}
        for split in ('train', 'val'):
            losses = []
            for i in range(a.eval_batches):
                x, y = fixed_batches[split][i] if fixed_batches is not None else batch(split)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                losses.append(float(loss.item()))
            results[split] = sum(losses) / len(losses)
        model.train()
        return results

    def save(best=False):
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'model_config': cfg, 'mode': a.mode,
                 'step': step, 'tokenizer_sha256': tokenizer_hash, 'eval_seed': a.eval_seed, 'best_val': best_val,
                 'torch_rng_state': torch.get_rng_state(),
                 'python_rng_state': random.getstate()}
        if device == 'cuda':
            state['cuda_rng_state'] = torch.cuda.get_rng_state()
        name = f'{a.mode}_best.pt' if best else f'{a.mode}_latest.pt'
        target = out / name
        tmp = out / (name + '.tmp')
        torch.save(state, tmp)
        tmp.replace(target)
        print(f'Checkpoint saved: {target}')

    if not a.smoke_test:
        baseline = evaluate()
        best_val = baseline['val']
        print('BASELINE_FIXED:', json.dumps(baseline), flush=True)
        save(best=True)  # preserve starting weights unless actual validation improvement

    # For weights-only restart, step counts NEW updates. With --resume, it remains absolute.
    while step < a.max_steps:
        optimizer.zero_grad(set_to_none=True)
        schedule_step = step
        if schedule_step < a.warmup_steps:
            lr = a.lr * (schedule_step + 1) / max(1, a.warmup_steps)
        else:
            progress = (schedule_step - a.warmup_steps) / (
                a.lr_decay_steps - a.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))

            lr = a.min_lr + 0.5 * (a.lr - a.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = []
        for _ in range(a.grad_accum):
            x, y = batch('train')
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            losses.append(float(loss.detach()))
            scaler.scale(loss / a.grad_accum).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        print(f'step={step} train_loss={sum(losses)/len(losses):.4f} lr={lr:.3g}', flush=True)
        if step % a.eval_every == 0 or step == a.max_steps:
            scores = evaluate()
            print('eval:', json.dumps(scores), flush=True)
            if not a.smoke_test and scores['val'] < best_val:
                best_val = scores['val']
                save(best=True)
        if not a.smoke_test and (step % a.save_every == 0 or step == a.max_steps):
            save()
    if a.smoke_test:
        print('PASS: one real GPT optimizer step (no checkpoint saved)')


if __name__ == '__main__':
    main()
