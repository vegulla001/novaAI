"""StoryAI V4: train your randomly initialized PyTorch Transformer on pretokenized novels."""
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
from dataset import get_batch, load_tokens
from model import GPT, GPTConfig


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lr_at(step, warmup, total, peak, minimum):
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return minimum
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='dataset/processed')
    p.add_argument('--checkpoint', default='checkpoints/storyai_v6.pt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=1000, help='Total optimizer updates, including resumed steps')
    p.add_argument('--batch-size', type=int, default=8, help='Micro-batch size')
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-head', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=3e-5)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--amp', choices=['auto', 'off', 'fp16', 'bf16'], default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=0)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.eval_every < 1:
        p.error('steps, batch-size, grad-accum, and eval-every must be positive')
    if args.warmup < 0 or args.min_lr < 0 or args.lr <= 0 or args.min_lr > args.lr:
        p.error('Require warmup >= 0 and 0 <= min-lr <= lr')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    if args.amp in ('fp16', 'bf16') and device != 'cuda':
        p.error('fp16/bf16 AMP requires CUDA; use --amp off on CPU')
    use_amp = device == 'cuda' and args.amp != 'off'
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' or (args.amp == 'auto' and torch.cuda.is_bf16_supported()) else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        p.error('This GPU does not support bfloat16; choose --amp fp16')
    autocast_ctx = lambda: torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    folder = Path(args.data_dir)
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if not meta.get('genre_conditioned'):
        raise ValueError('Run python prepare_dataset.py to create V5 genre data')
    tokenizer_path = folder / 'tokenizer.json'
    tokenizer_hash = sha256_file(tokenizer_path)
    train = load_tokens(folder / 'train.npy')
    val = load_tokens(folder / 'val.npy')
    config = GPTConfig(vocab_size=meta['vocab_size'], block_size=args.block_size,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       dropout=args.dropout)
    start_step = 0
    checkpoint_path = Path(args.checkpoint)
    if args.resume:
        # Only load checkpoints you created/trust. PyTorch state files are not an interchange security boundary.
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config = GPTConfig(**saved['config'])
        if saved['tokenizer_sha256'] != tokenizer_hash:
            raise ValueError('Tokenizer differs from checkpoint; restore the original dataset/processed tokenizer')
        if saved.get('dataset_sha256') != sha256_file(folder / 'metadata.json'):
            raise ValueError('Dataset metadata changed; do not resume against a different corpus')
        start_step = int(saved['step'])
    if min(len(train), len(val)) <= config.block_size:
        raise ValueError('Not enough tokens for block size; add novels or lower --block-size')
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scaler.is_enabled() and saved.get('scaler') is not None:
            scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'])
        np_state = saved['numpy_rng']
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4]))
        random.setstate(tuple(saved['python_rng']))
        if device == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Device={device} AMP={amp_dtype if use_amp else "off"} parameters={model.get_num_params():,} '
          f'train_tokens={len(train):,} val_tokens={len(val):,} start_step={start_step}', flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(val, args.batch_size, config.block_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses)

    best_val = float('inf')
    stale_evals = 0
    model.train()
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.warmup, args.steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.batch_size, config.block_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            train_loss += float(loss.detach()) / args.grad_accum
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        if completed == 1 or completed % args.eval_every == 0 or completed == args.steps:
            val_loss = evaluate()
            print(f'step={completed} lr={lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f}', flush=True)
            if val_loss < best_val:
                best_val, stale_evals = val_loss, 0
                torch.save({'model': model.state_dict(), 'config': asdict(config), 'step': completed,
                            'tokenizer_sha256': tokenizer_hash}, checkpoint_path.with_name('storyai_v6_best.pt'))
            else:
                stale_evals += 1
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict() if scaler.is_enabled() else None,
                         config=asdict(config), step=completed,
                         tokenizer_sha256=tokenizer_hash,
                         dataset_sha256=sha256_file(folder / 'metadata.json'),
                         torch_rng=torch.get_rng_state(), numpy_rng=(lambda s: (s[0], s[1].tolist(), s[2], s[3], s[4]))(np.random.get_state()),
                         python_rng=random.getstate(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None)
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(state, temp_path)
            temp_path.replace(checkpoint_path)
            if args.patience and stale_evals >= args.patience:
                print("Early stopping: validation did not improve")
                break
    print(f'Saved: {checkpoint_path}')

if __name__ == '__main__':
    main()
