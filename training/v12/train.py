"""Fine-tune StoryAI V8 on curated instruction/story examples with masked targets."""
import argparse, hashlib, json, math, random
from pathlib import Path
import torch
from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--data', default='dataset/v12/processed')
    p.add_argument('--tokenizer', default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output', default='checkpoints/storyai_v12_best.pt')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every) < 1: p.error('Counts must be positive')
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data)
    meta = json.loads((data_dir / 'meta.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if digest != meta['tokenizer_sha256']: raise ValueError('Dataset tokenizer hash mismatch')
    base = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if base.get('tokenizer_sha256') and digest != base['tokenizer_sha256']:
        raise ValueError('Checkpoint tokenizer hash mismatch')
    cfg = GPTConfig(**base['config'])
    if meta['max_length'] > cfg.block_size: raise ValueError('Dataset length exceeds checkpoint context')
    model = GPT(cfg).to(device); model.load_state_dict(base['model']); del base
    def load(name):
        rows = [json.loads(s) for s in (data_dir / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        if not rows: raise ValueError(f'{name} empty')
        return rows
    train, val = load('train'), load('val')
    def batch(rows):
        chosen = random.choices(rows, k=args.batch_size)
        length = max(len(row['ids']) - 1 for row in chosen)
        x = torch.zeros((len(chosen), length), dtype=torch.long)
        y = torch.full((len(chosen), length), -1, dtype=torch.long)
        for i, row in enumerate(chosen):
            seq = row['ids']; labels = row['labels']; n = len(seq) - 1
            x[i, :n] = torch.tensor(seq[:-1], dtype=torch.long)
            y[i, :n] = torch.tensor(labels, dtype=torch.long)
        return x.to(device), y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best = float('inf')
    for step in range(1, args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.0
        for _ in range(args.grad_accum):
            x, y = batch(train)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum).backward(); total += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if step % 10 == 0 or step == 1: print(f'step={step} train_loss={total/args.grad_accum:.4f}', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval(); losses = []
            with torch.no_grad():
                for _ in range(min(10, max(1, len(val)))):
                    x, y = batch(val)
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                        _, loss = model(x, y)
                    losses.append(loss.item())
            score = sum(losses) / len(losses)
            print(f'VAL step={step} loss={score:.4f} ppl={math.exp(min(score,20)):.2f}', flush=True)
            if score < best:
                best = score
                path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model': model.state_dict(), 'config': vars(cfg),
                            'tokenizer_sha256': digest, 'step': step, 'best_val': best}, path)
                print('Saved', path, flush=True)

if __name__ == '__main__': main()
