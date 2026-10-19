"""Train a clean authored-only V8 branch; no synthetic replay or placeholder checkpoint."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from training.v12.train_natural_scenes import load_jsonl, make_authored_samples, generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--data-dir', default='dataset/v12/pilot_expanded')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-checkpoint', default='checkpoints/storyai_v12_v8_authored.pt')
    p.add_argument('--output', default='output/v12_v8_authored')
    args = p.parse_args()
    if args.steps < 1: p.error('--steps must be positive')
    if Path(args.checkpoint).resolve() == Path(args.save_checkpoint).resolve(): p.error('Input/output checkpoint collision')
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    path = Path(args.tokenizer)
    tok = BPETokenizer.load(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('tokenizer_sha256') and ck['tokenizer_sha256'] != sha: raise ValueError('Tokenizer mismatch')
    cfg = GPTConfig(**ck['config']) if isinstance(ck['config'], dict) else ck['config']
    model = GPT(cfg)
    model.load_state_dict(ck['model'], strict=True)
    root = Path(args.data_dir)
    train_rows = load_jsonl(root / 'train.jsonl')
    val_rows = load_jsonl(root / 'val.jsonl')
    family = lambda r: r.get('source_family', r.get('title'))
    if {family(r) for r in train_rows} & {family(r) for r in val_rows}:
        raise ValueError('Train/validation family leakage')
    train = [sample for r in train_rows for sample in make_authored_samples(r, tok, cfg.block_size)]
    val = [sample for r in val_rows for sample in make_authored_samples(r, tok, cfg.block_size)]
    if not train or not val: raise ValueError('No fitting authored scenes')
    def evaluate():
        model.eval()
        with torch.no_grad():
            return sum(model(x, y)[1].item() for (x, y), _ in val) / len(val)
    before = evaluate()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    for step in range(1, args.steps + 1):
        model.train()
        (x, y), _ = rng.choice(train)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f'{step}/{args.steps} authored_loss={loss.item():.4f}')
    after = evaluate()
    torch.manual_seed(args.seed + 1)
    examples = []
    for _, meta in val[:4]:
        raw = generate(model, tok, meta['prompt'], cfg.block_size)
        examples.append({'family': meta['family'], 'prompt': meta['prompt'], 'reference': meta['target'],
                         'generated_raw': raw, 'has_char_1': '<CHAR_1>' in raw, 'has_char_2': '<CHAR_2>' in raw})
    report = {'status': 'experimental_authored_only_v8_branch', 'steps': args.steps,
              'train_scenes': len(train), 'validation_scenes': len(val),
              'validation_loss_before': before, 'validation_loss_after': after,
              'synthetic_replay': False, 'test_set_untouched': True, 'examples': examples}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    dest = Path(args.save_checkpoint)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': cfg, 'tokenizer_sha256': sha,
                'training_stage': 'authored_only_v8', 'steps': args.steps}, dest)
    print(f'Validation: {before:.4f} -> {after:.4f}; checkpoint: {dest}; report: {out / "report.json"}')

if __name__ == '__main__': main()
