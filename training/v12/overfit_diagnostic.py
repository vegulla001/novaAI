"""CPU-only V12 learning diagnostic on the 8 hand-written pilot training examples.

Never writes or overwrites checkpoints. This is NOT a V12 training recipe.
Run from project root: python -m training.v12.overfit_diagnostic
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def prompt_for(r):
    # Keep this aligned with smoke_test.py and V12's training prompt format.
    return ('<INSTRUCTION>\nWrite a ' + ', '.join(r['genres']) + ' story as narrative prose.\n'
            + 'Title: ' + r['title'] + '\nPremise: ' + r['premise'] + '\n'
            + 'Characters: ' + ', '.join(r['characters']) + '\n'
            + 'Setting: ' + r['setting'] + '\nChapter 1: ' + r['objective'] + '\n'
            + 'Write only the scene.\n<STORY>\n')


def encode_example(r, tokenizer, block_size):
    prefix = tokenizer.encode(prompt_for(r))
    story = tokenizer.encode(r['story'].strip() + '\n<END_STORY>')
    if len(prefix) >= block_size - 2:
        raise ValueError(f'Instruction too long for model context: {r["title"]}')
    ids = (prefix + story)[:block_size + 1]
    if len(ids) <= len(prefix) + 1:
        raise ValueError(f'No story tokens available: {r["title"]}')
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    y[:, :len(prefix)-1] = -1  # Predict first story token, not instruction tokens.
    assert (y != -1).any()
    return x, y


def mean_loss(model, samples):
    model.eval()
    with torch.no_grad():
        losses = [model(x, y)[1].item() for x, y in samples]
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='dataset/v12/pilot/train.jsonl')
    ap.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not 1 <= args.steps <= 200:
        ap.error('--steps must be 1..200 (CPU diagnostic only)')
    if not 0 < args.lr <= 1e-3:
        ap.error('--lr must be >0 and <=0.001')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    records = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('No training examples; run python -m training.v12.build_pilot first')
    for r in records:
        for key in ('title', 'genres', 'characters', 'setting', 'premise', 'objective', 'story'):
            if not r.get(key):
                raise ValueError(f'Missing {key} in record {r.get("title", "unknown")}')
        for name in r['characters']:
            if name.casefold() not in r['story'].casefold():
                raise ValueError(f'Character {name!r} missing in story {r["title"]}')

    path = Path(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected = ckpt.get('tokenizer_sha256')
    if expected is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Tokenizer SHA256 mismatch; do not bypass this check')
    tok = BPETokenizer.load(path)
    cfg = GPTConfig(**ckpt['config'])
    model = GPT(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    samples = [encode_example(r, tok, cfg.block_size) for r in records]
    initial = mean_loss(model, samples)
    print(f'Pilot examples: {len(samples)} | Initial mean story loss: {initial:.4f}', flush=True)
    print('Running a TEMPORARY in-memory CPU learning diagnostic; no files will be changed.', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    for step in range(args.steps):
        model.train()
        x, y = samples[step % len(samples)]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step+1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, args.steps // 6) == 0 or step == 0:
            print(f'Step {step+1}/{args.steps}: training example loss={loss.item():.4f}', flush=True)
    final = mean_loss(model, samples)
    print(f'Initial mean loss: {initial:.4f} | Final mean loss: {final:.4f} | Change: {final-initial:+.4f}')
    if final < initial:
        print('PASS: model can reduce loss on these examples. This does NOT prove instruction following.')
    else:
        print('WARNING: loss did not decrease. Inspect learning rate, masking, and sample construction.')
    print('No checkpoint saved; V8 remains unchanged. Do not train on the 8-example pilot for production.')


if __name__ == '__main__':
    main()
