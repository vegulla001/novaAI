"""Fine-tune V12 placeholder checkpoint on authored scene records, with validation and replay.
Run: python -m training.v12.train_natural_scenes --steps 400
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from training.v12.train_placeholder_binding import (CHAR1, CHAR2, STOP, SETTINGS,
                                                    prompt as template_prompt,
                                                    answer as template_answer)


def get_names(rec):
    chars = rec.get('characters', [])
    if not isinstance(chars, list) or len(chars) != 2:
        return None
    names = []
    for c in chars:
        if isinstance(c, str):
            name = c.strip()
        elif isinstance(c, dict):
            name = str(c.get('name', '')).strip()
        else:
            return None
        if not name or any(x in name for x in ('<', '>', '\n')):
            return None
        names.append(name)
    if names[0].casefold() == names[1].casefold():
        return None
    return names


def replace_names(text, names):
    # Longer names first; one pass, case-insensitive, word boundaries.
    order = sorted(range(2), key=lambda i: len(names[i]), reverse=True)
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(names[i]) for i in order) + r')(?!\w)', re.I)
    lookup = {names[0].casefold(): CHAR1, names[1].casefold(): CHAR2}
    return pattern.sub(lambda m: lookup[m.group().casefold()], text)


def scene_prompt(rec, previous=''):
    # No actual names appear in LM prompt. Only the role placeholders are learned.
    setting = rec.get('setting', '')
    objective = rec.get('objective', '') or rec.get('premise', '')
    obstacle = rec.get('obstacle', '')
    outcome = rec.get('outcome', '')
    fields = [f'Characters: {CHAR1}, {CHAR2}', f'Setting: {setting}',
              f'Goal: {objective}', f'Obstacle: {obstacle}', f'Outcome: {outcome}']
    if previous:
        fields.append('Previous scene: ' + previous)
    return '<INSTRUCTION>\n' + '\n'.join(fields) + '\n<STORY>\n'


def sentences(text):
    # Split at paragraph boundaries and sentence endings, retaining punctuation.
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\s*\n', text.strip()) if s.strip()]


def fit(tok, p, body, block):
    p_ids = tok.encode(p)
    body_ids = tok.encode(body + STOP)
    if len(p_ids) + len(body_ids) > block + 1:
        return None
    ids = p_ids + body_ids
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    y[:, :len(p_ids)-1] = -1
    if not (y != -1).any():
        return None
    return x, y


def make_authored_samples(rec, tok, block, max_scenes=8):
    names = get_names(rec)
    story = rec.get('story', '')
    if not names or not isinstance(story, str) or not story.strip():
        return []
    converted = replace_names(story, names)
    chunks = sentences(converted)
    if not chunks:
        return []
    # Reject unconverted character names (rather than silently teaching wrong bindings).
    if any(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', converted, re.I) for n in names):
        return []
    result = []
    cursor = 0
    prev = ''
    while cursor < len(chunks) and len(result) < max_scenes:
        p = scene_prompt(rec, prev)
        end = cursor
        best = None
        # Pack complete sentences only. Never truncate mid-sentence or claim the ending was learned.
        while end < len(chunks):
            candidate = ' '.join(chunks[cursor:end+1])
            sample = fit(tok, p, candidate, block)
            if sample is None:
                break
            best = (sample, candidate)
            end += 1
        if best is None:
            # If one sentence alone is too long, skip it rather than cutting it.
            cursor += 1
            prev = ''
            continue
        sample, scene = best
        result.append((sample, {'prompt': p, 'target': scene, 'family': rec.get('source_family', rec.get('title', 'unknown'))}))
        cursor = end
        # Short continuity memory; if prompt is too large, it is dropped automatically.
        prev_candidate = scene[-160:]
        prev = prev_candidate if len(tok.encode(scene_prompt(rec, prev_candidate))) < block // 2 else ''
    return result


def generate(model, tok, p, block, max_new=220, temperature=0.85, top_k=30):
    ids = tok.encode(p)
    if len(ids) >= block:
        raise ValueError('Prompt exceeds model context')
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(torch.tensor(ids[-block:], dtype=torch.long).unsqueeze(0))
            scores = logits[0, -1] / max(temperature, 1e-5)
            if top_k > 0:
                vals, idx = torch.topk(scores, min(top_k, scores.numel()))
                probs = torch.softmax(vals, dim=-1)
                next_id = int(idx[torch.multinomial(probs, 1)].item())
            else:
                next_id = int(torch.multinomial(torch.softmax(scores, -1), 1).item())
            ids.append(next_id)
            out.append(next_id)
            if STOP in tok.decode(out):
                break
    return tok.decode(out).split(STOP, 1)[0]


def load_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; use the pilot_expanded authored dataset')
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{line_no}: {e}') from e
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='dataset/v12/pilot_expanded')
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v12_placeholder.pt')
    ap.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    ap.add_argument('--save-checkpoint', default='checkpoints/storyai_v12_natural_candidate.pt')
    ap.add_argument('--output', default='output/v12_natural_scenes')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--replay-ratio', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.steps < 1 or not (0 <= args.replay_ratio <= 1):
        ap.error('Require steps > 0 and 0 <= replay-ratio <= 1')
    if Path(args.checkpoint).resolve() == Path(args.save_checkpoint).resolve():
        ap.error('Input and output checkpoints must be different')
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    tok_path = Path(args.tokenizer)
    tok = BPETokenizer.load(str(tok_path))
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('tokenizer_sha256') and ckpt['tokenizer_sha256'] != sha:
        raise ValueError('Tokenizer does not match checkpoint')
    cfg = ckpt['config']
    cfg = GPTConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = GPT(cfg)
    model.load_state_dict(ckpt['model'], strict=True)

    root = Path(args.data_dir)
    train_rows = load_jsonl(root / 'train.jsonl')
    val_rows = load_jsonl(root / 'val.jsonl')
    test_path = root / 'test.jsonl'  # intentionally never loaded
    train_families = {r.get('source_family', r.get('title')) for r in train_rows}
    val_families = {r.get('source_family', r.get('title')) for r in val_rows}
    if train_families & val_families:
        raise ValueError('Train/validation source_family overlap; fix data leakage before training')

    train = [item for r in train_rows for item in make_authored_samples(r, tok, cfg.block_size)]
    val = [item for r in val_rows for item in make_authored_samples(r, tok, cfg.block_size)]
    if not train or not val:
        raise ValueError('No fitting authored scenes. Inspect story, characters and 512-token budget.')
    print(f'Authored scenes: {len(train)} train / {len(val)} validation; test untouched: {test_path}')

    replay = []
    for row in SETTINGS:
        r = dict(zip(('setting', 'objective', 'obstacle', 'outcome'), row))
        for v in range(3):
            s = fit(tok, template_prompt(r), template_answer(r, v).removesuffix(STOP), cfg.block_size)
            if s:
                replay.append(s)

    def validation_loss():
        model.eval()
        with torch.no_grad():
            return sum(model(*xy)[1].item() for xy, _ in val) / len(val)

    before = validation_loss()
    print(f'Validation loss before: {before:.4f}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        model.train()
        if replay and rng.random() < args.replay_ratio:
            x, y = rng.choice(replay)
            label = 'replay'
        else:
            (x, y), _ = rng.choice(train)
            label = 'authored'
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f'{step}/{args.steps} {label} loss={loss.item():.4f}')
    after = validation_loss()

    # Samples are from held-out authored families. Sampling is reproducible.
    torch.manual_seed(args.seed + 1)
    examples = []
    for (_, meta) in val[:min(4, len(val))]:
        raw = generate(model, tok, meta['prompt'], cfg.block_size)
        examples.append({'source_family': meta['family'], 'prompt': meta['prompt'],
                         'reference': meta['target'], 'generated_raw': raw,
                         'has_char_1': CHAR1 in raw, 'has_char_2': CHAR2 in raw})
    report = {'status': 'small_authored_pilot_not_proven_novel_quality',
              'checkpoint_input': args.checkpoint, 'train_records': len(train_rows),
              'validation_records': len(val_rows), 'train_scenes': len(train),
              'validation_scenes': len(val), 'train_families': sorted(map(str, train_families)),
              'validation_families': sorted(map(str, val_families)),
              'steps': args.steps, 'replay_ratio': args.replay_ratio,
              'validation_loss_before': before, 'validation_loss_after': after,
              'examples': examples, 'test_set_untouched': True}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    save = Path(args.save_checkpoint)
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': cfg, 'tokenizer_sha256': sha,
                'training_stage': 'natural_scenes_pilot', 'steps': args.steps}, save)
    print(f'Validation loss: {before:.4f} -> {after:.4f}')
    print(f'Report: {out / "report.json"}')
    print(f'Candidate checkpoint: {save}')
    print('Review generated_raw for genuine prose quality; low loss is not proof of coherence.')


if __name__ == '__main__':
    main()
