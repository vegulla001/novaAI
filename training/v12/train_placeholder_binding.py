"""Train role-placeholder scene generation; bind user names deterministically at inference.
Run: python -m training.v12.train_placeholder_binding --steps 600
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

# Reuse established scenario catalog; character names are deliberately NOT in the LM prompt.
from training.v12.train_grounding import SETTINGS

CHAR1 = '<CHAR_1>'
CHAR2 = '<CHAR_2>'
STOP = '<END_STORY>'


def validate_names(names):
    if len(names) != 2 or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('Exactly two nonempty character names are required')
    return [n.strip() for n in names]


def prompt(r):
    return (f'<INSTRUCTION>\nCharacters: {CHAR1}, {CHAR2}\n'
            f'Setting: {r["setting"]}\nGoal: {r["objective"]}\n'
            f'Obstacle: {r["obstacle"]}\nOutcome: {r["outcome"]}\n<STORY>\n')


def answer(r, variant=0):
    if variant % 3 == 0:
        body = (f'{CHAR1} and {CHAR2} entered {r["setting"]}. '
                f'They needed to recover {r["objective"]}. '
                f'But {r["obstacle"]} threatened their work. '
                f'{CHAR1} kept watch while {CHAR2} found a way forward. '
                f'Together they managed to {r["outcome"]}.')
    elif variant % 3 == 1:
        body = (f'Inside {r["setting"]}, {CHAR2} warned {CHAR1} about {r["obstacle"]}. '
                f'Their task was to recover {r["objective"]}. '
                f'{CHAR1} steadied the equipment as {CHAR2} searched for a solution. '
                f'In the end, they managed to {r["outcome"]}.')
    else:
        body = (f'{CHAR1} brought {CHAR2} to {r["setting"]} to recover {r["objective"]}. '
                f'When {r["obstacle"]} put the mission at risk, {CHAR2} acted quickly. '
                f'{CHAR1} helped, and together they managed to {r["outcome"]}.')
    return body + STOP


def encode(tok, r, variant, block):
    p = tok.encode(prompt(r))
    a = tok.encode(answer(r, variant))
    ids = p + a
    if len(ids) > block + 1:
        raise ValueError(f'Sample length {len(ids)} exceeds block_size+1={block+1}')
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    y[:, :len(p)-1] = -1
    return x, y


def generate(model, tok, r, block, max_new=180):
    ids = tok.encode(prompt(r))
    if len(ids) > block:
        raise ValueError('Prompt exceeds context length')
    produced = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor(ids[-block:], dtype=torch.long).unsqueeze(0)
            logits, _ = model(x)
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
            produced.append(next_id)
            if STOP in tok.decode(produced):
                break
    return tok.decode(produced).split(STOP, 1)[0]


def bind_characters(text, names):
    """Only substitute exact role markers; never alter incidental prose or names."""
    first, second = validate_names(names)
    # Single regex pass avoids replacing placeholder-looking substrings in user names.
    return re.sub(r'<CHAR_[12]>', lambda m: first if m.group() == CHAR1 else second, text)


def check(text, names, r):
    bound = bind_characters(text, names)
    return {
        'placeholder_1': CHAR1 in text,
        'placeholder_2': CHAR2 in text,
        'both_names_after_binding': all(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', bound) for n in names),
        'setting': r['setting'].lower() in bound.lower(),
        'outcome': r['outcome'].lower() in bound.lower(),
        'unresolved_markers': bool(re.search(r'<CHAR_\d+>', bound)),
        'story': bound,
    }


def load_model(path, tokpath):
    tok = BPETokenizer.load(str(tokpath))
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sha = hashlib.sha256(tokpath.read_bytes()).hexdigest()
    if ckpt.get('tokenizer_sha256') and ckpt['tokenizer_sha256'] != sha:
        raise ValueError('Tokenizer hash does not match checkpoint')
    config = ckpt['config']
    if isinstance(config, dict):
        config = GPTConfig(**config)
    model = GPT(config)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, tok, config, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/storyai_v12_character_copy.pt')
    parser.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    parser.add_argument('--save-checkpoint', default='checkpoints/storyai_v12_placeholder.pt')
    parser.add_argument('--output', default='output/v12_placeholder')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if Path(args.checkpoint).resolve() == Path(args.save_checkpoint).resolve():
        parser.error('Input and output checkpoint must differ')
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model, tok, config, sha = load_model(args.checkpoint, Path(args.tokenizer))
    # Separate scenario families (not random duplicates) for validation.
    train_scenarios = SETTINGS[:8]
    val_scenarios = SETTINGS[8:]
    train_records = [dict(zip(('setting', 'objective', 'obstacle', 'outcome'), row)) for row in train_scenarios]
    val_records = [dict(zip(('setting', 'objective', 'obstacle', 'outcome'), row)) for row in val_scenarios]
    train = [encode(tok, r, v, config.block_size) for r in train_records for v in range(3)]
    val = [encode(tok, r, v, config.block_size) for r in val_records for v in range(3)]

    def val_loss():
        model.eval()
        with torch.no_grad():
            return sum(model(x, y)[1].item() for x, y in val) / len(val)

    before = val_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = rng.choice(train)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f'{step}/{args.steps} train_loss={loss.item():.4f}')
    after = val_loss()
    # Distinct arbitrary names at evaluation, not represented in LM tokens.
    names = ['Marek', 'Celia']
    rows = []
    for r in val_records:
        raw = generate(model, tok, r, config.block_size)
        result = check(raw, names, r)
        rows.append({'scenario': r, 'names': names, 'raw': raw, **result})
    report = {
        'status': 'placeholder_binding_pilot_not_novel_quality',
        'train_scenario_families': len(train_records),
        'held_out_scenario_families': len(val_records),
        'loss_before': before, 'loss_after': after,
        'both_placeholders_rate': sum(r['placeholder_1'] and r['placeholder_2'] for r in rows) / len(rows),
        'both_names_after_binding_rate': sum(bool(r['both_names_after_binding']) for r in rows) / len(rows),
        'setting_rate': sum(r['setting'] for r in rows) / len(rows),
        'outcome_rate': sum(r['outcome'] for r in rows) / len(rows),
        'examples': rows,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    save = Path(args.save_checkpoint)
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config, 'tokenizer_sha256': sha,
                'training_stage': 'placeholder_binding', 'steps': args.steps}, save)
    print(f'Validation loss: {before:.4f} -> {after:.4f}')
    print('Both placeholders:', f'{report["both_placeholders_rate"]:.0%}')
    print('Both names after binding:', f'{report["both_names_after_binding_rate"]:.0%}')
    print('Report:', out / 'report.json')
    print('Checkpoint:', save)


if __name__ == '__main__':
    main()
