"""V12: train explicit unseen-name copying and story grounding; never overwrite V8.
Run: python -m training.v12.train_character_copy --steps 1200
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

NAMES = ['Rhea','Pax','Oren','Lysa','Ilse','Rowan','Dara','Luis','Mira','Tovin',
         'Selka','Bram','Neris','Calen','Veya','Jorin','Tessa','Hale','Kira','Soren',
         'Aven','Bela','Ciro','Demi','Eren','Fara','Galen','Hina','Ivo','Jessa',
         'Kelan','Luma','Niko','Orla','Pera','Quin','Rumi','Sela','Taro','Una',
         'Vico','Wren','Xena','Yara','Zeno','Asha','Borin','Celia','Doran','Elin',
         'Fenn','Greta','Hugo','Inara','Jace','Kael','Lina','Marek','Nora','Olin']
SETTINGS = [
    ('an underwater archive','a damaged navigation chart','rising seawater','copy the chart to a mechanical recorder'),
    ('a mountain observatory','a missing star map','a failing telescope','recover the map from the backup drawer'),
    ('a riverside workshop','a broken signal lantern','an approaching storm','repair the lantern before nightfall'),
    ('an abandoned station','a sealed supply crate','a jammed door','open the crate with the emergency key'),
    ('a forest watchtower','a lost messenger note','a spreading wildfire','carry the note down the safe trail'),
    ('a desert library','a faded treaty','a collapsing roof','save the treaty inside a metal case'),
    ('a harbor warehouse','a stolen manifest','a locked loading gate','retrieve the manifest through the side entrance'),
    ('a snowbound cabin','a damaged radio','a depleted battery','power the radio with a hand crank'),
    ('a village infirmary','a medicine ledger','a leaking ceiling','move the ledger into a dry cabinet'),
    ('an underground greenhouse','a rare seed collection','a failed water pump','seal the seeds in protective jars'),
]


def make_record(rng, names):
    a, b = rng.sample(names, 2)
    setting, objective, obstacle, outcome = rng.choice(SETTINGS)
    return dict(characters=[a, b], setting=setting, objective=objective,
                obstacle=obstacle, outcome=outcome)


def copy_prompt(r):
    a, b = r['characters']
    return f'<INSTRUCTION>\nCharacters: {a}, {b}\n<COPY_CHARACTERS>\n'


def copy_answer(r):
    return f'{r["characters"][0]}, {r["characters"][1]}\n<END_COPY>'


def story_prompt(r):
    a, b = r['characters']
    return (f'<INSTRUCTION>\nCharacters: {a}, {b}\nSetting: {r["setting"]}\n'
            f'Goal: {r["objective"]}\nObstacle: {r["obstacle"]}\n'
            f'Outcome: {r["outcome"]}\n<STORY>\n')


def story_answer(r):
    a, b = r['characters']
    return (f'{a} and {b} entered {r["setting"]}. '
            f'They needed to recover {r["objective"]}. '
            f'But {r["obstacle"]} threatened their work. '
            f'{a} kept watch while {b} found a way forward. '
            f'Together they managed to {r["outcome"]}.<END_STORY>')


def encode_task(tok, prompt, answer, block):
    # IMPORTANT: GPT.forward() uses same-position CE; targets must be shifted here.
    # Separate segment encoding matches inference, including BPE boundary behavior.
    p = tok.encode(prompt)
    a = tok.encode(answer)
    ids = p + a
    if len(ids) > block + 1:
        raise ValueError(f'{len(ids)} tokens exceed context limit {block + 1}')
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    y[:, :len(p)-1] = -1
    assert int((y != -1).sum()) == len(a)
    return x, y


def generate(model, tok, prompt, block, stop, max_new=150):
    ids = tok.encode(prompt)
    output = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor(ids[-block:], dtype=torch.long).unsqueeze(0)
            logits, _ = model(x)
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
            output.append(next_id)
            decoded = tok.decode(output)
            if stop in decoded:
                return decoded.split(stop)[0]
    return tok.decode(output).split(stop)[0]


def evaluate_loss(model, samples):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in samples:
            _, loss = model(x, y)
            total += loss.item()
    return total / len(samples)


def evaluate_generation(model, tok, records, block):
    rows = []
    for r in records:
        copied = generate(model, tok, copy_prompt(r), block, '<END_COPY>', 45)
        story = generate(model, tok, story_prompt(r), block, '<END_STORY>', 150)
        a, b = r['characters']
        expected = f'{a}, {b}'
        mentions = {n: bool(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', story, re.I))
                    for n in (a, b)}
        rows.append(dict(required=r, copied=copied, expected_copy=expected,
                         copy_exact=copied.strip() == expected,
                         generated_story=story, story_names=mentions,
                         story_both_names=all(mentions.values()),
                         setting_mentioned=r['setting'].lower() in story.lower(),
                         outcome_mentioned=r['outcome'].lower() in story.lower()))
    return rows


def rates(rows):
    return dict(copy_exact_rate=sum(r['copy_exact'] for r in rows)/len(rows),
                story_both_names_rate=sum(r['story_both_names'] for r in rows)/len(rows),
                setting_rate=sum(r['setting_mentioned'] for r in rows)/len(rows),
                outcome_rate=sum(r['outcome_mentioned'] for r in rows)/len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v12_grounding.pt')
    ap.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    ap.add_argument('--save-checkpoint', default='checkpoints/storyai_v12_character_copy.pt')
    ap.add_argument('--output', default='output/v12_character_copy')
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--copy-weight', type=float, default=3.0)
    ap.add_argument('--train-size', type=int, default=1200)
    ap.add_argument('--val-size', type=int, default=60)
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()
    if args.steps < 1 or args.train_size < 1 or args.val_size < 1 or args.copy_weight <= 0:
        ap.error('steps, sizes and copy-weight must be positive')
    if Path(args.save_checkpoint).resolve() == Path(args.checkpoint).resolve():
        ap.error('Refusing to overwrite input checkpoint')
    if Path(args.save_checkpoint).resolve() == Path('checkpoints/storyai_v8_best.pt').resolve():
        ap.error('Refusing to overwrite V8')
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok_path = Path(args.tokenizer)
    tok = BPETokenizer.load(str(tok_path))
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('tokenizer_sha256') and ckpt['tokenizer_sha256'] != sha:
        raise ValueError('Checkpoint tokenizer SHA256 does not match')
    cfg = ckpt['config']
    cfg = GPTConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = GPT(cfg)
    model.load_state_dict(ckpt['model'], strict=True)

    train = [make_record(rng, NAMES[:40]) for _ in range(args.train_size)]
    val = [make_record(rng, NAMES[40:]) for _ in range(args.val_size)]
    train_copy = [encode_task(tok, copy_prompt(r), copy_answer(r), cfg.block_size) for r in train]
    train_story = [encode_task(tok, story_prompt(r), story_answer(r), cfg.block_size) for r in train]
    val_copy = [encode_task(tok, copy_prompt(r), copy_answer(r), cfg.block_size) for r in val]
    val_story = [encode_task(tok, story_prompt(r), story_answer(r), cfg.block_size) for r in val]
    print(f'Examples: {len(train)} train, {len(val)} validation; held-out names: {NAMES[40:]}')
    before_loss = dict(copy=evaluate_loss(model, val_copy), story=evaluate_loss(model, val_story))
    print('Validation before:', before_loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    for step in range(1, args.steps + 1):
        idx = rng.randrange(len(train))
        model.train()
        opt.zero_grad(set_to_none=True)
        _, cl = model(*train_copy[idx])
        _, sl = model(*train_story[idx])
        loss = args.copy_weight * cl + sl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f'{step}/{args.steps} copy_loss={cl.item():.4f} story_loss={sl.item():.4f}')
    after_loss = dict(copy=evaluate_loss(model, val_copy), story=evaluate_loss(model, val_story))
    print('Validation after:', after_loss)
    rows = evaluate_generation(model, tok, val[:12], cfg.block_size)
    metrics = rates(rows)
    print('Held-out generation:', metrics)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='synthetic_character_copy_pilot_not_novel_quality',
                  input_checkpoint=args.checkpoint, steps=args.steps,
                  train_examples=len(train), val_examples=len(val),
                  held_out_names=NAMES[40:], copy_weight=args.copy_weight,
                  validation_before=before_loss, validation_after=after_loss,
                  metrics=metrics, examples=rows)
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    dest = Path(args.save_checkpoint)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=cfg, tokenizer_sha256=sha,
                    training_stage='character_copy_and_story', steps=args.steps), dest)
    print('Report:', out/'report.json')
    print('Checkpoint:', dest, '(input and V8 unchanged)')
    print('Note: low copy loss alone does not prove correct generated names.')


if __name__ == '__main__':
    main()
