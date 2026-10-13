"""CPU-only StoryAI V12 controlled pilot; preserves original checkpoints."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def records(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_for(r):
    genres = ', '.join(str(g).upper() for g in r['genres'])
    return (f'<INSTRUCTION>\nWrite a {genres} story as narrative prose.\n'
            f'Title: {r["title"]}\nPremise: {r["premise"]}\n'
            f'Characters: {", ".join(r["characters"])}\nChapter 1:\n'
            'Write only the scene as narrative prose.\n<STORY>\n')


def pack(r, tokenizer, max_context):
    prompt_ids = tokenizer.encode(prompt_for(r))
    story_ids = tokenizer.encode(r['story'] + '\n<END_STORY>')
    ids = prompt_ids + story_ids
    # One additional token is needed to build shifted x/y pairs.
    if len(ids) > max_context + 1:
        raise ValueError(f'{r["title"]}: {len(ids)} tokens exceed maximum {max_context + 1}')
    if not prompt_ids or not story_ids:
        raise ValueError(f'{r["title"]}: empty prompt or story')
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    # Position len(prompt_ids)-1 predicts the FIRST story token and is supervised.
    y[:, :len(prompt_ids) - 1] = -1
    supervised = y.ne(-1)
    if not supervised.any():
        raise AssertionError('No supervised story tokens')
    # Every supervised target except the last must equal the NEXT input token.
    assert torch.equal(y[:, :-1][supervised[:, :-1]],
                       x[:, 1:][supervised[:, :-1]]), 'Next-token alignment failed'
    return x, y


def loss_on(model, samples):
    model.eval()
    weighted_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in samples:
            _, loss = model(x, y)
            n = int(y.ne(-1).sum().item())
            weighted_loss += float(loss.item()) * n
            count += n
    return weighted_loss / count


def generate(model, tokenizer, r, context, max_new, temperature, top_k):
    model.eval()
    prompt_ids = tokenizer.encode(prompt_for(r))
    if len(prompt_ids) >= context:
        raise ValueError('Prompt longer than context')
    ids = list(prompt_ids)
    eos = tokenizer.encode('<END_STORY>')
    with torch.no_grad():
        for _ in range(max_new):
            idx = torch.tensor(ids[-context:], dtype=torch.long).unsqueeze(0)
            logits, _ = model(idx)
            next_logits = logits[0, -1].float() / temperature
            k = min(top_k, next_logits.numel())
            threshold = torch.topk(next_logits, k).values[-1]
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))
            probabilities = torch.softmax(next_logits, dim=-1)
            token = int(torch.multinomial(probabilities, 1).item())
            ids.append(token)
            if eos and ids[-len(eos):] == eos:
                break
    return tokenizer.decode(ids[len(prompt_ids):]).split('<END_STORY>')[0].strip()


def score_names(text, r):
    return {name: bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I))
            for name in r['characters']}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    a.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    a.add_argument('--data-dir', default='dataset/v12/pilot_expanded')
    a.add_argument('--output-dir', default='output/v12_controlled_experiment_fixed')
    a.add_argument('--steps', type=int, default=10)
    a.add_argument('--lr', type=float, default=1e-5)
    a.add_argument('--max-new-tokens', type=int, default=180)
    a.add_argument('--seed', type=int, default=42)
    args = a.parse_args()
    if args.steps < 0 or args.lr <= 0:
        raise ValueError('steps must be >= 0 and learning rate > 0')
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected_hash = ckpt.get('tokenizer_sha256')
    actual_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if expected_hash and expected_hash != actual_hash:
        raise ValueError('Tokenizer SHA256 mismatch: refusing experiment')
    config = ckpt['config']
    if isinstance(config, dict):
        config = GPTConfig(**config)
    model = GPT(config)
    model.load_state_dict(ckpt['model'], strict=True)
    train = records(Path(args.data_dir) / 'train.jsonl')
    val = records(Path(args.data_dir) / 'val.jsonl')
    if not train or not val:
        raise ValueError('Empty train/validation split')
    if {r['source_family'] for r in train} & {r['source_family'] for r in val}:
        raise ValueError('Source-family leakage between train and validation')
    train_samples = [pack(r, tokenizer, config.block_size) for r in train]
    val_samples = [pack(r, tokenizer, config.block_size) for r in val]
    print('PASS: shifted next-token targets and story-only masking')
    print('Model loaded. Train:', len(train), 'Validation:', len(val))
    before_loss = loss_on(model, val_samples)
    print(f'Before validation loss: {before_loss:.4f}')
    before = []
    for r in val:
        story = generate(model, tokenizer, r, config.block_size, args.max_new_tokens, .7, 20)
        before.append({'title': r['title'], 'generated_story': story,
                       'character_mentions': score_names(story, r)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = train_samples[(step - 1) % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f'Step {step}/{args.steps} train loss: {loss.item():.4f}')
    after_loss = loss_on(model, val_samples)
    print(f'After validation loss: {after_loss:.4f}')
    after = []
    for r in val:
        story = generate(model, tokenizer, r, config.block_size, args.max_new_tokens, .7, 20)
        after.append({'title': r['title'], 'generated_story': story,
                      'character_mentions': score_names(story, r)})
    result = {'note': 'Small CPU pilot; do not infer production instruction adherence.',
              'train_examples': len(train), 'val_examples': len(val), 'steps': args.steps,
              'lr': args.lr, 'target_alignment': 'shifted_next_token',
              'validation_loss_before': before_loss, 'validation_loss_after': after_loss,
              'before': before, 'after': after,
              'manual_review_required': ['setting', 'objective', 'obstacle', 'outcome', 'coherence']}
    (out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for phase, items in [('before', before), ('after', after)]:
        for idx, item in enumerate(items):
            slug = re.sub(r'[^a-z0-9]+', '_', item['title'].lower()).strip('_')
            (out / f'{phase}_{idx+1}_{slug}.txt').write_text(item['generated_story'], encoding='utf-8')
    print('Saved:', out / 'report.json')
    print('No model checkpoint saved; V8 unchanged.')


if __name__ == '__main__':
    main()
