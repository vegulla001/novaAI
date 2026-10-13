"""CPU-only V12 plan+story supervision experiment. Does not overwrite checkpoints."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from training.v12.plan_training import build_example, instruction_for, plan_for


def records(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def losses(model, samples):
    model.eval()
    totals = [0.0, 0.0]
    with torch.no_grad():
        for x, plan_y, story_y in samples:
            totals[0] += float(model(x, plan_y)[1])
            totals[1] += float(model(x, story_y)[1])
    return {'plan': totals[0] / len(samples), 'story': totals[1] / len(samples)}


def sample_completion(model, tokenizer, prefix_ids, context, max_new, temperature, top_k):
    ids = list(prefix_ids)
    generated = []
    end_ids = tokenizer.encode('<END_STORY>')
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            idx = torch.tensor(ids[-context:], dtype=torch.long).unsqueeze(0)
            logits, _ = model(idx)
            next_logits = logits[0, -1].float() / temperature
            k = min(top_k, next_logits.numel())
            threshold = torch.topk(next_logits, k).values[-1]
            next_logits[next_logits < threshold] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            token = int(torch.multinomial(probs, 1).item())
            ids.append(token)
            generated.append(token)
            if len(generated) >= len(end_ids) and generated[-len(end_ids):] == end_ids:
                break
    return tokenizer.decode(generated).split('<END_STORY>')[0].strip()


def generate_story(model, tokenizer, record, context, max_new, temperature, top_k):
    # Use the VERIFIED plan rather than an unreliable generated plan.
    # Segment encoding must match build_example() exactly.
    prefix = (tokenizer.encode(instruction_for(record))
              + tokenizer.encode(plan_for(record))
              + tokenizer.encode('<STORY>\n'))
    if len(prefix) >= context:
        raise ValueError('Prompt exceeds model context')
    return sample_completion(model, tokenizer, prefix, context, max_new, temperature, top_k)


def character_mentions(story, record):
    return {name: bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', story, re.I))
            for name in record['characters']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    parser.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    parser.add_argument('--data-dir', default='dataset/v12/pilot_expanded')
    parser.add_argument('--output-dir', default='output/v12_plan_story_experiment')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--plan-weight', type=float, default=1.0)
    parser.add_argument('--story-weight', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0 or args.temperature <= 0 or args.top_k < 1:
        parser.error('steps, lr, temperature and top-k must be positive')
    if args.plan_weight < 0 or args.story_weight < 0 or args.plan_weight + args.story_weight == 0:
        parser.error('loss weights must be nonnegative with positive sum')
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected = ckpt.get('tokenizer_sha256')
    if expected and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Tokenizer SHA256 mismatch')
    config = ckpt['config']
    if isinstance(config, dict):
        config = GPTConfig(**config)
    model = GPT(config)
    model.load_state_dict(ckpt['model'], strict=True)
    train = records(Path(args.data_dir) / 'train.jsonl')
    val = records(Path(args.data_dir) / 'val.jsonl')
    if not train or not val:
        raise ValueError('Empty training or validation split')
    if {r['source_family'] for r in train} & {r['source_family'] for r in val}:
        raise ValueError('Training/validation source-family overlap')
    train_samples = [build_example(r, tokenizer, config.block_size) for r in train]
    val_samples = [build_example(r, tokenizer, config.block_size) for r in val]
    print('PASS: shifted targets, separate plan/story masks, verified-plan generation')
    print(f'Train: {len(train)} | Validation: {len(val)} | Test: untouched')
    before_loss = losses(model, val_samples)
    print('Validation BEFORE:', before_loss)
    before = []
    for r in val:
        s = generate_story(model, tokenizer, r, config.block_size, args.max_new_tokens, args.temperature, args.top_k)
        before.append({'title': r['title'], 'generated_story': s, 'character_mentions': character_mentions(s, r)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        model.train()
        x, plan_y, story_y = train_samples[(step - 1) % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        _, lp = model(x, plan_y)
        _, ls = model(x, story_y)
        loss = args.plan_weight * lp + args.story_weight * ls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f'Step {step}/{args.steps} | plan={lp.item():.4f} story={ls.item():.4f}')
    after_loss = losses(model, val_samples)
    print('Validation AFTER:', after_loss)
    after = []
    for r in val:
        s = generate_story(model, tokenizer, r, config.block_size, args.max_new_tokens, args.temperature, args.top_k)
        after.append({'title': r['title'], 'generated_story': s, 'character_mentions': character_mentions(s, r)})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'small_pilot_not_production', 'train_examples': len(train), 'val_examples': len(val),
              'steps': args.steps, 'lr': args.lr, 'loss_weights': {'plan': args.plan_weight, 'story': args.story_weight},
              'validation_before': before_loss, 'validation_after': after_loss,
              'generation_uses_verified_plan': True, 'before': before, 'after': after,
              'manual_review_required': ['setting', 'objective', 'obstacle', 'outcome', 'coherence']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for phase, items in [('before', before), ('after', after)]:
        for i, item in enumerate(items, 1):
            slug = re.sub(r'[^a-z0-9]+', '_', item['title'].lower()).strip('_')
            (out / f'{phase}_{i}_{slug}.txt').write_text(item['generated_story'], encoding='utf-8')
    print('Saved:', out / 'report.json')
    print('No model checkpoint saved. V8 unchanged.')


if __name__ == '__main__':
    main()
