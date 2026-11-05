#!/usr/bin/env python3
"""Compare two StoryAI checkpoints on held-out SFT targets without training.

Uses direct greedy decoding, independent of GPT.generate implementation details.
Reports invalid/skipped examples instead of silently reporting zero cases.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig


def load_model(path, device, tokenizer_hash):
    c = torch.load(path, map_location='cpu', weights_only=False)  # trusted local checkpoint
    if c.get('tokenizer_sha256') != tokenizer_hash:
        raise ValueError(f'Checkpoint/tokenizer mismatch: {path}')
    cfg = GPTConfig(**c['model_config'])
    m = GPT(cfg).to(device)
    m.load_state_dict(c['model'], strict=True)
    m.eval()
    return m, cfg


def row_parts(row, max_tokens):
    ids = [int(x) for x in row['input_ids']]
    labels = [int(x) for x in row['labels']]
    if len(ids) != len(labels):
        return None, 'length_mismatch'
    b = next((i for i, x in enumerate(labels) if x >= 0), None)
    if b is None or b < 1:
        return None, 'no_target_or_prompt'
    target = []
    for v in labels[b:]:
        if v >= 0:
            target.append(v)
        else:
            break
        if len(target) >= max_tokens:
            break
    if not target:
        return None, 'no_supervised_tokens'
    return (ids[:b], target), None


@torch.inference_mode()
def greedy_predict(model, prompt, count, block_size, device):
    seq = torch.tensor([prompt[-block_size:]], dtype=torch.long, device=device)
    produced = []
    for _ in range(count):
        logits, _ = model(seq[:, -block_size:])
        next_token = int(logits[0, -1].argmax().item())
        produced.append(next_token)
        seq = torch.cat([seq, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1)
    return produced


@torch.inference_mode()
def teacher_predict(model, prompt, target, block_size, device):
    # For each ground-truth next token, predict it from the real previous tokens.
    context = prompt[-block_size:]
    result = []
    for truth in target:
        x = torch.tensor([context[-block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        result.append(int(logits[0, -1].argmax().item()))
        context.append(truth)
    return result


def assess(model, cfg, selected, device, prefix_tokens, show):
    counts = {'cases': 0, 'teacher_hits': 0, 'free_hits': 0, 'token_total': 0,
              'first_hits': 0, 'exact_hits': 0}
    skipped = {}
    examples = []
    for index, row in enumerate(selected):
        parts, reason = row_parts(row, prefix_tokens)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        prompt, target = parts
        if not prompt:
            skipped['empty_prompt'] = skipped.get('empty_prompt', 0) + 1
            continue
        predicted = greedy_predict(model, prompt, len(target), cfg.block_size, device)
        teacher = teacher_predict(model, prompt, target, cfg.block_size, device)
        n = len(target)
        counts['cases'] += 1
        counts['token_total'] += n
        free_hits = sum(a == b for a, b in zip(predicted, target))
        counts['free_hits'] += free_hits
        counts['teacher_hits'] += sum(a == b for a, b in zip(teacher, target))
        counts['first_hits'] += (predicted[0] == target[0])
        counts['exact_hits'] += (predicted == target)
        if len(examples) < show:
            examples.append({'selected_index': index,
                             'target_token_ids': target[:12],
                             'generated_token_ids': predicted[:12],
                             'token_agreement': round(free_hits / n, 4)})
    total = counts['token_total']
    cases = counts['cases']
    return {
        'cases_requested': len(selected), 'cases': cases, 'skipped': skipped,
        'teacher_forced_token_accuracy': round(counts['teacher_hits']/total, 4) if total else None,
        'free_run_token_accuracy': round(counts['free_hits']/total, 4) if total else None,
        'first_token_accuracy': round(counts['first_hits']/cases, 4) if cases else None,
        'exact_prefix_accuracy': round(counts['exact_hits']/cases, 4) if cases else None,
        'prefix_tokens': prefix_tokens, 'examples': examples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--dataset', default='dataset/v16_grounded_extractive_sft/val_sft.pt')
    p.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--cases', type=int, default=100)
    p.add_argument('--prefix-tokens', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--output', default='reports/v16_extraction_comparison.json')
    a = p.parse_args()
    if a.cases < 1 or a.prefix_tokens < 1:
        p.error('--cases and --prefix-tokens must be positive')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    hash_ = hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    Tokenizer.from_file(a.tokenizer)
    rows = torch.load(a.dataset, map_location='cpu', weights_only=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError('SFT validation data must be a nonempty list')
    selected = random.Random(a.seed).sample(rows, min(a.cases, len(rows)))
    report = {'dataset': a.dataset, 'device': device, 'seed': a.seed, 'comparisons': {}}
    for ck in a.checkpoints:
        print('EVALUATING', ck, flush=True)
        model, cfg = load_model(ck, device, hash_)
        stats = assess(model, cfg, selected, device, a.prefix_tokens, 3)
        if stats['cases'] == 0:
            raise RuntimeError(f'No valid cases. Skip reasons: {stats["skipped"]}')
        report['comparisons'][ck] = stats
        print(json.dumps(stats, indent=2), flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT', out)


if __name__ == '__main__':
    main()
