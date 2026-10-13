"""Generate from an existing scene prompt using an unchanged StoryAI checkpoint."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='output/v12_scene_plan/scene_prompt.txt')
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    ap.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    ap.add_argument('--output', default='output/v12_scene_plan/v8_scene_generated.txt')
    ap.add_argument('--max-new-tokens', type=int, default=350)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.temperature <= 0 or args.top_k < 1 or args.max_new_tokens < 1:
        ap.error('temperature, top-k, and max-new-tokens must be positive')
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    prompt = Path(args.prompt).read_text(encoding='utf-8')
    if not prompt.rstrip().endswith('<STORY>'):
        raise ValueError('Prompt must end with <STORY> (no reference story appended)')
    if '<END_STORY>' in prompt:
        raise ValueError('Prompt already contains <END_STORY>; refusing potential target leakage')
    if not prompt.endswith('\n'):
        prompt += '\n'
    tok_path = Path(args.tokenizer)
    tokenizer = BPETokenizer.load(str(tok_path))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected = ckpt.get('tokenizer_sha256')
    actual = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    if expected and expected != actual:
        raise ValueError('Tokenizer SHA256 mismatch')
    cfg = ckpt['config']
    if isinstance(cfg, dict):
        cfg = GPTConfig(**cfg)
    model = GPT(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) >= cfg.block_size:
        raise ValueError(f'Prompt too long: {len(prompt_ids)} >= {cfg.block_size}')
    ids = list(prompt_ids)
    eos = tokenizer.encode('<END_STORY>')
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long)
            logits, _ = model(x)
            scores = logits[0, -1].float() / args.temperature
            k = min(args.top_k, scores.numel())
            cutoff = torch.topk(scores, k).values[-1]
            scores[scores < cutoff] = -float('inf')
            token = int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())
            ids.append(token)
            if eos and ids[-len(eos):] == eos:
                break
    generated = tokenizer.decode(ids[len(prompt_ids):]).split('<END_STORY>', 1)[0].strip()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding='utf-8')
    names = []
    match = re.search(r'^Characters:\s*(.+)$', prompt, re.M)
    if match:
        names = [n.strip() for n in match.group(1).split(',') if n.strip()]
    report = {'checkpoint': args.checkpoint, 'prompt': args.prompt,
              'generated_tokens': len(ids) - len(prompt_ids),
              'character_mentions': {n: bool(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', generated, re.I)) for n in names},
              'note': 'Character mentions are lexical checks; assess setting, events and coherence manually.'}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', output)
    print(json.dumps(report['character_mentions'], indent=2))
    print('Original checkpoint unchanged.')


if __name__ == '__main__':
    main()
