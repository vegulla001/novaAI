"""Prepare independently framed instruction/story examples (JSONL -> JSONL token ids)."""
import argparse, hashlib, json, random
from pathlib import Path
from tokenizer import BPETokenizer


def format_prompt(row):
    chars = row.get('characters', [])
    names = ', '.join(c if isinstance(c, str) else f"{c.get('name','')} ({c.get('role','character')})" for c in chars)
    return ('<INSTRUCTION>\n'
            f"Title: {row.get('title', 'Untitled')}\n"
            f"Genres: {', '.join(row.get('genres', [row.get('genre', 'fantasy')]))}\n"
            f"Characters: {names}\n"
            f"Setting: {row.get('setting', '')}\n"
            f"Objective: {row.get('objective', '')}\n"
            f"Premise: {row.get('premise', '')}\n"
            '<STORY>\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Curated JSONL; each row has story + instruction fields')
    p.add_argument('--tokenizer', default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output', default='dataset/v12/processed')
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0 < args.val_ratio < 0.5: p.error('--val-ratio must be between 0 and 0.5')
    if args.max_length < 32: p.error('--max-length too small')
    tok = BPETokenizer.load(args.tokenizer)
    examples = []
    for lineno, line in enumerate(Path(args.input).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip(): continue
        row = json.loads(line)
        story = row.get('story', '').strip()
        if not story: raise ValueError(f'Line {lineno}: missing story')
        prompt = format_prompt(row)
        prefix = tok.encode(prompt)
        continuation = tok.encode(story + '\n<END_STORY>\n')
        if len(prefix) + len(continuation) > args.max_length:
            print(f'SKIP line {lineno}: {len(prefix)+len(continuation)} tokens exceeds {args.max_length}')
            continue
        # Targets predict the NEXT token. Mask targets corresponding to instruction tokens.
        ids = prefix + continuation
        labels = [-1] * (len(prefix) - 1) + continuation
        assert len(ids) == len(labels) + 1
        examples.append({'ids': ids, 'labels': labels})
    if len(examples) < 2: raise ValueError('Need at least 2 fitting examples for train/validation split')
    if len(examples) < 20:
        print(f'WARNING: {len(examples)} fitting examples: pipeline smoke test only; DO NOT TRAIN.', flush=True)
    random.Random(args.seed).shuffle(examples)
    n_val = max(1, round(len(examples) * args.val_ratio))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, subset in [('val', examples[:n_val]), ('train', examples[n_val:])]:
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for example in subset: f.write(json.dumps(example) + '\n')
        print(name, len(subset))
    meta = {'demo_only': len(examples) < 20, 'tokenizer_sha256': hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
            'max_length': args.max_length, 'num_examples': len(examples), 'source': args.input}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
