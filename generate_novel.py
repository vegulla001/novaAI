"""Write a multi-chapter draft to a Markdown file using genre-conditioned StoryAI.

Each chapter uses the supplied outline plus the previous chapter's ending.
This is an experimental draft generator, not a guarantee of coherent full novels.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from genres import genre_prefix, normalize_genres
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--genres', nargs='+', required=True, help='e.g. ACTION ADVENTURE')
    p.add_argument('--title', default='Untitled Adventure')
    p.add_argument('--premise', required=True)
    p.add_argument('--outline-file', help='Optional UTF-8 text: one chapter outline per line')
    p.add_argument('--chapters', type=int, default=8)
    p.add_argument('--tokens-per-chapter', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--checkpoint', default='checkpoints/storyai_v5_best.pt')
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--output', default='output/novel.md')
    args = p.parse_args()
    if args.chapters < 1 or args.tokens_per_chapter < 1: p.error('chapters and tokens-per-chapter must be positive')
    genres = normalize_genres(args.genres)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer_path = Path(args.tokenizer)
    if 'tokenizer_sha256' in ckpt:
        actual = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if actual != ckpt['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint; restore the tokenizer used for training')
    tok = BPETokenizer.load(tokenizer_path)
    model = GPT(GPTConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    outlines = []
    if args.outline_file:
        outlines = [line.strip() for line in Path(args.outline_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_ending = ''
    with output.open('w', encoding='utf-8') as out:
        out.write(f'# {args.title}\n\n**Genres:** {", ".join(genres)}\n\n')
        for chapter in range(1, args.chapters + 1):
            outline = outlines[chapter - 1] if chapter <= len(outlines) else f'Continue the story in chapter {chapter}.'
            prompt = (genre_prefix(genres) + f'Title: {args.title}\n'
                      f'Premise: {args.premise}\nChapter {chapter}: {outline}\n'
                      f'Previously: {previous_ending}\n\n')
            # Keep the prefix and outline as much as the context window permits.
            ids = tok.encode(prompt)[-model.config.block_size:]
            if not ids: raise ValueError('Empty encoded prompt')
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            result = model.generate(idx, max_new_tokens=args.tokens_per_chapter,
                                    temperature=args.temperature, top_k=args.top_k)
            body = tok.decode(result[0, len(ids):].tolist()).strip()
            out.write(f'## Chapter {chapter}\n\n{body}\n\n')
            out.flush()
            previous_ending = body[-400:]
            print(f'Chapter {chapter}/{args.chapters} written to {output}', flush=True)
    print(f'Finished draft: {output}')

if __name__ == '__main__':
    main()
