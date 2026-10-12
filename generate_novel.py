"""StoryAI V8 chapter generation using structured instruction prompts."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from auto_genres import predict
from genres import normalize_genres
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def sample(model, tok, prompt, n_tokens, temperature, top_k, repetition_penalty):
    ids = tok.encode(prompt)
    if not ids:
        raise ValueError('Prompt encoded to zero tokens')
    device = next(model.parameters()).device
    context = ids[-model.config.block_size:]
    generated = []
    # Decode only periodically to avoid decoding the whole sequence every token.
    with torch.inference_mode():
        for step in range(n_tokens):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            scores = logits[0, -1].float().clone()
            if repetition_penalty > 1:
                for token in set(context[-128:]):
                    scores[token] = (scores[token] / repetition_penalty
                                     if scores[token] >= 0 else scores[token] * repetition_penalty)
            scores /= temperature
            if top_k > 0:
                threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                scores[scores < threshold] = -float('inf')
            token = int(torch.multinomial(F.softmax(scores, dim=-1), 1).item())
            generated.append(token)
            context = (context + [token])[-model.config.block_size:]
            if (step + 1) % 8 == 0 and '<END_STORY>' in tok.decode(generated):
                break
    return tok.decode(generated).split('<END_STORY>', 1)[0].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--genres', nargs='+')
    p.add_argument('--foundation', help='Editable JSON from story_foundation.py')
    p.add_argument('--title', default='Untitled')
    p.add_argument('--premise', default='')
    p.add_argument('--chapters', type=int, default=1)
    p.add_argument('--tokens-per-chapter', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--repetition-penalty', type=float, default=1.12)
    p.add_argument('--outline-file', help='One chapter beat per line')
    p.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    p.add_argument('--tokenizer', default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output', default='output/novel_v8.txt')
    args = p.parse_args()

    if args.chapters < 1 or args.tokens_per_chapter < 1 or args.temperature <= 0 or args.repetition_penalty < 1:
        p.error('Invalid chapter count, token count, temperature, or repetition penalty')

    foundation = json.loads(Path(args.foundation).read_text(encoding='utf-8')) if args.foundation else None
    title = foundation.get('title', args.title) if foundation else args.title
    premise = foundation.get('premise', args.premise) if foundation else args.premise
    if not premise:
        p.error('Supply --premise or --foundation')
    if args.genres:
        genre_values = args.genres
    elif foundation and foundation.get('genres'):
        genre_values = foundation['genres']
    else:
        predicted = predict(title + ' ' + premise)
        genre_values = predicted[0] if isinstance(predicted, tuple) else predicted
        if isinstance(genre_values, str):
            genre_values = [genre_values]
    genres = normalize_genres(genre_values)
    if foundation and foundation.get('chapters'):
        args.chapters = len(foundation['chapters'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer_path = Path(args.tokenizer)
    if checkpoint.get('tokenizer_sha256'):
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if digest != checkpoint['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint')
    tok = BPETokenizer.load(tokenizer_path)
    model = GPT(GPTConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    plans = []
    if args.outline_file:
        plans = [s.strip() for s in Path(args.outline_file).read_text(encoding='utf-8').splitlines() if s.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = ''
    with output.open('w', encoding='utf-8') as f:
        f.write(f"{title}\nGenres: {', '.join(genres)}\n\n")
        if foundation and foundation.get('introduction_style') == 'profiles':
            f.write('CHARACTERS\n\n')
            for character in foundation.get('characters', []):
                f.write(f"{character['name']} ({character.get('role', 'character')})\n")
                for key in ('age', 'personality', 'goal', 'backstory'):
                    if character.get(key):
                        f.write(f"  {key}: {character[key]}\n")
                f.write('\n')
        if foundation and foundation.get('introduction_style') == 'prologue':
            f.write(f'PROLOGUE\n\n{premise}\n\n')
        for i in range(1, args.chapters + 1):
            chapter = foundation['chapters'][i - 1] if foundation and foundation.get('chapters') else {}
            if chapter:
                plan = f"{chapter.get('objective', '')}. Events: {', '.join(chapter.get('events', []))}"
            elif i <= len(plans):
                plan = plans[i - 1]
            else:
                plan = 'Begin the story with a vivid scene.' if i == 1 else 'Continue the story.'
            cast = []
            if foundation:
                selected = set(chapter.get('characters', [])) if chapter else set()
                for character in foundation.get('characters', []):
                    if not selected or character.get('name') in selected:
                        cast.append(f"{character.get('name', '')}: {character.get('role', '')}; {character.get('goal', '')}")
            instruction = (
                f"Write a {', '.join(genres)} story as narrative prose.\n"
                f"Title: {title}\nPremise: {premise}\n"
                + (f"Characters: {' | '.join(cast)}\n" if cast else '')
                + f"Chapter {i}: {plan}\n"
                + (f"Previously: {previous}\n" if previous else '')
                + 'Write only the scene, not a character profile or metadata.'
            )
            prompt = f'<INSTRUCTION>\n{instruction}\n<STORY>\n'
            prose = sample(model, tok, prompt, args.tokens_per_chapter,
                           args.temperature, args.top_k, args.repetition_penalty)
            prose = re.sub(r'<(?:INSTRUCTION|STORY|END_STORY)>', '', prose).strip()
            chapter_title = chapter.get('title', f'Chapter {i}')
            f.write(f'{chapter_title}\n\n{prose}\n\n')
            f.flush()
            previous = prose[-160:]
            print(f'Wrote chapter {i}/{args.chapters}', flush=True)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
