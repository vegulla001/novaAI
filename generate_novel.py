"""Experimental plain-text chapter generation from a scratch-trained StoryAI model."""
import argparse
import json
import hashlib
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from genres import normalize_genres, genre_prefix
from auto_genres import predict
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

TAGS = re.compile(r"<GENRE_[A-Z_]+>|<STORY>", re.I)

def sample(model, tok, prompt, n_tokens, temperature, top_k, repetition_penalty):
    ids = tok.encode(prompt)
    if not ids:
        raise ValueError("Prompt is empty")
    context = ids[-model.config.block_size:]
    generated = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for _ in range(n_tokens):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            scores = logits[0, -1].float()
            if repetition_penalty > 1:
                for token in set(context[-128:]):
                    scores[token] = scores[token] / repetition_penalty if scores[token] >= 0 else scores[token] * repetition_penalty
            scores = scores / temperature
            if top_k > 0:
                threshold = torch.topk(scores, min(top_k, len(scores))).values[-1]
                scores[scores < threshold] = -float("inf")
            token = int(torch.multinomial(F.softmax(scores, dim=-1), 1).item())
            context = (context + [token])[-model.config.block_size:]
            generated.append(token)
    return tok.decode(generated)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genres", nargs="+")
    p.add_argument("--foundation", help="Editable JSON from story_foundation.py")
    p.add_argument("--title", default="Untitled")
    p.add_argument("--premise", default="")
    p.add_argument("--chapters", type=int, default=3)
    p.add_argument("--tokens-per-chapter", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.12)
    p.add_argument("--outline-file", help="UTF-8 text file, one chapter beat per line")
    p.add_argument("--checkpoint", default="checkpoints/storyai_v5_best.pt")
    p.add_argument("--tokenizer", default="dataset/processed/tokenizer.json")
    p.add_argument("--output", default="output/novel.txt")
    args = p.parse_args()
    if args.chapters < 1 or args.tokens_per_chapter < 1 or args.temperature <= 0 or args.repetition_penalty < 1:
        p.error("Invalid chapter count, token count, temperature, or repetition penalty")
    foundation = json.loads(Path(args.foundation).read_text(encoding='utf-8')) if args.foundation else None
    title = foundation['title'] if foundation else args.title
    premise = foundation['premise'] if foundation else args.premise
    genres = normalize_genres(args.genres or (foundation['genres'] if foundation else predict(title+' '+premise)[0]))
    if not premise: p.error('Supply --premise or --foundation')
    if foundation: args.chapters = len(foundation['chapters'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tok_path = Path(args.tokenizer)
    if 'tokenizer_sha256' in checkpoint:
        digest = hashlib.sha256(tok_path.read_bytes()).hexdigest()
        if checkpoint['tokenizer_sha256'] != digest:
            raise ValueError('Tokenizer does not match checkpoint. Restore the original tokenizer.')
    tok = BPETokenizer.load(tok_path)
    model = GPT(GPTConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    plans = []
    if args.outline_file:
        plans = [x.strip() for x in Path(args.outline_file).read_text(encoding='utf-8').splitlines() if x.strip()]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write(f"{title}\nGenres: {', '.join(genres)}\n\n")
        if foundation and foundation.get('introduction_style') == 'profiles':
            f.write('CHARACTERS\n\n')
            for char in foundation.get('characters', []):
                f.write(f"{char['name']} ({char.get('role','character')})\n")
                for key in ('age','personality','goal','backstory'):
                    if char.get(key): f.write(f"  {key}: {char[key]}\n")
                f.write('\n')
        if foundation and foundation.get('introduction_style') == 'prologue':
            f.write(f'PROLOGUE\n\n{premise}\n\n')
        previous = ''
        for i in range(1, args.chapters + 1):
            chapter = foundation['chapters'][i-1] if foundation else None
            plan = (f"{chapter.get('objective','')}. Events: {', '.join(chapter.get('events',[]))}" if chapter else (plans[i-1] if i <= len(plans) else f'Continue chapter {i}.'))
            cast = ', '.join(chapter.get('characters', [])) if chapter else ''
            if foundation and chapter:
                cast_details = []
                for character in foundation.get('characters', []):
                    if character['name'] in chapter.get('characters', []):
                        cast_details.append(f"{character['name']}: {character.get('role','')}; {character.get('goal','')}")
                cast = '; '.join(cast_details)
            prompt = genre_prefix(genres) + f'{premise}\nCharacters: {cast}\nChapter {i}: {plan}\n'
            if previous:
                prompt += 'Previously: ' + previous + '\n'
            raw = sample(model, tok, prompt, args.tokens_per_chapter, args.temperature, args.top_k, args.repetition_penalty)
            prose = TAGS.sub('', raw).strip()
            prose = re.sub(r'\n{4,}', '\n\n', prose)
            f.write(f"{chapter.get('title',f'Chapter {i}') if chapter else f'Chapter {i}'}\n\n{prose}\n\n")
            f.flush()
            previous = prose[-120:]
            print(f'Wrote chapter {i}/{args.chapters}')
    print(f'Saved {path}. Note: small models may still produce incoherent prose.')

if __name__ == '__main__':
    main()
