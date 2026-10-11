"""Generate an experimental genre-conditioned novel draft as UTF-8 plain text."""
import argparse
import hashlib
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from genres import genre_prefix, normalize_genres
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

TAGS = re.compile(r'<(?:GENRE_[A-Z_]+|STORY)>', re.I)

def clean(text):
    text = TAGS.sub('', text)
    text = re.sub(r'(?im)^\s*#{0,3}\s*Chapter\s+\d+\s*$', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

@torch.inference_mode()
def sample(model, tok, prompt, count, temperature, top_k, penalty):
    context = tok.encode(prompt)[-model.config.block_size:]
    if not context:
        raise ValueError('Prompt produced no tokens')
    generated = []
    device = next(model.parameters()).device
    for _ in range(count):
        x = torch.tensor([context], device=device, dtype=torch.long)
        logits, _ = model(x)
        logits = logits[0, -1].float()
        for token_id in set(context[-96:]):
            logits[token_id] = logits[token_id] / penalty if logits[token_id] > 0 else logits[token_id] * penalty
        logits = logits / temperature
        if top_k:
            threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits[logits < threshold] = -float('inf')
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
        generated.append(next_id)
        context = (context + [next_id])[-model.config.block_size:]
    return tok.decode(generated)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--genres', nargs='+', required=True)
    p.add_argument('--title', default='Untitled Novel')
    p.add_argument('--premise', required=True)
    p.add_argument('--outline-file')
    p.add_argument('--chapters', type=int, default=3)
    p.add_argument('--tokens-per-chapter', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--repetition-penalty', type=float, default=1.12)
    p.add_argument('--checkpoint', default='checkpoints/storyai_v5_best.pt')
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--output', default='output/novel.txt')
    a = p.parse_args()
    if a.chapters < 1 or a.tokens_per_chapter < 1 or a.temperature <= 0 or a.repetition_penalty < 1:
        p.error('chapters and tokens-per-chapter must be positive; temperature > 0; repetition-penalty >= 1')
    genres = normalize_genres(a.genres)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved = torch.load(a.checkpoint, map_location=device, weights_only=True)
    tok_path = Path(a.tokenizer)
    if 'tokenizer_sha256' in saved:
        if hashlib.sha256(tok_path.read_bytes()).hexdigest() != saved['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint')
    else:
        print('WARNING: checkpoint has no tokenizer fingerprint; verify it matches your training tokenizer.')
    tok = BPETokenizer.load(tok_path)
    model = GPT(GPTConfig(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    outlines = []
    if a.outline_file:
        outlines = [s.strip() for s in Path(a.outline_file).read_text(encoding='utf-8').splitlines() if s.strip()]
    outpath = Path(a.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    previous = ''
    with outpath.open('w', encoding='utf-8') as f:
        f.write(f'{a.title}\nGenres: {", ".join(genres)}\n\n')
        for chapter in range(1, a.chapters + 1):
            plan = outlines[chapter - 1] if chapter <= len(outlines) else f'Continue the story, chapter {chapter}.'
            prompt = (genre_prefix(genres) + f'{a.title}. {a.premise}\n'
                      + f'Chapter {chapter}: {plan}\n'
                      + (f'Previously: {previous}\n' if previous else ''))
            raw = sample(model, tok, prompt, a.tokens_per_chapter,
                         a.temperature, a.top_k, a.repetition_penalty)
            body = clean(raw)
            f.write(f'Chapter {chapter}\n\n{body}\n\n')
            f.flush()
            previous = body[-150:]
            print(f'Wrote chapter {chapter}/{a.chapters} to {outpath}', flush=True)
    print(f'Finished: {outpath}')
    print('Note: removing genre tags does not fix an undertrained model\'s grammar or story coherence.')

if __name__ == '__main__':
    main()
