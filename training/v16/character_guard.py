"""Fail-closed character-safe Story AI v16 scene generation.

No training is performed. A model is never credited with text produced by the fallback.
Usage: python -m training.v16.character_guard --scene scene.json [--checkpoint ...]
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

# Conservative defaults: unknown capitalized words are treated as potential characters.
NON_NAMES = set('The A An I He She They We It His Her Their Our This That There Then When Where What Who Why How But And Or No Yes Never Chapter Story Continuation Characters Location Rule Scene Keep Do Not Before After Meanwhile At In On Of To From With As For By Under Beyond Outside Inside Behind Toward Through Today Tomorrow Yesterday North South East West'.split())

def mentions(text: str, name: str) -> int:
    return len(re.findall(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, flags=re.I))

def validate(text: str, scene: dict) -> list[str]:
    issues = []
    names = [c['name'] for c in scene['characters']]
    if not text.strip():
        return ['empty output']
    for name in names:
        if not mentions(text, name):
            issues.append('missing character: ' + name)
    if re.search(r'"(?:context_evidence|observed_next_text)"\s*:', text):
        issues.append('extraction JSON contamination')
    if re.search(r'\b(?:CHAPTER\s+[IVXLCDM\d]+)\b', text, re.I):
        issues.append('chapter-heading contamination')
    # Treat new title-case names after punctuation as possible unknown characters.
    # No model-only claim of name filtering perfection: names can be ordinary words.
    for token in set(re.findall(r'\b[A-Z][a-z]{2,}\b', text)):
        if token not in NON_NAMES and token not in names and token not in scene.get('allowed_names', []):
            issues.append('unapproved capitalized word (possible name): ' + token)
    for banned in scene.get('forbidden_phrases', []):
        if banned.casefold() in text.casefold():
            issues.append('forbidden phrase: ' + banned)
    return issues

def safe_fallback(scene: dict) -> str:
    """Human-authored, editable outline beats; never present this as model output."""
    beats = scene.get('safe_beats', [])
    if not beats:
        raise ValueError('scene.json must provide nonempty safe_beats for fail-closed fallback')
    return '\n\n'.join(beats)

def try_model(scene: dict, checkpoint_path: str, tokenizer_path: str, device: str, seed: int, max_tokens: int) -> str:
    import torch
    from tokenizers import Tokenizer
    from training.v15.model import GPT, GPTConfig
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    tok = Tokenizer.from_file(tokenizer_path)
    people = ', '.join(c['name'] for c in scene['characters'])
    prompt = (
        'Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'
        f'Characters to preserve: {people}\n\n'
        f'Story so far:\n{scene["context"]}\n\nContinuation:\n'
    )
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device=device, dtype=torch.long)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=0.7, top_k=40)
    return tok.decode(y[0].tolist()[len(ids):], skip_special_tokens=True).strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--checkpoint', default=None, help='Optional; tries model once, then fails closed')
    p.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=150)
    p.add_argument('--output', default='reports/v16_character_guard.json')
    a = p.parse_args()
    scene = json.loads(Path(a.scene).read_text(encoding='utf-8'))
    if len(scene['characters']) < 2:
        raise ValueError('scene must declare at least two characters')
    fallback = safe_fallback(scene)
    fallback_problems = validate(fallback, scene)
    if fallback_problems:
        raise ValueError('safe_beats did not pass validation: ' + '; '.join(fallback_problems))
    candidate, candidate_issues = None, ['model not requested']
    if a.checkpoint:
        candidate = try_model(scene, a.checkpoint, a.tokenizer, a.device, a.seed, a.max_new_tokens)
        candidate_issues = validate(candidate, scene)
    accepted = candidate is not None and not candidate_issues
    result = {
        'mode': 'model_accepted' if accepted else 'approved_outline_fallback',
        'generated_by_model': bool(accepted),
        'text': candidate if accepted else fallback,
        'model_candidate': candidate,
        'model_rejection_reasons': candidate_issues,
        'character_mentions': {c['name']: mentions(candidate if accepted else fallback, c['name']) for c in scene['characters']},
        'note': 'Fallback is predetermined scene text, not evidence of model learning. Validation catches visible names, not all semantic contradictions.',
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== ACCEPTED STORY ===\n' + result['text'])
    print('\nMODE:', result['mode'])
    print('MODEL REJECTION REASONS:', candidate_issues)
    print('SAVED', out)

if __name__ == '__main__':
    main()
