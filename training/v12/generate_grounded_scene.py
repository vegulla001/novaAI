"""Grounded scene generation: model drafts are accepted only if all required facts survive.
If the small model fails, return a deterministic, explicitly labeled fallback instead of wrong plot.
"""
import argparse
import hashlib
import re
from pathlib import Path
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from training.v12.train_natural_scenes import generate

C1, C2 = '<CHAR_1>', '<CHAR_2>'

def prompt(setting, goal, obstacle, outcome):
    return ('<INSTRUCTION>\nCharacters: <CHAR_1>, <CHAR_2>\n'
            f'Setting: {setting}\nGoal: {goal}\nObstacle: {obstacle}\nOutcome: {outcome}\n<STORY>\n')

def normalized(s):
    return re.sub(r'\s+', ' ', s.casefold()).strip(' .!?,')

def grounded(raw, setting, goal, obstacle, outcome):
    text = normalized(raw)
    required = [C1.casefold(), C2.casefold(), normalized(setting), normalized(goal),
                normalized(obstacle), normalized(outcome)]
    return all(x in text for x in required) and '<instruction>' not in text and '<story>' not in text

def fallback(setting, goal, obstacle, outcome):
    # Deliberately conservative: no unrequested facts, and explicit grounding of all fields.
    setting = setting.strip().rstrip('.')
    goal = goal.strip().rstrip('.')
    obstacle = obstacle.strip().rstrip('.')
    outcome = outcome.strip().rstrip('.')
    return (f'At {setting}, {C1} and {C2} faced a task neither could abandon: {goal}.\n\n'
            f'“We need to act now,” {C1} said.\n\n'
            f'{C2} moved first, but {obstacle} forced them to stop and reconsider. '
            f'{C1} stayed beside {C2}, working through the danger instead of leaving the task unfinished.\n\n'
            f'Together they found a way to {outcome}. The immediate danger had passed, '
            f'and both knew how narrowly they had succeeded.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--characters', nargs=2, required=True, metavar=('FIRST', 'SECOND'))
    p.add_argument('--setting', required=True)
    p.add_argument('--goal', required=True)
    p.add_argument('--obstacle', required=True)
    p.add_argument('--outcome', required=True)
    p.add_argument('--checkpoint', default='checkpoints/storyai_v12_v8_authored.pt')
    p.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    p.add_argument('--candidates', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.candidates < 0: p.error('--candidates must be >= 0')
    if any('<' in n or '>' in n or '\n' in n for n in args.characters): p.error('Invalid character name')
    torch.manual_seed(args.seed)
    tok_path = Path(args.tokenizer)
    tok = BPETokenizer.load(str(tok_path))
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256') and ck['tokenizer_sha256'] != sha: raise ValueError('Tokenizer mismatch')
    cfg = GPTConfig(**ck['config']) if isinstance(ck['config'], dict) else ck['config']
    model = GPT(cfg)
    model.load_state_dict(ck['model'], strict=True)
    ptxt = prompt(args.setting, args.goal, args.obstacle, args.outcome)
    selected = None
    for _ in range(args.candidates):
        raw = generate(model, tok, ptxt, cfg.block_size, max_new=220)
        if grounded(raw, args.setting, args.goal, args.obstacle, args.outcome):
            selected = raw
            break
    source = 'MODEL_DRAFT' if selected is not None else 'GROUNDED_FALLBACK'
    if selected is None:
        selected = fallback(args.setting, args.goal, args.obstacle, args.outcome)
    # Do not allow arbitrary names to be generated as replacements for the two roles.
    result = selected.replace(C1, args.characters[0]).replace(C2, args.characters[1])
    print(f'[{source}]')
    print(result)

if __name__ == '__main__': main()
