"""Generate a short scene with exact user-supplied character names.
Example: python -m training.v12.generate_placeholder_story --characters Rhea Pax --scenario 0
"""
import argparse
from pathlib import Path
from training.v12.train_placeholder_binding import SETTINGS, bind_characters, generate, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v12_placeholder.pt')
    ap.add_argument('--tokenizer', default='dataset/processed/tokenizer.json')
    ap.add_argument('--characters', nargs=2, required=True, metavar=('NAME1', 'NAME2'))
    ap.add_argument('--scenario', type=int, default=0, choices=range(len(SETTINGS)))
    ap.add_argument('--max-new', type=int, default=180)
    args = ap.parse_args()
    model, tok, config, _ = load_model(args.checkpoint, Path(args.tokenizer))
    scenario = dict(zip(('setting', 'objective', 'obstacle', 'outcome'), SETTINGS[args.scenario]))
    raw = generate(model, tok, scenario, config.block_size, args.max_new)
    print(bind_characters(raw, args.characters))
    if '<CHAR_1>' not in raw or '<CHAR_2>' not in raw:
        print('\nWARNING: model omitted a character placeholder; names cannot be guaranteed in this output.')


if __name__ == '__main__':
    main()
