#!/usr/bin/env python3
"""Story AI v16: character-anchored inference without changing model weights.

Guarantees that given character names are present in the *assembled continuation*
by inserting explicit, pre-written anchors before each generation chunk.
It does NOT guarantee the neural network remembers the names by itself, prevents
new characters, or writes coherent prose. Reports forced vs model-written names.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig

SCENES = {
    'castle': {
        'context': 'The rain had not stopped for three days. Arin stood outside the abandoned castle, holding a letter from his missing father. Behind him, Elara approached with a lantern. "Arin," she whispered, "there is something moving beyond the northern gate." Arin unfolded the letter. The final warning was written in his father\'s handwriting: Never open the northern gate.',
        'characters': ['Arin', 'Elara'],
        'anchors': [
            'Arin tightened his grip on the letter. ',
            'Elara raised her lantern toward the northern gate. ',
            'Arin remembered his father\'s warning. ',
            'Elara stood beside him in the rain. ',
        ],
    },
    'harbor': {
        'context': 'At dawn, Mira waited at the harbor for her brother Jonas. He had promised to return before the storm. The fishing boats were empty, and a red scarf lay on the wet pier. Mira recognized it as Jonas\'s scarf. She called his name, but nobody answered.',
        'characters': ['Mira', 'Jonas'],
        'anchors': [
            'Mira picked up Jonas\'s red scarf. ',
            'Mira searched the harbor for Jonas. ',
            'Jonas was still missing, and Mira had no explanation. ',
            'Mira called Jonas\'s name again. ',
        ],
    },
    'library': {
        'context': 'The library was silent when Nila found the torn map. Her friend Dev had hidden it beneath a history book. They had agreed not to enter the old tunnel until sunrise. Outside, the clock struck midnight. Nila heard footsteps in the corridor.',
        'characters': ['Nila', 'Dev'],
        'anchors': [
            'Nila folded the torn map that Dev had hidden. ',
            'Nila listened to the footsteps outside the library. ',
            'Dev had warned Nila to wait until sunrise. ',
            'Nila kept the tunnel map close. ',
        ],
    },
}

HEADER = ('Continue the following fiction scene naturally, preserving its characters, '
          'point of view, and events.\n\nStory so far:\n')


def count_mentions(s, names):
    return {name: len(re.findall(r'(?<!\w)' + re.escape(name) + r'(?!\w)', s, flags=re.I))
            for name in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/v16_novel_writer_01/sft_best.pt')
    p.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--scene', choices=[*SCENES], default='castle')
    p.add_argument('--tokens-per-chunk', type=int, default=60)
    p.add_argument('--chunks', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--output', default='reports/v16_character_generation.json')
    a = p.parse_args()
    if a.chunks < 1 or a.tokens_per_chunk < 1:
        p.error('--chunks and --tokens-per-chunk must be positive')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    tok = Tokenizer.from_file(a.tokenizer)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ck['model_config'])).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    block_size = ck['model_config']['block_size']
    scene = SCENES[a.scene]

    torch.manual_seed(a.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(a.seed)

    prompt = HEADER + scene['context'] + '\n\nContinuation:\n'
    assembled = ''
    parts = []
    for i in range(a.chunks):
        anchor = scene['anchors'][i % len(scene['anchors'])]
        # The anchor is injected as fixed context; the model is only responsible
        # for the suffix. Distinguish these in reports.
        separator = '\n\n' if i else ''
        prefix = prompt + assembled + separator + anchor
        ids = tok.encode(prefix).ids
        if len(ids) >= block_size:
            print(f'Context limit reached before chunk {i + 1}; stopping.', flush=True)
            break
        n = min(a.tokens_per_chunk, block_size - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            y = model.generate(x, max_new_tokens=n, temperature=a.temperature, top_k=a.top_k)
        model_text = tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True)
        assembled += separator + anchor + model_text
        parts.append({'chunk': i + 1, 'forced_anchor': anchor, 'model_generated': model_text,
                      'model_only_mentions': count_mentions(model_text, scene['characters'])})

    report = {
        'checkpoint': a.checkpoint,
        'scene': a.scene,
        'characters': scene['characters'],
        'generated_text_with_forced_anchors': assembled,
        'total_mentions_in_assembled_text': count_mentions(assembled, scene['characters']),
        'model_only_mentions': count_mentions(' '.join(part['model_generated'] for part in parts), scene['characters']),
        'chunks': parts,
        'note': 'Names in forced anchors are supplied by code, not evidence the model learned character retention. This is a runtime continuity workaround.'
    }
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== CHARACTER-ANCHORED CONTINUATION ===\n' + assembled)
    print('\nTOTAL MENTIONS:', report['total_mentions_in_assembled_text'])
    print('MODEL-ONLY MENTIONS:', report['model_only_mentions'])
    print('SAVED', target)


if __name__ == '__main__':
    main()
