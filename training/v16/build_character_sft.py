#!/usr/bin/env python3
"""Build character-conditioned SFT from existing fiction continuations and controlled examples.

No fabricated labels are claimed human-reviewed. Validation uses held-out template/name
combinations; novel-coherence needs separate human evaluation. All targets are causal LM
response-only masked (-100). The original SFT dataset remains unchanged.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

HEADER = 'Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'
STOP = set(('The A An And But Or If When While She He His Her They Them It Its I We You My Your Our This That These Those '
            'Chapter Story Continuation Mr Mrs Miss Dr Sir Lady Lord King Queen God Yes No Oh Ah Well Why What Where '
            'There Then After Before Now Today Tomorrow Monday Tuesday Wednesday Thursday Friday Saturday Sunday '
            'January February March April May June July August September October November December One Two Three Four '
            'Some All Many Most More First Second Last Next Old Young Little Great').split())
NAME_RE = re.compile(r'(?<![\w])([A-Z][a-z]{2,}(?:[-\'][A-Z][a-z]+)?)(?![\w])')

def candidates(text):
    return {n for n in NAME_RE.findall(text) if n not in STOP and not n.endswith('ly')}

def shared_names(before, after):
    names = candidates(before) & candidates(after)
    # Try to favor recurring names, not isolated capitalized prose words.
    return sorted(names, key=lambda n: (-len(re.findall(r'\b'+re.escape(n)+r'\b', after)), n))[:3]

def encode_row(tok, prompt, completion, max_tokens):
    # Encode the assembled string in one call: avoids BPE boundary drift.
    # Delineate exact prompt boundary by tokenizer's offset mapping.
    full = prompt + completion
    encoded = tok.encode(full, add_special_tokens=False)
    ids = encoded.ids
    offsets = encoded.offsets
    if len(ids) > max_tokens or len(ids) < 10:
        return None
    boundary = len(prompt)
    labels = [-100 if stop <= boundary else ident for ident, (_, stop) in zip(ids, offsets)]
    supervised = sum(v != -100 for v in labels)
    if supervised < 10:
        return None
    return {'input_ids': ids, 'labels': labels}

def real_rows(tok, src_rows, max_tokens):
    outputs = []
    counts = Counter()
    for row in src_rows:
        ids = list(row['input_ids'])
        labels = list(row['labels'])
        if len(ids) != len(labels):
            counts['length_mismatch'] += 1; continue
        supervised = [i for i,v in enumerate(labels) if v != -100]
        if not supervised:
            counts['no_answer'] += 1; continue
        boundary = supervised[0]
        prompt_original = tok.decode(ids[:boundary], skip_special_tokens=True)
        continuation = tok.decode(ids[boundary:], skip_special_tokens=True).strip()
        context = prompt_original.rsplit('Story so far:', 1)[-1].rsplit('Continuation:', 1)[0].strip()
        if len(context) < 100 or len(continuation) < 60:
            counts['too_short'] += 1; continue
        # Original chunks may end in a fragment. Trim only complete final sentences.
        match = list(re.finditer(r'[.!?][\"\u201d\']?(?=\s|$)', continuation))
        if match:
            continuation = continuation[:match[-1].end()].strip()
        else:
            counts['no_sentence_end'] += 1; continue
        names = shared_names(context, continuation)
        if not names:
            counts['no_repeated_names'] += 1; continue
        prompt = (HEADER + 'Characters to preserve: ' + ', '.join(names) + '\n\n'
                  + 'Story so far:\n' + context + '\n\nContinuation:\n')
        example = encode_row(tok, prompt, continuation, max_tokens)
        if example is None:
            counts['token_length_or_boundary'] += 1; continue
        outputs.append(example)
    counts['accepted'] = len(outputs)
    return outputs, dict(counts)

TRAIN_NAMES = ['Arin','Elara','Mira','Jonas','Nila','Dev','Tarin','Sela','Borin','Lysa',
               'Kael','Rina','Orin','Tessa','Lorin','Fara','Darin','Mara','Varek','Iona',
               'Joren','Lina','Ravel','Yara','Kiran','Sorin','Brina','Evan','Cora','Milan']
VAL_NAMES = ['Aven','Ilyra','Rovan','Eska','Tovin','Naera','Belen','Osric','Ysola','Damar']
SETTINGS = [
    ('abandoned castle','northern gate','a sealed letter','the old warning'),
    ('stormy harbor','empty pier','a red scarf','the missing boat'),
    ('silent library','locked tunnel','a torn map','the midnight footsteps'),
    ('snow-covered village','old bridge','a brass compass','the river crossing'),
    ('forest campsite','eastern trail','a small lantern','the distant signal'),
    ('mountain watchtower','broken staircase','a leather journal','the approaching storm'),
    ('desert outpost','western archway','a silver key','the hidden passage'),
    ('market square','stone fountain','a folded note','the urgent meeting'),
]

def synthetic_rows(tok, total, names, max_tokens, seed):
    rng = random.Random(seed)
    records = []
    seen = set()
    attempts = 0
    while len(records) < total and attempts < total * 20:
        attempts += 1
        a,b = rng.sample(names, 2)
        place, landmark, item, mystery = rng.choice(SETTINGS)
        variation = rng.randrange(6)
        scene = (f'{a} and {b} reached the {place} before dawn. '
                 f'{a} carried {item}, while {b} watched the {landmark}. '
                 f'They had promised not to approach the {landmark} until they understood {mystery}. '
                 f'Suddenly, {b} heard footsteps nearby.')
        if variation == 0:
            response = f'{a} stopped beside {b}. "Stay close," {a} whispered. {b} nodded and pointed toward the {landmark}. Neither moved closer.'
        elif variation == 1:
            response = f'{b} touched {a}\'s sleeve. "Did you hear that?" {b} asked. {a} held {item} tightly and looked toward the {landmark}.'
        elif variation == 2:
            response = f'{a} glanced at {b}, then at the {landmark}. "We should keep our promise," {a} said. {b} agreed, and they waited together.'
        elif variation == 3:
            response = f'{b} stepped back as {a} raised {item}. "Someone is there," {b} said. {a} remained beside {b} instead of approaching the {landmark}.'
        elif variation == 4:
            response = f'{a} asked {b} to listen carefully. {b} heard the footsteps again. Together, {a} and {b} moved away from the {landmark}.'
        else:
            response = f'"Not yet," {a} told {b}. {b} looked toward the {landmark}, then followed {a} to a safer place. The mystery would have to wait.'
        key = (a,b,place,variation)
        if key in seen:
            continue
        seen.add(key)
        prompt = (HEADER + f'Characters to preserve: {a}, {b}\n\n'
                  f'Story so far:\n{scene}\n\nContinuation:\n')
        row = encode_row(tok, prompt, response, max_tokens)
        if row: records.append(row)
    if len(records) < total:
        raise RuntimeError(f'Only {len(records)} unique synthetic examples available; requested {total}')
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', default='dataset/v15_prose_expanded/continuation_sft')
    ap.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    ap.add_argument('--out-dir', default='dataset/v16_character_retention_sft')
    ap.add_argument('--train-synthetic', type=int, default=3000)
    ap.add_argument('--val-synthetic', type=int, default=300)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    source = Path(a.source_dir)
    old_train = torch.load(source/'train_sft.pt', map_location='cpu', weights_only=True)
    old_val = torch.load(source/'val_sft.pt', map_location='cpu', weights_only=True)
    train_real, train_stats = real_rows(tok, old_train, a.max_tokens)
    val_real, val_stats = real_rows(tok, old_val, a.max_tokens)
    train = train_real + synthetic_rows(tok, a.train_synthetic, TRAIN_NAMES, a.max_tokens, a.seed)
    val = val_real + synthetic_rows(tok, a.val_synthetic, VAL_NAMES, a.max_tokens, a.seed+1)
    random.Random(a.seed).shuffle(train)
    random.Random(a.seed+1).shuffle(val)
    if len(train) < 500 or len(val) < 50:
        raise RuntimeError('Insufficient valid training or validation rows')
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(train, out/'train_sft.pt')
    torch.save(val, out/'val_sft.pt')
    report = {'train':len(train), 'val':len(val), 'real_train':len(train_real),
              'real_val':len(val_real), 'controlled_train':a.train_synthetic,
              'controlled_val':a.val_synthetic, 'train_stats':train_stats,
              'val_stats':val_stats,
              'caveat':'Automatically selected fiction names are heuristic, not human verified. Controlled examples are synthetic; do not infer novel coherence from heldout loss.'}
    (out/'build_report_character.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CHARACTER_SFT_BUILT', json.dumps(report,indent=2))
if __name__ == '__main__':
    main()
