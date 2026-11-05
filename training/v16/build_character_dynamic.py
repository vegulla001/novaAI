#!/usr/bin/env python3
"""Story AI v16: real-fiction name substitution with held-out names.

No hard-coded synthetic prose templates. Creates masked response SFT records
compatible with training.v16.train_character. The selected real-fiction source
names remain heuristic; report is not a claim of semantic validation.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer
from training.v16.build_character_strict import (
    HEADER, SENT_END, real_names, extract_original, encode,
)

# These names occur only in *validation substitutions* and raw-generation tests.
# If they occur naturally in source text, those source records are excluded.
HELDOUT = ('Arin', 'Elara', 'Mira', 'Jonas', 'Nila', 'Dev')
SYLL_A = ('Al','Bel','Cor','Dar','Fen','Gal','Hal','Ira','Jen','Kel','Lor','Mar',
          'Nor','Or','Pel','Quen','Ral','Sel','Tor','Ul','Val','Wen','Yor','Zan')
SYLL_B = ('adin','ara','en','ion','ella','ric','orin','essa','ian','ora','eth',
          'ira','orin','arae','ielle','enor','isa','orinel','alis','evan')
BANNED = set(HELDOUT) | {'Caden','Evan','Davin','Milan','Christian','John','Raoul','Christine','Erik'}
NAME_RE = lambda n: re.compile(r'(?<!\w)' + re.escape(n) + r'(?!\w)')


def make_names():
    pool = list(dict.fromkeys(a+b for a in SYLL_A for b in SYLL_B))
    return [n for n in pool if n not in BANNED and len(n) >= 5]


def split_text(tok, row):
    ids, labels = list(row['input_ids']), list(row['labels'])
    boundary = next((i for i, x in enumerate(labels) if x != -100), None)
    if boundary is None: return None
    return tok.decode(ids[:boundary], skip_special_tokens=True), tok.decode(ids[boundary:], skip_special_tokens=True)


def aligned_record(tok, prompt, answer, max_tokens):
    # Encode prefix + completion separately to avoid a BPE merge across the mask boundary.
    pids = tok.encode(prompt, add_special_tokens=False).ids
    aids = tok.encode(answer, add_special_tokens=False).ids
    if len(pids) < 20 or len(aids) < 20 or len(pids) + len(aids) > max_tokens: return None
    return {'input_ids': pids + aids, 'labels': [-100]*len(pids) + aids}


def clean_source(tok, raw, banned, stats):
    parsed = extract_original(tok, raw)
    if not parsed:
        stats['invalid_source'] += 1; return None
    ctx, answer = parsed
    if len(ctx) < 130 or len(answer) < 100:
        stats['short'] += 1; return None
    ce = list(SENT_END.finditer(ctx))
    ae = list(SENT_END.finditer(answer))
    if not ce or len(ae) < 2:
        stats['boundary'] += 1; return None
    ctx = ctx[:ce[-1].end()].strip()
    answer = answer[ae[0].end():ae[-1].end()].strip()
    if len(ctx) < 120 or len(answer) < 90:
        stats['trim_short'] += 1; return None
    if any(NAME_RE(n).search(ctx) or NAME_RE(n).search(answer) for n in banned):
        stats['heldout_in_source'] += 1; return None
    names = real_names(ctx, answer)
    if len(names) != 2:
        stats['not_two_names'] += 1; return None
    if any(NAME_RE(n).search(ctx) is None or NAME_RE(n).search(answer) is None for n in names):
        stats['name_not_in_both'] += 1; return None
    # Avoid names embedded in one another and original names colliding with substitutions.
    if any(n.lower() == m.lower() for n in names for m in banned):
        stats['name_collision'] += 1; return None
    return ctx, answer, names


def substitute(ctx, answer, original, replacement):
    for old, new in sorted(zip(original, replacement), key=lambda p: -len(p[0])):
        pat = NAME_RE(old)
        ctx = pat.sub(new, ctx)
        answer = pat.sub(new, answer)
    return ctx, answer


def build_split(tok, source, count, variants, pool, max_tokens, seed, heldout=False):
    rng = random.Random(seed)
    indices = list(range(len(source)))
    rng.shuffle(indices)
    records, stats, seen = [], Counter(), set()
    manifest = []
    for idx in indices:
        if len(records) >= count * (variants + 1): break
        sample = clean_source(tok, source[idx], HELDOUT, stats)
        if not sample: continue
        ctx, answer, names = sample
        source_hash = hashlib.sha256((ctx+'\0'+answer).encode('utf8')).hexdigest()
        if source_hash in seen:
            stats['duplicate_source'] += 1; continue
        if any(NAME_RE(n).search(ctx) or NAME_RE(n).search(answer) for n in pool):
            stats['replacement_present_in_original'] += 1; continue
        original_prompt = HEADER+f'Characters to preserve: {names[0]}, {names[1]}\n\nStory so far:\n{ctx}\n\nContinuation:\n'
        orig = aligned_record(tok, original_prompt, answer, max_tokens)
        if orig is None:
            stats['token_limit'] += 1; continue
        transformed = []
        for variation in range(variants):
            r1,r2 = rng.sample(pool,2)
            new_ctx,new_answer = substitute(ctx,answer,names,(r1,r2))
            if not all(NAME_RE(n).search(new_ctx) and NAME_RE(n).search(new_answer) for n in (r1,r2)):
                stats['replacement_failed'] += 1; break
            if any(NAME_RE(n).search(new_ctx) or NAME_RE(n).search(new_answer) for n in names):
                stats['old_name_leaked'] += 1; break
            prompt = HEADER+f'Characters to preserve: {r1}, {r2}\n\nStory so far:\n{new_ctx}\n\nContinuation:\n'
            row = aligned_record(tok,prompt,new_answer,max_tokens)
            if row is None:
                stats['transformed_too_long'] += 1; break
            transformed.append((row,(r1,r2)))
        if len(transformed) != variants: continue
        seen.add(source_hash)
        records.append(orig)
        records.extend(row for row, _ in transformed)
        # Only record hashes and name mappings, never source novel text.
        manifest.append({'source_index':idx,'source_sha256':source_hash,
                         'original_names':names,'replacement_pairs':[list(pair) for _,pair in transformed]})
        stats['source_passages'] += 1
    if len(records) < count * (variants + 1):
        raise RuntimeError(f'Only {len(records)} records built; expected {count*(variants+1)}. '
                           f'Stats={dict(stats)}. Lower --train-sources/--val-sources, do not remove checks silently.')
    rng.shuffle(records)
    stats['records'] = len(records)
    stats['original_records'] = count
    stats['substituted_records'] = count*variants
    return records, manifest, dict(stats)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',default='dataset/v15_prose_expanded/continuation_sft')
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--out-dir',default='dataset/v16_character_dynamic_sft')
    p.add_argument('--train-sources',type=int,default=4000)
    p.add_argument('--val-sources',type=int,default=300)
    p.add_argument('--variants',type=int,default=1,help='Substituted records per original; 1 gives 50:50 original/substituted')
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--seed',type=int,default=20261105)
    a=p.parse_args()
    if min(a.train_sources,a.val_sources,a.variants) < 1: raise ValueError('counts must be positive')
    tok = Tokenizer.from_file(a.tokenizer)
    root = Path(a.source_dir)
    train_src=torch.load(root/'train_sft.pt',map_location='cpu',weights_only=True)
    val_src=torch.load(root/'val_sft.pt',map_location='cpu',weights_only=True)
    # Source splits are inherited. No new book/source-level disjointness is claimed.
    names=make_names(); rng=random.Random(a.seed); rng.shuffle(names)
    train_names=names[:320]
    val_names=list(HELDOUT) + names[320:]
    assert not set(train_names)&set(val_names)
    train,tm,ts=build_split(tok,train_src,a.train_sources,a.variants,train_names,a.max_tokens,a.seed)
    val,vm,vs=build_split(tok,val_src,a.val_sources,a.variants,val_names,a.max_tokens,a.seed+1)
    train_source_ids={x['source_sha256'] for x in tm}
    val_source_ids={x['source_sha256'] for x in vm}
    overlap=sorted(train_source_ids & val_source_ids)
    if overlap: raise RuntimeError(f'Duplicate passage across train and val: {len(overlap)}')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save(train,out/'train_sft.pt')
    torch.save(val,out/'val_sft.pt')
    report={'train':len(train),'val':len(val), 'train_stats':ts,'val_stats':vs,
            'train_name_count':len(train_names),'validation_name_count':len(val_names),
            'heldout_test_names':list(HELDOUT),'train_val_name_overlap':len(set(train_names)&set(val_names)),
            'exact_source_passage_overlap':len(overlap), 'source_split_note':'Inherited from continuation_sft; book-level disjointness not verified',
            'limitations':['Real-fiction character labels selected by heuristics, not human-verified',
                           'Original passages can contain other characters; substitution targets two only',
                           'No guarantee of prose coherence or unseen-name generalization']}
    (out/'build_report_dynamic.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (out/'substitution_manifest.json').write_text(json.dumps({'train':tm,'val':vm},indent=2),encoding='utf8')
    print('DYNAMIC_CHARACTER_SFT_BUILT',json.dumps(report,indent=2))

if __name__=='__main__': main()
